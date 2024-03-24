from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k
    # print(pred_labels[50:60, 0:10])

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("IRRA.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qfeats, gfeats, qids, gids
    
    def eval(self, model, i2t_metric=False):

        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        similarity = qfeats @ gfeats.t()

        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))
        
        return t2i_cmc[0]

    def eval_nafs(self, model):
        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        qfeats = F.normalize(qfeats, p=2, dim=1)  # text features
        gfeats = F.normalize(gfeats, p=2, dim=1)  # image features

        # t2i
        unique = []
        set_id = set()
        for id in qids:
            i_id = id.item()
            if i_id in set_id:
                unique.append(0)
            else:
                unique.append(1)
            set_id.add(i_id)
        unique_text = torch.tensor(unique) == 1

        query_global = qfeats[unique_text]
        gallery_global = gfeats
        sim_cosine_global_t2i = torch.matmul(query_global, gallery_global.t())

        reid_sim = torch.matmul(query_global, query_global.t())
        global_result = topk(sim_cosine_global_t2i.cpu(), gids, qids[unique_text], reid_sim=reid_sim)
        # global_result = topk(sim_cosine_global_t2i.cpu(), gids, qids[unique_text])
        print('********t2i', global_result)

        # i2t
        # unique = []
        # set_id = set()
        # for id in gids:
        #     i_id = id.item()
        #     if i_id in set_id:
        #         unique.append(0)
        #     else:
        #         unique.append(1)
        #     set_id.add(i_id)
        # unique_image = torch.tensor(unique) == 1
        #
        # query_global = gfeats[unique_image]
        # gallery_global = qfeats
        # sim_cosine_global_i2t = torch.matmul(query_global, gallery_global.t())
        #
        # # reid_sim = torch.matmul(query_global, query_global.t())
        # # global_result = topk(sim_cosine_global_i2t.cpu(), qids, gids[unique_image], reid_sim=reid_sim)
        # global_result = topk(sim_cosine_global_i2t.cpu(), qids, gids[unique_image])
        # print('********i2t', global_result)

        return global_result


def jaccard(a_list, b_list):
    return 1.0 - float(len(set(a_list) & set(b_list))) / float(len(set(a_list) | set(b_list))) * 1.0


def topk(sim, target_gallery, target_query, k=[1, 5, 10], dim=1, print_index=False, reid_sim=None):
    result = []
    maxk = max(k)
    size_total = len(target_query)
    if reid_sim is None:
        _, pred_index = sim.topk(maxk, dim, True, True)
        pred_labels = target_gallery[pred_index]
    else:
        K = 5
        sim = sim.cpu().numpy()
        reid_sim = reid_sim.cpu().numpy()
        pred_index = np.argsort(-sim, axis=1)
        reid_pred_index = np.argsort(-reid_sim, axis=1)

        q_knn = pred_index[:, 0:K]
        g_knn = reid_pred_index[:, 0:K]

        jaccard_dist = np.zeros_like(sim)
        for i, qq in enumerate(q_knn):
            for j, gg in enumerate(g_knn):
                jaccard_dist[i, j] = 1.0 - jaccard(qq, gg)
        _, pred_index = torch.Tensor(sim + jaccard_dist).topk(maxk, dim, True, True)
        pred_labels = target_gallery[pred_index]

    # pred
    if dim == 1:
        pred_labels = pred_labels.t()

    correct = pred_labels.eq(target_query.view(1, -1).expand_as(pred_labels))
    for topk in k:
        # correct_k = torch.sum(correct[:topk]).float()
        correct_k = torch.sum(correct[:topk], dim=0)
        correct_k = torch.sum(correct_k > 0).float()
        result.append(correct_k * 100 / size_total)
    return result
