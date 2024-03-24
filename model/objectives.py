import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

# def compute_sdm(image_fetures, text_fetures, pids, logit_scale, image_ids, dic_pid_imageGToken, dic_pid_textGToken, dic_pid_imageId, factor=0.3, epsilon=1e-8):
#     labels_pids = pids.clone()
#     for pid in pids:
#         int_pid = pid.item()
#         if int_pid in dic_pid_imageGToken:
#             for index, image_gToken in enumerate(dic_pid_imageGToken[int_pid]):
#                 labels_pids = torch.concat([labels_pids, pid.unsqueeze(0)], dim=0)
#                 image_fetures = torch.concat([image_fetures, dic_pid_imageGToken[int_pid][index].unsqueeze(0)], dim=0)
#                 text_fetures = torch.concat([text_fetures, dic_pid_textGToken[int_pid][index].unsqueeze(0)], dim=0)
#                 image_ids = torch.concat([image_ids, dic_pid_imageId[int_pid][index].unsqueeze(0)], dim=0)
#
#     labels_pids = labels_pids.reshape((-1, 1)) # make sure pid size is [batch_size, 1]
#     pid_dist = labels_pids - labels_pids.t()
#     labels = (pid_dist == 0).float()
#
#     if image_ids != None:
#         # print("Mix PID and ImageID to create soft label.")
#         image_ids = image_ids.reshape((-1, 1))
#         image_id_dist = image_ids - image_ids.t()
#         image_id_mask = (image_id_dist == 0).float()
#         labels = (labels - image_id_mask) * factor + image_id_mask
#         # labels = (labels + image_id_mask) / 2
#
#     image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
#     text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)
#
#     t2i_cosine_theta = text_norm @ image_norm.t()
#     i2t_cosine_theta = t2i_cosine_theta.t()
#
#     text_proj_image = logit_scale * t2i_cosine_theta
#     image_proj_text = logit_scale * i2t_cosine_theta
#
#     # normalize the true matching distribution
#     labels_distribute = labels / labels.sum(dim=1)
#
#     i2t_pred = F.softmax(image_proj_text, dim=1)
#     i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
#     t2i_pred = F.softmax(text_proj_image, dim=1)
#     t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))
#
#     loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
#
#     return loss


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i + loss_t)/2

    return loss


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss


# def compute_patch(image_feats, text_feats, caption_ids, pid, logit_scale):
#     mask_text = (caption_ids > 0).float()
#     # 去除句子的开始和结束token
#     mask_text[:, 0] = 0
#     for i, column in enumerate(mask_text):
#         index = column.nonzero()
#         mask_text[i, index[-1]] = 0
#     mask_text = mask_text.unsqueeze(1)
#
#     image_features = image_feats[:, 1:, :]
#     text_features = text_feats
#
#     image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
#     text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
#
#     scores_i2t_array = []
#     for i, image_feature in enumerate(image_features):
#         logits_image2text = torch.matmul(image_feature, text_features.transpose(-1, -2)) * mask_text * logit_scale
#         logits_image_mean = torch.mean(torch.max(logits_image2text, -1)[0], -1)
#         scores_i2t_array.append(logits_image_mean)
#     scores_i2t = torch.stack(scores_i2t_array, 0)
#
#     batch_size = image_feats.shape[0]
#     pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
#     pid_dist = pid - pid.t()
#     labels = (pid_dist == 0).float()
#
#     # normalize the true matching distribution
#     labels_distribute = labels / labels.sum(dim=1)
#
#     i2t_pred = F.softmax(scores_i2t, dim=1)
#     i2t_loss = i2t_pred * (F.log_softmax(scores_i2t, dim=1) - torch.log(labels_distribute + 1e-8))
#
#     loss = torch.mean(torch.sum(i2t_loss, dim=1))
#     return loss

def compute_patch(image_feats, text_feats, caption_ids, pid, logit_scale):
    mask_text = (caption_ids > 0).float()
    # 去除句子的开始和结束token
    mask_text[:, 0] = 0
    for i, column in enumerate(mask_text):
        index = column.nonzero()
        mask_text[i, index[-1]] = 0
    mask_text = (mask_text.unsqueeze(-1)).half()

    image_feats = image_feats / image_feats.norm(p=2, dim=-1, keepdim=True)
    text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)

    image_features_patch = image_feats[:, 1:, :]
    text_features_patch = text_feats * mask_text

    # scores_i2t_array = []
    # for i, image_feature in enumerate(image_features_patch):
    #     logits_image2text = torch.matmul(image_feature, text_features_patch.transpose(-1, -2)) * logit_scale
    #     logits_image_mean = torch.mean(torch.max(logits_image2text, -1)[0], -1)
    #     scores_i2t_array.append(logits_image_mean)
    # scores_i2t = torch.stack(scores_i2t_array, 0)

    scores_t2i_array = []
    for i, text_feature in enumerate(text_features_patch):
        logits_text2image = torch.matmul(text_feature, image_features_patch.transpose(-1, -2)) * logit_scale
        max_text2image = torch.max(logits_text2image, -1)[0]
        sum_text2image = torch.sum(max_text2image, dim=-1)
        count_max_nonzero = torch.count_nonzero(max_text2image, dim=-1)
        logits_text_mean = sum_text2image / count_max_nonzero
        # logits_text_mean = torch.mean(torch.max(logits_text2image, -1)[0], -1)
        scores_t2i_array.append(logits_text_mean)
    scores_t2i = torch.stack(scores_t2i_array, 0)

    batch_size = image_feats.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    # i2t_pred = F.softmax(scores_i2t, dim=1)
    # i2t_loss = i2t_pred * (F.log_softmax(scores_i2t, dim=1) - torch.log(labels_distribute + 1e-8))
    t2i_pred = F.softmax(scores_t2i, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(scores_t2i, dim=1) - torch.log(labels_distribute + 1e-8))

    # loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
    loss = torch.mean(torch.sum(t2i_loss, dim=1))
    return loss


def compute_global(image_features, text_features, pid, logit_scale):
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logit
    logit_per_image = logit_scale * image_norm @ text_norm.t()

    scores_i2t = logit_per_image.softmax(dim=-1)
    loss = calculate_triplet_loss(scores_i2t, pid)
    return loss


def calculate_triplet_loss(scores_i2t, labels):
    batch_size = len(labels)
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()

    # anchor
    image_anchor = torch.ones((batch_size, batch_size)).to(scores_i2t.device)

    # positive
    distance_i2t = image_anchor - scores_i2t
    labels_mask_positive = (labels_dist == 0).float()
    scores_i2t_positive = distance_i2t * labels_mask_positive + 1e-15
    text_positive_index = torch.multinomial(scores_i2t_positive, num_samples=batch_size, replacement=True)
    text_positive = scores_i2t.gather(1, text_positive_index)

    # negative
    labels_mask_negative = (labels_dist != 0).float()
    scores_i2t_negative = scores_i2t * labels_mask_negative + 1e-15
    text_negative_index = torch.multinomial(scores_i2t_negative, num_samples=batch_size, replacement=True)
    text_negative = scores_i2t.gather(1, text_negative_index)

    triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: x - y, margin=1, reduction='mean')
    loss = triplet_loss(image_anchor, text_positive, text_negative)
    return loss


def compute_gitm(i_feats, t_feats, pid, logit_scale, mlp_itm):
    # normalized features
    image_norm = i_feats / i_feats.norm(dim=-1, keepdim=True)
    text_norm = t_feats / t_feats.norm(dim=-1, keepdim=True)

    # positive
    positive_features = torch.concat([image_norm, text_norm], dim=-1)

    # negative
    scores_i2t = F.softmax(logit_scale * image_norm @ text_norm.t(), dim=1) + 1e-5

    batch_size = i_feats.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist != 0).float()

    negative_text_features_array = []
    scores_i2t_negative = scores_i2t * labels
    for index, scores in enumerate(scores_i2t_negative):
        negative_idx = torch.multinomial(scores, 1).item()
        negative_text_features_array.append(t_feats[negative_idx])
    negative_text_features = torch.stack(negative_text_features_array, dim=0)
    negative_features = torch.concat([image_norm, negative_text_features], dim=-1)

    input = torch.concat([positive_features, negative_features])
    output = mlp_itm(input.half())
    gitm_labels = torch.cat([torch.ones(batch_size, dtype=torch.long), torch.zeros(batch_size, dtype=torch.long)]).to(i_feats.device)
    gitm_loss = F.cross_entropy(output, gitm_labels)
    return gitm_loss


def compute_gtm(image_features, text_features, logit_scale):
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    image_norm = image_norm * logit_scale
    text_norm = text_norm * logit_scale

    image_pred = F.softmax(image_norm, dim=1)
    text_pred = F.softmax(text_norm, dim=1)
    i2t_loss = image_pred * (F.log_softmax(image_norm, dim=1) - F.log_softmax(text_norm, dim=1))
    t2i_loss = text_pred * (F.log_softmax(text_norm, dim=1) - F.log_softmax(image_norm, dim=1))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
    return 0.1 * loss


# def compute_iikl(image_fetures, pid, logit_scale, epsilon=1e-8):
#     batch_size = image_fetures.shape[0]
#     pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
#     pid_dist = pid - pid.t()
#     labels = (pid_dist == 0).float()
#
#     image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
#
#     i2i_cosine_theta = image_norm @ image_norm.t()
#
#     image_proj_image = logit_scale * i2i_cosine_theta
#
#     # normalize the true matching distribution
#     labels_distribute = labels / labels.sum(dim=1)
#
#     i2i_pred = F.softmax(image_proj_image, dim=1)
#     i2i_loss = i2i_pred * (F.log_softmax(image_proj_image, dim=1) - torch.log(labels_distribute + epsilon))
#
#     loss = torch.mean(torch.sum(i2i_loss, dim=1))
#
#     return loss

def compute_iikl(image_fetures, pids, logit_scale, dic_pid_imageGToken, epsilon=1e-8):
    labels_pids = pids.clone()
    for pid in pids:
        int_pid = pid.item()
        if int_pid in dic_pid_imageGToken:
            for image_gToken in dic_pid_imageGToken[int_pid]:
                labels_pids = torch.concat([labels_pids, pid.unsqueeze(0)], dim=0)
                image_fetures = torch.concat([image_fetures, image_gToken.unsqueeze(0)], dim=0)


    labels_pids = labels_pids.reshape((-1, 1))  # make sure pid size is [batch_size, 1]
    pids_dist = labels_pids - labels_pids.t()
    labels = (pids_dist == 0).float()

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)

    i2i_cosine_theta = image_norm @ image_norm.t()

    image_proj_image = logit_scale * i2i_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2i_pred = F.softmax(image_proj_image, dim=1)
    i2i_loss = i2i_pred * (F.log_softmax(image_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2i_loss, dim=1))

    return loss
