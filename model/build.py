from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class VFE_TPS(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        if 'itm' in args.loss_names:
            self.itm_depth = args.itm_depth
            if self.itm_depth > 0:
                self.cross_attention_itm, self.cross_modal_transformer_itm, self.itm_ln_pre_t, self.itm_ln_pre_i, self.itm_ln_post = self.get_cross(self.itm_depth)

            self.mlp_itm = nn.Linear(self.embed_dim, 2)
            nn.init.normal_(self.mlp_itm.weight.data, std=0.001)
            nn.init.constant_(self.mlp_itm.bias.data, val=0.0)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

        if 'mim' in args.loss_names:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, base_cfg['vision_width']))

            self.cross_attention_image, self.cross_modal_transformer_image, self.image_ln_pre_t, self.image_ln_pre_i, self.image_ln_post = self.get_cross(args.mim_depth)

            self.patch_size = base_cfg['vision_patch_size']
            self.decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.embed_dim,
                    out_channels=self.patch_size ** 2 * 3, kernel_size=1),
                nn.PixelShuffle(self.patch_size),
            )

            # init decoder
            for m in self.decoder:
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()

        if 'g2id' in args.loss_names:
            self.gToken2id = nn.Linear(2 * self.embed_dim, self.num_classes)
            nn.init.normal_(self.gToken2id.weight.data, std=0.001)
            nn.init.constant_(self.gToken2id.bias.data, val=0.0)

        self.dic_pid_imageGToken = {}  # key:pid, value:[imageGToken, imageGToken]
        # self.dic_pid_textGToken = {}  # key:pid, value:[textGToken, textGToken]
        # self.dic_pid_imageId = {}  # key:pid, value:[imageId, imageId]

    def clear_dic(self):
        self.dic_pid_imageGToken.clear()
        # self.dic_pid_textGToken.clear()
        # self.dic_pid_imageId.clear()

    def get_cross(self, layers):
        cross_attention = nn.MultiheadAttention(self.embed_dim, self.embed_dim // 64, batch_first=True)

        cross_modal_transformer = Transformer(width=self.embed_dim, layers=layers, heads=self.embed_dim // 64)

        scale = cross_modal_transformer.width ** -0.5
        proj_std = scale * ((2 * cross_modal_transformer.layers) ** -0.5)
        attn_std = scale
        fc_std = (2 * cross_modal_transformer.width) ** -0.5
        # init cross_modal_transformer
        for block in cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # init cross_attention
        nn.init.normal_(cross_attention.in_proj_weight, std=attn_std)
        nn.init.normal_(cross_attention.out_proj.weight, std=proj_std)

        ln_pre_t = LayerNorm(self.embed_dim)
        ln_pre_i = LayerNorm(self.embed_dim)
        ln_post = LayerNorm(self.embed_dim)

        return cross_attention, cross_modal_transformer, ln_pre_t, ln_pre_i, ln_post

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def cross_former_compute(self, q, k, v, cross_attention, cross_modal_transformer, ln_pre_t, ln_pre_i, ln_post):
        x = cross_attention(
                ln_pre_i(q),
                ln_pre_t(k),
                ln_pre_t(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            ret.update({'itc_loss': objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss': objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})
            # ret.update({'sdm_loss': objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale, batch['image_ids'])})
            # ret.update({'sdm_loss': objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale, batch['image_ids'],
            #                                                self.dic_pid_imageGToken, self.dic_pid_textGToken, self.dic_pid_imageId)})

        if 'patch' in self.current_task:
            # ret.update({'patch_loss': objectives.compute_patch(image_feats, text_feats, caption_ids, logit_scale)})
            ret.update({'patch_loss': objectives.compute_patch(image_feats, text_feats, caption_ids, batch['pids'], logit_scale)})

        if 'global' in self.current_task:
            ret.update({'global_loss': objectives.compute_global(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'gitm' in self.current_task:
            # ret.update({'itm_loss': objectives.compute_gitm(i_feats, t_feats, batch['pids'], logit_scale, self.mlp_itm)})

            # normalized features
            image_norm = i_feats / i_feats.norm(dim=-1, keepdim=True)
            text_norm = t_feats / t_feats.norm(dim=-1, keepdim=True)

            # negative
            scores_i2t = F.softmax(logit_scale * image_norm @ text_norm.t(), dim=1) + 1e-5

            batch_size = i_feats.shape[0]
            pid = batch['pids'].reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
            pid_dist = pid - pid.t()
            labels = (pid_dist != 0).float()

            negative_text_features_array = []
            scores_i2t_negative = scores_i2t * labels
            for index, scores in enumerate(scores_i2t_negative):
                negative_idx = torch.multinomial(scores, 1).item()
                negative_text_features_array.append(t_feats[negative_idx])
            negative_text_features = torch.stack(negative_text_features_array, dim=0)

            if self.itm_depth > 0:
                t_features = torch.cat([text_norm, negative_text_features])
                i_features = torch.cat([image_norm, image_norm])
                z = self.cross_former_compute(t_features.unsqueeze(1).half(), i_features.unsqueeze(1).half(), i_features.unsqueeze(1).half(),
                                              self.cross_attention_itm, self.cross_modal_transformer_itm,
                                              self.itm_ln_pre_t, self.itm_ln_pre_i, self.itm_ln_post)
                z = z.squeeze(1)
            else:
                positive_features = image_norm + text_norm
                negative_features = image_norm + negative_text_features
                z = torch.cat([positive_features, negative_features])
            output = self.mlp_itm(z.half())
            gitm_labels = torch.cat([torch.ones(batch_size, dtype=torch.long), torch.zeros(batch_size, dtype=torch.long)]).to(i_feats.device)
            gitm_loss = F.cross_entropy(output, gitm_labels)
            ret.update({'gitm_loss': gitm_loss})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss': objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss': objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        if 'mim' in self.current_task:
            visionTransformer = self.base_model.visual
            x = visionTransformer.conv1(images.half())  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

            B, L, _ = x.shape

            mask_token = self.mask_token.expand(B, L, -1)
            mask = batch['mim_mask']
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

            x = torch.cat([visionTransformer.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                          device=x.device), x],
                          dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + visionTransformer.positional_embedding.to(x.dtype)
            x = visionTransformer.ln_pre(x)

            x = x.permute(1, 0, 2).half()  # NLD -> LND
            x = visionTransformer.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD

            mim_feats = visionTransformer.ln_post(x)

            if visionTransformer.proj is not None:
                mim_feats = mim_feats @ visionTransformer.proj

            # z = self.cross_former_compute(mim_feats, text_feats, text_feats,
            #                               self.cross_attention_image, self.cross_modal_transformer_image,
            #                               self.image_ln_pre_t, self.image_ln_pre_i, self.image_ln_post)
            z = self.cross_former_compute(mim_feats, mim_feats, mim_feats,
                                          self.cross_attention_image, self.cross_modal_transformer_image,
                                          self.image_ln_pre_t, self.image_ln_pre_i, self.image_ln_post)
            z = z[:, 1:]
            B, L, C = z.shape
            H = 24
            W = 8
            z = z.permute(0, 2, 1).reshape(B, C, H, W)

            x_rec = self.decoder(z)

            mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(
                1).contiguous()
            loss_recon = F.l1_loss(images, x_rec, reduction='none')
            loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / 3
            ret.update({'mim_loss': 10 * loss})

        if 'gtm' in self.current_task:
            ret.update({'gtm_loss': objectives.compute_gtm(i_feats, t_feats, logit_scale)})

        if 'iikl' in self.current_task:
            ret.update({'iikl_loss': objectives.compute_iikl(i_feats, batch['pids'], logit_scale, self.dic_pid_imageGToken)})

        if 'g2id' in self.current_task:
            it_feats = torch.concat([i_feats, t_feats], -1)
            it_logits = self.gToken2id(it_feats.half()).float()
            crossEntropyLoss = nn.CrossEntropyLoss(reduction="mean")
            g2id_loss = crossEntropyLoss(it_logits, batch['pids'])
            ret.update({'g2id_loss': g2id_loss})

        if 'gi2id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            crossEntropyLoss = nn.CrossEntropyLoss(reduction="mean")
            gi2id_loss = crossEntropyLoss(image_logits, batch['pids'])
            ret.update({'gi2id_loss': gi2id_loss})

        # update dic
        pids = batch['pids']
        # image_ids = batch['image_ids']
        for index, pid in enumerate(pids):
            int_pid = pid.item()
            if int_pid not in self.dic_pid_imageGToken:
                self.dic_pid_imageGToken[int_pid] = [i_feats[index].clone().detach()]
                # self.dic_pid_textGToken[int_pid] = [t_feats[index].clone().detach()]
                # self.dic_pid_imageId[int_pid] = [image_ids[index]]
            else:
                self.dic_pid_imageGToken[int_pid].append(i_feats[index].clone().detach())
                # self.dic_pid_textGToken[int_pid].append(t_feats[index].clone().detach())
                # self.dic_pid_imageId[int_pid].append(image_ids[index])

        return ret


def build_model(args, num_classes=11003):
    model = VFE_TPS(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
