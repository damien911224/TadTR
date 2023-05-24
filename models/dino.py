# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms

from opts import cfg

# from util import box_ops
from util import segment_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized, inverse_sigmoid)

# from .backbone import build_backbone
from .matcher import build_matcher
# from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
#                            dice_loss)
from .custom_loss import sigmoid_focal_loss
# from .deformable_transformer import build_deformable_transformer
# from .utils import sigmoid_focal_loss, MLP

# from ..registry import MODULE_BUILD_FUNCS
from .dn_components import prepare_for_cdn, dn_post_process

from models.position_encoding import build_position_encoding
from .dab_transformer import build_transformer


class DINO(nn.Module):
    """ This is the Cross-Attention Detector module that performs object detection """

    def __init__(self, position_embedding, transformer, num_classes, num_queries,
                 aux_loss=True, iter_update=True,
                 query_dim=2,
                 random_refpoints_xy=False,
                 fix_refpoints_hw=-1,
                 num_feature_levels=1,
                 nheads=8,
                 # two stage
                 two_stage_type='no',  # ['no', 'standard']
                 two_stage_add_query_num=0,
                 dec_pred_class_embed_share=True,
                 dec_pred_bbox_embed_share=True,
                 two_stage_class_embed_share=True,
                 two_stage_bbox_embed_share=True,
                 decoder_sa_type='sa',
                 num_patterns=0,
                 dn_number=100,
                 dn_box_noise_scale=0.4,
                 dn_label_noise_ratio=0.5,
                 dn_labelbook_size=100,
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.

            fix_refpoints_hw: -1(default): learn w and h for each box seperately
                                >0 : given fixed number
                                -2 : learn a shared w and h
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.segment_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        # setting query dim
        self.query_dim = query_dim
        # assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        self.refpoint_embed = nn.Embedding(num_queries, query_dim)

        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(2048, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )])
        self.position_embedding = position_embedding

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.segment_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.segment_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # num_pred = transformer.decoder.num_layers
        if iter_update:
            # hack implementation for segment refinement
            self.transformer.decoder.segment_embed = self.segment_embed

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # # prepare input projection layers
        # if num_feature_levels > 1:
        #     num_backbone_outs = len(backbone.num_channels)
        #     input_proj_list = []
        #     for _ in range(num_backbone_outs):
        #         in_channels = backbone.num_channels[_]
        #         input_proj_list.append(nn.Sequential(
        #             nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
        #             nn.GroupNorm(32, hidden_dim),
        #         ))
        #     for _ in range(num_feature_levels - num_backbone_outs):
        #         input_proj_list.append(nn.Sequential(
        #             nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
        #             nn.GroupNorm(32, hidden_dim),
        #         ))
        #         in_channels = hidden_dim
        #     self.input_proj = nn.ModuleList(input_proj_list)
        # else:
        #     assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
        #     self.input_proj = nn.ModuleList([
        #         nn.Sequential(
        #             nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
        #             nn.GroupNorm(32, hidden_dim),
        #         )])

        # self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = None

        # self.iter_update = iter_update
        # assert iter_update, "Why not iter_update?"
        # # prepare pred layers
        # self.dec_pred_class_embed_share = dec_pred_class_embed_share
        # self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # # prepare class & box embed
        # _class_embed = nn.Linear(hidden_dim, num_classes)
        # _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # # init the two embed layers
        # prior_prob = 0.01
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        # nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        # nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        #
        # if dec_pred_bbox_embed_share:
        #     box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        # else:
        #     box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)]
        # if dec_pred_class_embed_share:
        #     class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        # else:
        #     class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers)]
        # self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        # self.class_embed = nn.ModuleList(class_embed_layerlist)
        # self.transformer.decoder.segment_embed = self.bbox_embed
        # self.transformer.decoder.class_embed = self.class_embed

        # # two stage
        # self.two_stage_type = two_stage_type
        # self.two_stage_add_query_num = two_stage_add_query_num
        # assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        # if two_stage_type != 'no':
        #     if two_stage_bbox_embed_share:
        #         assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
        #         self.transformer.enc_out_bbox_embed = _bbox_embed
        #     else:
        #         self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
        #
        #     if two_stage_class_embed_share:
        #         assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
        #         self.transformer.enc_out_class_embed = _class_embed
        #     else:
        #         self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)
        #
        #     self.refpoint_embed = None
        #     if self.two_stage_add_query_num > 0:
        #         self.init_ref_points(two_stage_add_query_num)
        #
        # self.decoder_sa_type = decoder_sa_type
        # assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']
        # if decoder_sa_type == 'ca_label':
        #     self.label_embedding = nn.Embedding(num_classes, hidden_dim)
        #     for layer in self.transformer.decoder.layers:
        #         layer.label_embedding = self.label_embedding
        # else:
        #     for layer in self.transformer.decoder.layers:
        #         layer.label_embedding = None
        #     self.label_embedding = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)
        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if self.fix_refpoints_hw > 0:
            print("fix_refpoints_hw: {}".format(self.fix_refpoints_hw))
            assert self.random_refpoints_xy
            self.refpoint_embed.weight.data[:, 2:] = self.fix_refpoints_hw
            self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(self.refpoint_embed.weight.data[:, 2:])
            self.refpoint_embed.weight.data[:, 2:].requires_grad = False
        elif int(self.fix_refpoints_hw) == -1:
            pass
        elif int(self.fix_refpoints_hw) == -2:
            print('learn a shared h and w')
            assert self.random_refpoints_xy
            self.refpoint_embed = nn.Embedding(use_num_queries, 2)
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
            self.hw_embed = nn.Embedding(1, 1)
        else:
            raise NotImplementedError('Unknown fix_refpoints_hw {}'.format(self.fix_refpoints_hw))

    def forward(self, samples, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            if isinstance(samples, (list, tuple)):
                samples = NestedTensor(*samples)
            else:
                samples = nested_tensor_from_tensor_list(samples)  # (n, c, t)

        pos = [self.position_embedding(samples)]
        src, mask = samples.tensors, samples.mask

        if self.dn_number > 0 or targets is not None:
            input_query_label, input_query_bbox, attn_mask, dn_meta = \
                prepare_for_cdn(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
                                training=self.training, num_queries=self.num_queries, num_classes=self.num_classes,
                                hidden_dim=self.hidden_dim, label_enc=self.label_enc)
        else:
            assert targets is None
            input_query_label = input_query_bbox = attn_mask = dn_meta = None

        bs, c, w = src.shape
        refpoint_embed = self.refpoint_embed.weight
        refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros(self.num_queries, bs, self.hidden_dim, device=refpoint_embed.device)

        if input_query_bbox is not None:
            input_query_label = input_query_label.transpose(0, 1)
            input_query_bbox = input_query_bbox.transpose(0, 1)

            input_query_bbox = torch.cat((input_query_bbox, refpoint_embed), dim=0)
            input_query_label = torch.cat((input_query_label, tgt), dim=0)
        else:
            input_query_bbox = refpoint_embed
            input_query_label = tgt

        hs, reference, memory, Q_weights, K_weights, C_weights = \
            self.transformer(self.input_proj[0](src), mask, input_query_bbox, pos[-1],
                             tgt=input_query_label, attn_mask=attn_mask)

        Q_weights = Q_weights[:, :, dn_meta["pad_size"]:, dn_meta["pad_size"]:]
        K_weights = K_weights[:, :, dn_meta["pad_size"]:, dn_meta["pad_size"]:]
        C_weights = C_weights[:, :, dn_meta["pad_size"]:, dn_meta["pad_size"]:]

        # In case num object=0
        hs[0] += self.label_enc.weight[0, 0] * 0.0

        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.segment_embed(hs)
        tmp[..., :self.query_dim] += reference_before_sigmoid
        outputs_coord = tmp.sigmoid()
        outputs_class = self.class_embed(hs)

        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord = \
                dn_post_process(outputs_class, outputs_coord,
                                dn_meta, self.aux_loss, self._set_aux_loss)

        out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_coord[-1],
               'Q_weights': Q_weights, 'K_weights': K_weights, 'C_weights': C_weights}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        # for encoder output
        # if hs_enc is not None:
        #     # prepare intermediate outputs
        #     interm_coord = ref_enc[-1]
        #     interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
        #     out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
        #     out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}
        #
        #     # prepare enc outputs
        #     if hs_enc.shape[0] > 1:
        #         enc_outputs_coord = []
        #         enc_outputs_class = []
        #         for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc) in enumerate(
        #                 zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc[:-1], ref_enc[:-1])):
        #             layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
        #             layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
        #             layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()
        #
        #             layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
        #             enc_outputs_coord.append(layer_enc_outputs_coord)
        #             enc_outputs_class.append(layer_enc_outputs_class)
        #
        #         out['enc_outputs'] = [
        #             {'pred_logits': a, 'pred_boxes': b} for a, b in zip(enc_outputs_class, enc_outputs_coord)
        #         ]

        out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_segments': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_segments, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_segments]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)

        src_segments = outputs['pred_segments'][idx]
        target_segments = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        IoUs = segment_ops.segment_iou(segment_ops.segment_cw_to_t1t2(src_segments),
                                       segment_ops.segment_cw_to_t1t2(target_segments))
        IoUs = torch.diag(IoUs).detach()

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot[idx] = target_classes_onehot[idx] * IoUs.unsqueeze(-1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_segments, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]  # nq
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

            # N, Q
            probs = torch.max(src_logits.sigmoid(), dim=-1)[0]
            top_k_indices = torch.argsort(-probs, dim=-1)
            top_1_indices = top_k_indices[..., 0]
            top_2_indices = top_k_indices[..., 1]
            # score_gap = torch.mean(probs[top_1_indices] - probs[top_2_indices], dim=0)
            score_gap = torch.mean(probs[torch.arange(len(top_1_indices)),
                                         top_1_indices[torch.arange(len(top_1_indices))]] -
                                   probs[torch.arange(len(top_2_indices)),
                                         top_2_indices[torch.arange(len(top_2_indices))]], dim=0)

            losses['score_gap'] = score_gap

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_segments):
        """Compute the losses related to the segmentes, the L1 regression loss and the IoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'pred_segments' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_segments = outputs['pred_segments'][idx]
        target_segments = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_segment = F.l1_loss(src_segments, target_segments, reduction='none')

        losses = {}
        losses['loss_segments'] = loss_segment.sum() / num_segments

        loss_iou = 1 - torch.diag(segment_ops.segment_iou(
            segment_ops.segment_cw_to_t1t2(src_segments),
            segment_ops.segment_cw_to_t1t2(target_segments)))
        losses['loss_iou'] = loss_iou.sum() / num_segments
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_QQ(self, outputs, targets, indices, num_segments):
        """Compute the actionness regression loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'Q_weights' in outputs
        assert 'C_weights' in outputs

        # Q_weights = torch.mean(outputs["Q_weights"], dim=0)

        # Q_weights = outputs["Q_weights"]
        # normalized_Q_weights = Q_weights[0]
        # for i in range(len(Q_weights) - 1):
        #     normalized_Q_weights = torch.sqrt(
        #         torch.bmm(normalized_Q_weights, Q_weights[i + 1].transpose(1, 2)) + 1.0e-7)
        #     normalized_Q_weights = normalized_Q_weights / torch.sum(normalized_Q_weights, dim=-1, keepdim=True)
        # Q_weights = normalized_Q_weights

        # L, N, Q, Q = outputs["Q_weights"].shape
        Q_weights = outputs["Q_weights"].flatten(0, 1)

        # src_segments = outputs['pred_segments'].detach()
        # IoUs = list()
        # for n_i in range(len(targets)):
        #     tgt_segments = targets[n_i]['segments']
        #     this_IoU = segment_ops.segment_iou(segment_ops.segment_cw_to_t1t2(src_segments[n_i]),
        #                                        segment_ops.segment_cw_to_t1t2(tgt_segments))
        #     this_IoU = torch.max(this_IoU, dim=1)[0]
        #     IoUs.append(this_IoU)
        # IoUs = torch.stack(IoUs)
        # # IoUs = IoUs - torch.min(IoUs, dim=-1)[0].unsqueeze(-1) + 0.05
        # # IoUs = IoUs / torch.max(IoUs, dim=-1)[0].unsqueeze(-1)
        # # IoUs = torch.clamp(IoUs, min=0.10)
        # IoUs = IoUs.detach()
        #
        # max_IoU = torch.max(IoUs, dim=-1)[0]
        # max_IoU = torch.clamp(max_IoU / 0.5, max=1.0)

        # C_weights = C_weights * IoUs.unsqueeze(-1)
        # C_weights = C_weights / torch.sum(C_weights, dim=-1, keepdim=True)

        # iou_mat = segment_ops.segment_iou(segment_ops.segment_cw_to_t1t2(src_segments), target_segments[..., :2])
        # gt_iou = iou_mat.max(dim=1)[0]
        # scores = gt_iou.view(src_logits.shape[:2]).detach().cpu()

        # valid_masks = list()
        # for n_i, (b, l, s) in enumerate(zip(boxes, labels, scores)):
        #     # 2: batched nms (only implemented on CPU)
        #     nms_indices = dynamic_nms(
        #         b.contiguous(), s.contiguous(), l.contiguous(),
        #         iou_threshold=0.70,
        #         min_score=0.0,
        #         max_seg_num=1000,
        #         use_soft_nms=False,
        #         multiclass=False,
        #         sigma=0.75,
        #         voting_thresh=0.0)
        #     valid_mask = torch.isin(torch.arange(len(b)), nms_indices).float()
        #     valid_masks.append(valid_mask)
        # # N, Q, 1
        # valid_masks = torch.stack(valid_masks, dim=0).cuda()

        # N, Q, K = C_weights.shape

        # C_indices = torch.argsort(-C_weights, dim=-1).float()
        # QQ_weights = torch.bmm(C_indices, C_indices.transpose(1, 2))
        # target_Q_weights = F.softmax(QQ_weights, dim=-1)

        # C_weights = F.softmax(C_weights, dim=-1)
        # target_Q_weights = F.log_softmax(QQ_weights, dim=-1)
        # target_Q_weights = F.softmax(QQ_weights * 25.0, dim=-1)

        # # QQ_weights = torch.bmm(C_weights[-1], C_weights[-1].transpose(1, 2))
        # C_weights = torch.mean(outputs["C_weights"], dim=0)
        # QQ_weights = torch.bmm(C_weights, C_weights.transpose(1, 2))
        # QQ_weights = torch.sqrt(QQ_weights + 1.0e-7)
        # target_Q_weights = QQ_weights / torch.sum(QQ_weights, dim=-1, keepdim=True)

        # C_weights = outputs["C_weights"].detach()
        # normalized_QQ_weights = torch.sqrt(torch.bmm(C_weights[0], C_weights[0].transpose(1, 2)) + 1.0e-7)
        # normalized_QQ_weights = normalized_QQ_weights / torch.sum(normalized_QQ_weights, dim=-1, keepdim=True)
        # for i in range(len(C_weights) - 1):
        #     QQ_weights = torch.sqrt(torch.bmm(C_weights[i + 1], C_weights[i + 1].transpose(1, 2)) + 1.0e-7)
        #     QQ_weights = QQ_weights / torch.sum(QQ_weights, dim=-1, keepdim=True)
        #     normalized_QQ_weights = torch.sqrt(torch.bmm(normalized_QQ_weights, QQ_weights) + 1.0e-7)
        #     normalized_QQ_weights = normalized_QQ_weights / torch.sum(normalized_QQ_weights, dim=-1, keepdim=True)
        # target_Q_weights = normalized_QQ_weights

        C_weights = outputs["C_weights"].flatten(0, 1).detach()
        QQ_weights = torch.sqrt(torch.bmm(C_weights, C_weights.transpose(1, 2)) + 1.0e-7)
        target_Q_weights = QQ_weights / torch.sum(QQ_weights, dim=-1, keepdim=True)

        # target_Q_weights = torch.eye(Q).unsqueeze(0).repeat(L * N, 1, 1).to(Q_weights.device)

        # src_C_weights = C_weights.unsqueeze(2).tile(1, 1, Q, 1).flatten(0, 2)
        # src_C_weights = (src_C_weights + 1.0e-7).log()
        # tgt_C_weights = C_weights.unsqueeze(1).tile(1, Q, 1, 1).flatten(0, 2)
        # tgt_C_weights = (tgt_C_weights + 1.0e-7).log()
        # QQ_weights = F.kl_div(src_C_weights, tgt_C_weights, log_target=True, reduction="none").sum(-1)
        # target_Q_weights = F.softmax(QQ_weights.view(N, Q, Q), dim=-1)
        # temparature_scale = (torch.max(C_weights) / torch.max(QQ_weights)).detach()
        # target_Q_weights = F.softmax(QQ_weights * temparature_scale, dim=-1)
        # target_Q_weights = F.log_softmax(QQ_weights * 10000.0, dim=-1)
        # target_Q_weights = F.log_softmax(torch.bmm(torch.log(C_weights),
        #                                            torch.log(C_weights).transpose(1, 2)), dim=-1)

        # print(torch.argsort(-target_Q_weights[0].detach().cpu(), dim=-1)[:10, :10].numpy())
        # print(torch.max(target_Q_weights[0].detach().cpu(), dim=-1)[0][:10].numpy())
        # print(torch.max(C_weights[0].detach().cpu(), dim=-1)[0][:10].numpy())
        # print(target_Q_weights[0, 0].detach().cpu().numpy())
        # print((torch.max(C_weights) - torch.max(target_Q_weights)).detach().cpu().numpy())

        # NQ, Q
        # src_QQ = F.normalize(Q_weights, dim=-1).flatten(0, 1)
        src_QQ = (Q_weights.flatten(0, 1) + 1.0e-7).log()
        # src_QQ = F.log_softmax(Q_weights.flatten(0, 1), -1)
        # NQ, Q
        # tgt_QQ = F.normalize(target_Q_weights, dim=-1).flatten(0, 1)
        tgt_QQ = (target_Q_weights.flatten(0, 1) + 1.0e-7).log()

        losses = {}

        # loss_QQ = 1.0 - torch.bmm(src_QQ.unsqueeze(-1), tgt_QQ.unsqueeze(1))
        # loss_QQ = torch.square(src_QQ - dummy)
        # loss_QQ = torch.sum(-tgt_QQ * torch.log(src_QQ + 1.0e-5), dim=-1)
        # loss_QQ = loss_QQ.sum(dim=(1, 2))
        loss_QQ = F.kl_div(src_QQ, tgt_QQ, log_target=True, reduction="none").sum(-1)

        # loss_QQ = loss_QQ * max_IoU[..., None]

        loss_QQ = loss_QQ.mean()

        losses['loss_QQ'] = loss_QQ

        # tgt = Q_weights
        # # NL, W, W, D
        # diversity = tgt.unsqueeze(1) - tgt.unsqueeze(2)
        # # NL, W
        # diversity = torch.sqrt(torch.linalg.norm(diversity, ord=1, dim=(2, 3)) *
        #                        torch.linalg.norm(diversity, ord=np.inf, dim=(2, 3)))
        # # NL
        # diversity = torch.min(diversity, dim=1)[0]
        # diversity_loss = torch.mean(diversity)
        # losses['loss_QQ'] = -diversity_loss

        return losses

    def loss_KK(self, outputs, targets, indices, num_segments):
        """Compute the actionness regression loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'K_weights' in outputs
        assert 'C_weights' in outputs

        # L, N, K, K = outputs["K_weights"].shape

        # K_weights = torch.mean(outputs["K_weights"], dim=0)

        K_weights = outputs["K_weights"]
        normalized_K_weights = K_weights[0]
        for i in range(len(K_weights) - 1):
            normalized_K_weights = torch.sqrt(
                torch.bmm(normalized_K_weights, K_weights[i + 1].transpose(1, 2)) + 1.0e-7)
            normalized_K_weights = normalized_K_weights / torch.sum(normalized_K_weights, dim=-1, keepdim=True)
        K_weights = normalized_K_weights

        # C_weights = outputs["C_weights"].detach()

        # src_segments = outputs['pred_segments'].detach()
        # IoUs = list()
        # for n_i in range(len(targets)):
        #     tgt_segments = targets[n_i]['segments']
        #     this_IoU = segment_ops.segment_iou(segment_ops.segment_cw_to_t1t2(src_segments[n_i]),
        #                                        segment_ops.segment_cw_to_t1t2(tgt_segments))
        #     this_IoU = torch.max(this_IoU, dim=1)[0]
        #     IoUs.append(this_IoU)
        # IoUs = torch.stack(IoUs)
        # # IoUs = IoUs - torch.min(IoUs, dim=-1)[0].unsqueeze(-1) + 0.05
        # # IoUs = IoUs / torch.max(IoUs, dim=-1)[0].unsqueeze(-1)
        # # IoUs = torch.clamp(IoUs, min=0.10)
        # IoUs = IoUs.detach()
        #
        # max_IoU = torch.max(IoUs, dim=-1)[0]
        # max_IoU = torch.clamp(max_IoU / 0.5, max=1.0)

        # C_weights = C_weights * IoUs.unsqueeze(-1)
        # C_weights = C_weights / torch.sum(C_weights, dim=-1, keepdim=True)

        # N, Q, K = C_weights.shape

        # target_K_weights = F.softmax(KK_weights * 50.0, dim=-1)

        # KK_weights = torch.bmm(C_weights[-1].transpose(1, 2), C_weights[-1])
        # KK_weights = torch.sqrt(KK_weights + 1.0e-7)
        # target_K_weights = KK_weights / torch.sum(KK_weights, dim=-1, keepdim=True)

        # C_weights = outputs["C_weights"].detach()
        # normalized_KK_weights = torch.sqrt(torch.bmm(C_weights[0].transpose(1, 2), C_weights[0]) + 1.0e-7)
        # normalized_KK_weights = normalized_KK_weights / torch.sum(normalized_KK_weights, dim=-1, keepdim=True)
        # for i in range(len(C_weights) - 1):
        #     KK_weights = torch.sqrt(torch.bmm(C_weights[i + 1].transpose(1, 2), C_weights[i + 1]) + 1.0e-7)
        #     KK_weights = KK_weights / torch.sum(KK_weights, dim=-1, keepdim=True)
        #     normalized_KK_weights = torch.sqrt(torch.bmm(normalized_KK_weights, KK_weights) + 1.0e-7)
        #     normalized_KK_weights = normalized_KK_weights / torch.sum(normalized_KK_weights, dim=-1, keepdim=True)
        # target_K_weights = normalized_KK_weights

        C_weights = torch.mean(outputs["C_weights"], dim=0).detach()
        KK_weights = torch.bmm(C_weights.transpose(1, 2), C_weights)
        KK_weights = torch.sqrt(KK_weights + 1.0e-7)
        target_K_weights = KK_weights / torch.sum(KK_weights, dim=-1, keepdim=True)

        # target_K_weights = torch.eye(K).unsqueeze(0).repeat(N, 1, 1).to(K_weights.device)

        # print(torch.argsort(-target_K_weights[0].detach().cpu(), dim=-1)[:10, :10].numpy())
        # print(torch.max(target_K_weights[0].detach().cpu(), dim=-1)[0][:10].numpy())
        # print(torch.max(C_weights[0].detach().cpu(), dim=-1)[0][:10].numpy())
        # print(target_K_weights[0, 0].detach().cpu().numpy())
        # print((torch.max(C_weights) - torch.max(target_K_weights)).detach().cpu().numpy())

        # NK, K
        src_KK = (K_weights.flatten(0, 1) + 1.0e-7).log()
        tgt_KK = (target_K_weights.flatten(0, 1) + 1.0e-7).log()

        losses = {}

        loss_KK = F.kl_div(src_KK, tgt_KK, log_target=True, reduction="none").sum(-1)

        # loss_KK = loss_KK * max_IoU[..., None]

        loss_KK = loss_KK.mean()

        losses['loss_KK'] = loss_KK

        # tgt = outputs["K_weights"].flatten(0, 1)
        # # NL, W, W, D
        # diversity = tgt.unsqueeze(1) - tgt.unsqueeze(2)
        # # NL, W
        # diversity = torch.sqrt(torch.linalg.norm(diversity, ord=1, dim=(2, 3)) *
        #                        torch.linalg.norm(diversity, ord=np.inf, dim=(2, 3)))
        # # NL
        # diversity = torch.min(diversity, dim=1)[0]
        # diversity_loss = torch.mean(diversity)
        #
        # losses['loss_KK'] = -diversity_loss

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            "QQ": self.loss_QQ,
            "KK": self.loss_KK,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device = next(iter(outputs.values())).device
        indices = self.matcher(outputs_without_aux, targets)

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs['dn_meta']

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.range(0, len(targets[i]['labels']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                if 'QQ' in loss or 'KK' in loss:
                    continue
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(
                    self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes * scalar, **kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            l_dict = dict()
            l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
            losses.update(l_dict)

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if 'QQ' in loss or 'KK' in loss:
                        continue

                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict = {}
                    for loss in self.losses:
                        if 'QQ' in loss or 'KK' in loss:
                            continue
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes * scalar,
                                                    **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    l_dict = dict()
                    l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups, pad_size = dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups

        return output_known_lbs_bboxes, single_pad, num_dn_groups


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the TADEvaluator"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, fuse_score=True):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size] containing the duration of each video of the batch
        """
        out_logits, out_segments = outputs['pred_logits'], outputs['pred_segments']

        assert len(out_logits) == len(target_sizes)
        # assert target_sizes.shape[1] == 1

        prob = out_logits.sigmoid()  # [bs, nq, C]
        if fuse_score:
            prob *= outputs['pred_actionness']

        segments = segment_ops.segment_cw_to_t1t2(out_segments)  # bs, nq, 2

        if cfg.postproc_rank == 1:  # default
            # sort across different instances, pick top 100 at most
            topk_values, topk_indexes = torch.topk(prob.view(
                out_logits.shape[0], -1), min(cfg.postproc_ins_topk, prob.shape[1] * prob.shape[2]), dim=1)
            scores = topk_values
            topk_segments = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]

            # bs, nq, 2; bs, num, 2
            segments = torch.gather(
                segments, 1, topk_segments.unsqueeze(-1).repeat(1, 1, 2))
            query_ids = topk_segments
        else:
            # pick topk classes for each query
            # pdb.set_trace()
            scores, labels = torch.topk(prob, cfg.postproc_cls_topk, dim=-1)
            scores, labels = scores.flatten(1), labels.flatten(1)
            # (bs, nq, 1, 2)
            segments = segments[:, [
                                       i // cfg.postproc_cls_topk for i in
                                       range(cfg.postproc_cls_topk * segments.shape[1])], :]
            query_ids = (torch.arange(0, cfg.postproc_cls_topk * segments.shape[1], 1, dtype=labels.dtype,
                                      device=labels.device) // cfg.postproc_cls_topk)[None, :].repeat(labels.shape[0],
                                                                                                      1)

        # from normalized [0, 1] to absolute [0, length] coordinates
        vid_length = target_sizes
        scale_fct = torch.stack([vid_length, vid_length], dim=1)
        segments = segments * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'segments': b, 'query_ids': q}
                   for s, l, b, q in zip(scores, labels, segments, query_ids)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# @MODULE_BUILD_FUNCS.registe_with_name(module_name='dino')
def build(args):
    if args.binary:
        num_classes = 1
    else:
        if args.dataset_name == 'thumos14':
            num_classes = 20
        elif args.dataset_name == 'muses':
            num_classes = 25
        elif args.dataset_name in ['activitynet', 'hacs']:
            num_classes = 200
        else:
            raise ValueError('unknown dataset {}'.format(args.dataset_name))

    pos_embed = build_position_encoding(args)
    transformer = build_transformer(args)

    dn_labelbook_size = num_classes

    model = DINO(
        pos_embed,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=2,
        # random_refpoints_xy=args.random_refpoints_xy,
        # fix_refpoints_hw=args.fix_refpoints_hw,
        num_feature_levels=1,
        nheads=args.nheads,
        # dec_pred_class_embed_share=dec_pred_class_embed_share,
        # dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        # # two stage
        # two_stage_type=args.two_stage_type,
        # # box_share
        # two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        # two_stage_class_embed_share=args.two_stage_class_embed_share,
        # decoder_sa_type=args.decoder_sa_type,
        # num_patterns=args.num_patterns,
        # dn_number=args.dn_number if args.use_dn else 0,
        # dn_box_noise_scale=args.dn_box_noise_scale,
        # dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
    )

    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.seg_loss_coef}
    weight_dict['loss_giou'] = args.iou_loss_coef

    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    # for DN training
    if args.use_dn:
        weight_dict['loss_ce_dn'] = args.cls_loss_coef
        weight_dict['loss_bbox_dn'] = args.seg_loss_coef
        weight_dict['loss_giou_dn'] = args.iou_loss_coef

    clean_weight_dict = copy.deepcopy(weight_dict)

    losses = ['labels', 'boxes']
    if args.use_KK:
        weight_dict["loss_KK"] = args.KK_loss_coef
        losses.append("KK")

    if args.use_QQ:
        weight_dict["loss_QQ"] = args.QQ_loss_coef
        losses.append("QQ")

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # if args.two_stage_type != 'no':
    #     interm_weight_dict = {}
    #     try:
    #         no_interm_box_loss = args.no_interm_box_loss
    #     except:
    #         no_interm_box_loss = False
    #     _coeff_weight_dict = {
    #         'loss_ce': 1.0,
    #         'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
    #         'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
    #     }
    #     try:
    #         interm_loss_coef = args.interm_loss_coef
    #     except:
    #         interm_loss_coef = 1.0
    #     interm_weight_dict.update(
    #         {k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
    #     weight_dict.update(interm_weight_dict)

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses)
    # criterion.to(device)

    # postprocessors = {'bbox': PostProcess(num_select=args.num_select, nms_iou_threshold=args.nms_iou_threshold)}

    postprocessor = PostProcess()

    return model, criterion, postprocessor