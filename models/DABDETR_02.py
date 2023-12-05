import math
import copy

import torch
import torch.nn.functional as F
from torch import nn

from util import segment_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from models.matcher_02 import build_matcher
from models.position_encoding import build_position_encoding
from .custom_loss import sigmoid_focal_loss
from .dab_transformer_02 import build_transformer, build_ST_transformer
from opts import cfg


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_norm(norm_type, dim, num_groups=None):
    if norm_type == 'gn':
        assert num_groups is not None, 'num_groups must be specified'
        return nn.GroupNorm(num_groups, dim)
    elif norm_type == 'bn':
        return nn.BatchNorm1d(dim)
    else:
        raise NotImplementedError


class DABDETR(nn.Module):
    """ This is the TadTR module that performs temporal action detection """

    def __init__(self, position_embedding, transformer, num_classes, num_queries,
                 aux_loss=True, with_segment_refine=True, with_act_reg=False,
                 random_refpoints_xy=False, query_dim=2):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See deformable_transformer.py
            num_classes: number of action classes
            num_queries: number of action queries, ie detection slot. This is the maximal number of actions
                         TadTR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_segment_refine: iterative segment refinement
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        # self.class_embed = MLP(hidden_dim, hidden_dim, num_classes, 3)
        self.segment_embed = MLP(hidden_dim, hidden_dim, query_dim, 3)
        # self.overlap_embed = nn.Linear(hidden_dim // 2, 1)
        # self.overlap_embed = MLP(hidden_dim // 2, hidden_dim // 2, 1, 3)
        # self.S_segment_embed = MLP(hidden_dim, hidden_dim, 3, 3)
        # self.T_segment_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_dim = query_dim

        self.refpoint_embed = nn.Embedding(num_queries, query_dim)
        # self.S_refpoint_embed = nn.Embedding(num_queries, 3)
        # self.T_refpoint_embed = nn.Embedding(num_queries, 2)
        self.random_refpoints_xy = random_refpoints_xy
        if random_refpoints_xy:
            # import ipdb; ipdb.set_trace()
            self.refpoint_embed.weight.data[:, :1].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :1] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :1])
            self.refpoint_embed.weight.data[:, :1].requires_grad = False

            # self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            # self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            # self.refpoint_embed.weight.data[:, :2].requires_grad = False

        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(2048, hidden_dim, kernel_size=1),
                # nn.Conv2d(2048, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )])
        # self.O_input_proj = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(2048, hidden_dim // 2, kernel_size=1),
        #         nn.GroupNorm(32, hidden_dim // 2),
        #     )])
        self.position_embedding = position_embedding
        self.aux_loss = aux_loss
        self.with_segment_refine = with_segment_refine
        self.with_act_reg = with_act_reg

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        # self.class_embed.layers[-1].bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.segment_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.segment_embed.layers[-1].bias.data, 0)
        # nn.init.constant_(self.S_segment_embed.layers[-1].weight.data, 0)
        # nn.init.constant_(self.S_segment_embed.layers[-1].bias.data, 0)
        # nn.init.constant_(self.T_segment_embed.layers[-1].weight.data, 0)
        # nn.init.constant_(self.T_segment_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # num_pred = transformer.decoder.num_layers
        if with_segment_refine:
            # hack implementation for segment refinement
            self.transformer.decoder.segment_embed = self.segment_embed
            # self.transformer.S_decoder.segment_embed = self.S_segment_embed
            # self.transformer.T_decoder.segment_embed = self.T_segment_embed

        # self.t_embeddings = nn.Embedding(512, hidden_dim)
        # self.h_embeddings = nn.Embedding(64, hidden_dim)
        # self.w_embeddings = nn.Embedding(64, hidden_dim)

    def forward(self, features):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            or a tuple of tensors and mask

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-action) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_segments": The normalized segments coordinates for all queries, represented as
                               (center, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized segment.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(features, NestedTensor):
            if isinstance(features, (list, tuple)):
                features = NestedTensor(*features)
            else:
                features = nested_tensor_from_tensor_list(features)  # (n, c, t)

        pos = self.position_embedding(features)
        src, mask = features.decompose()

        src = self.input_proj[0](src)

        embedweight = self.refpoint_embed.weight
        hs, reference, memory, Q_weights, K_weights, C_weights = \
            self.transformer(src, mask, embedweight, pos)

        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.segment_embed(hs)
        tmp += reference_before_sigmoid

        outputs_coord = tmp.sigmoid()
        outputs_class = self.class_embed(hs)

        out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_coord[-1],
               'Q_weights': Q_weights, 'K_weights': K_weights, 'C_weights': C_weights}

        # outputs_coord_all = outputs_coord
        # outputs_class_all = outputs_class
        #
        # Q_weights_all = Q_weights
        # K_weights_all = K_weights
        # Q_weights_all = C_weights
        #
        # outputs_class = outputs_class_all[:, :, :self.num_queries // 6]
        # outputs_coord = outputs_coord_all[:, :, :self.num_queries // 6]
        #
        # outputs_class_one2many = outputs_class_all[:, :, self.num_queries // 6:]
        # outputs_coord_one2many = outputs_coord_all[:, :, self.num_queries // 6:]
        #
        # out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_coord[-1],
        #        'pred_logits_one2many': outputs_class_one2many[-1], 'pred_segments_one2many': outputs_coord_one2many[-1],
        #        'Q_weights': Q_weights, 'K_weights': K_weights, 'C_weights': C_weights,
        #        ''}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
            # out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_overlap)
            # out['aux_outputs_one2many'] = self._set_aux_loss(outputs_class_one2many, outputs_coord_one2many, outputs_overlap_one2many)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_segments': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    # @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_coord, outputs_overlap):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     return [{'pred_logits': a, 'pred_segments': b, 'pred_overlap': c} for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_overlap[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for TadTR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth segments and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and segment)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of action categories, omitting the special no-action category
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

        # self.eos_coef = 0.1
        # empty_weight = torch.ones(self.num_classes + 1)
        # empty_weight[-1] = self.eos_coef
        # self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_segments, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_segments]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)

        idx = self._get_src_permutation_idx(indices)
        src_segments = outputs['pred_segments'][idx]
        target_segments = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        IoUs = segment_ops.segment_iou(segment_ops.segment_cw_to_t1t2(src_segments),
                                       segment_ops.segment_cw_to_t1t2(target_segments))
        IoUs = torch.diag(IoUs).detach()

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes[idx] = target_classes_o
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        # target_classes_onehot[idx] = target_classes_onehot[idx]
        target_classes_onehot[idx] = target_classes_onehot[idx] * IoUs.unsqueeze(-1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_segments,
                                     alpha=self.focal_alpha, gamma=2) * src_logits.shape[1] # nq
        # loss_ce = sigmoid_focal_loss(src_logits[idx], target_classes_onehot[idx], num_segments,
        #                              alpha=self.focal_alpha, gamma=2).sum() / num_segments
        # loss_ce = loss_ce.mean(1).sum() / num_segments * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # target_classes = torch.full(src_logits.shape[:2], self.num_classes,
        #                             dtype=torch.int64, device=src_logits.device)
        # target_classes[idx] = target_classes_o
        #
        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        # losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

            # # N, Q
            # probs = torch.max(src_logits.sigmoid(), dim=-1)[0]
            # top_k_indices = torch.argsort(-probs, dim=-1)
            # top_1_indices = top_k_indices[..., 0]
            # top_2_indices = top_k_indices[..., 1]
            # # score_gap = torch.mean(probs[top_1_indices] - probs[top_2_indices], dim=0)
            # score_gap = torch.mean(probs[torch.arange(len(top_1_indices)),
            #                              top_1_indices[torch.arange(len(top_1_indices))]] -
            #                        probs[torch.arange(len(top_2_indices)),
            #                              top_2_indices[torch.arange(len(top_2_indices))]], dim=0)
            #
            # losses['score_gap'] = score_gap

        return losses

    def loss_segments(self, outputs, targets, indices, num_segments):
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

    def loss_overlap(self, outputs, targets, indices, num_segments):
        """Compute the losses related to the segmentes, the L1 regression loss and the IoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'pred_overlap' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_segments = outputs['pred_segments'][idx]
        target_segments = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        src_overlap = outputs['pred_overlap'][idx].sigmoid()

        tgt_overlap = segment_ops.segment_iou(segment_ops.segment_cw_to_t1t2(target_segments),
                                              segment_ops.segment_cw_to_t1t2(src_segments))
        tgt_overlap = torch.diag(tgt_overlap).detach()

        loss_overlap = torch.square(src_overlap - tgt_overlap).sum() / num_segments

        # IoUs = segment_ops.segment_iou(segment_ops.segment_cw_to_t1t2(src_segments),
        #                                segment_ops.segment_cw_to_t1t2(target_segments))
        # IoUs = torch.diag(IoUs).detach()
        #
        # src_logits = outputs['pred_overlap']
        #
        # target_classes = torch.full(src_logits.shape[:2], 1, dtype=torch.int64, device=src_logits.device)
        # target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
        #                                     dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        #
        # target_classes_o = torch.zeros_like(torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]))
        # target_classes[idx] = target_classes_o
        # target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        # target_classes_onehot[idx] = target_classes_onehot[idx] * IoUs.unsqueeze(-1)
        #
        # target_classes_onehot = target_classes_onehot[:, :, :-1]
        # loss_overlap = sigmoid_focal_loss(src_logits, target_classes_onehot, num_segments,
        #                                   alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        # # loss_overlap = loss_overlap.mean(1).sum() / num_segments * src_logits.shape[1]

        losses['loss_overlap'] = loss_overlap
        return losses

    def loss_mask(self, outputs, targets, indices, num_segments):
        """Compute the losses related to the segmentes, the L1 regression loss and the IoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """

        # idx = self._get_src_permutation_idx(indices)
        # src_segments = outputs['pred_segments']
        # target_segments = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        #
        # losses = {}
        #
        # obj_center = target_segments
        # obj_boundary = segment_ops.segment_cw_to_t1t2(target_segments)
        #
        # # nb = src_segments.size(0)
        # # nk = outputs["K_weights"].size(2)
        # # center_memory_mask = torch.zeros(dtype=torch.float, size=(nb, nk), device=src_segments.device)
        # # boundary_memory_mask = torch.zeros(dtype=torch.float, size=(nb, nk), device=src_segments.device)
        # # for n_i, (center, boundary) in enumerate(zip(obj_center, obj_boundary)):
        # #     center = torch.clamp(torch.round(center * (nk - 1)).int(), 0, nk - 1)
        # #     boundary = torch.clamp(torch.round(boundary * (nk - 1)).int(), 0, nk - 1)
        # #     c, w = center[..., 0], center[..., 1]
        # #     s, e = boundary[..., 0], boundary[..., 1]
        # #     bw = torch.clamp(torch.round(w / 8).int(), min=1)
        # #     center_memory_mask[n_i, s:e + 1] = 1.0
        # #     boundary_memory_mask[n_i, torch.clamp(s - bw, min=0):torch.clamp(s + bw + 1, max=nk)] = 1.0
        # #     boundary_memory_mask[n_i, torch.clamp(e - bw, min=0):torch.clamp(e + bw + 1, max=nk)] = 1.0
        # # center_memory_mask = center_memory_mask.detach()
        # # boundary_memory_mask = boundary_memory_mask.detach()
        #
        # # nb = src_segments.size(0)
        # # nk = outputs["K_weights"].size(2)
        # # memory_mask = torch.zeros(dtype=torch.float, size=(nb, nk), device=src_segments.device)
        # # for n_i, (center, boundary) in enumerate(zip(obj_center, obj_boundary)):
        # #     center = torch.clamp(torch.round(center * (nk - 1)).int(), 0, nk - 1)
        # #     boundary = torch.clamp(torch.round(boundary * (nk - 1)).int(), 0, nk - 1)
        # #     c, w = center[..., 0], center[..., 1]
        # #     s, e = boundary[..., 0], boundary[..., 1]
        # #     bw = torch.clamp(torch.round(w / 8).int(), min=1)
        # #     memory_mask[n_i, s:e + 1] = 1.0
        # #     memory_mask[n_i, torch.clamp(s - bw, min=0):torch.clamp(s + bw + 1, max=nk)] = 1.0
        # #     memory_mask[n_i, torch.clamp(e - bw, min=0):torch.clamp(e + bw + 1, max=nk)] = 1.0
        # # memory_mask = memory_mask.detach()
        #
        # nk = outputs["K_weights"].size(2)
        # center = torch.clamp(torch.round(obj_center * (nk - 1)).int(), 0, nk - 1)
        # boundary = torch.clamp(torch.round(obj_boundary * (nk - 1)).int(), 0, nk - 1)
        # c, w = center[..., 0], center[..., 1]
        # s, e = boundary[..., 0], boundary[..., 1]
        # bw = torch.clamp(torch.round(w / 8).int(), min=1)
        # memory_mask = torch.logical_and(
        #     torch.arange(nk).unsqueeze(0).to(src_segments) >= torch.clamp(s - bw, min=0)[:, None],
        #     torch.arange(nk).unsqueeze(0).to(src_segments) <= torch.clamp(e + bw, max=nk - 1)[:, None])
        # memory_mask = memory_mask.float()
        #
        # # n, k
        # # C_C_weights = outputs["C_weights"].mean(0)[..., 0][idx]
        # # L_C_weights = outputs["C_weights"].mean(0)[..., 1][idx]
        # #
        # # C_C_weights = torch.sum(C_C_weights * center_memory_mask, dim=-1)
        # # L_C_weights = torch.sum(L_C_weights * boundary_memory_mask, dim=-1)
        # #
        # # loss_mask = (-torch.log(C_C_weights + 1.0e-8) + -torch.log(L_C_weights + 1.0e-8)).sum() / num_segments
        #
        # C_weights = outputs["C_weights"].mean(0)[idx]
        # C_weights = torch.sum(C_weights * memory_mask, dim=-1)
        #
        # loss_mask = (-torch.log(C_weights + 1.0e-8)).sum() / num_segments
        #
        # # loss_mask = 0.0
        # # for l_i in range(len(outputs["C_weights"])):
        # #     C_C_weights = outputs["C_weights"][l_i, ..., 0][idx]
        # #     L_C_weights = outputs["C_weights"][l_i, ..., 1][idx]
        # #
        # #     C_C_weights = torch.sum(C_C_weights * center_memory_mask, dim=-1)
        # #     L_C_weights = torch.sum(L_C_weights * boundary_memory_mask, dim=-1)
        # #
        # #     loss_mask += (-torch.log(C_C_weights + 1.0e-8) + -torch.log(L_C_weights + 1.0e-8)).sum() / num_segments
        #
        # loss_mask = loss_mask / len(outputs["C_weights"]) / 5

        src_segments = outputs['pred_segments']
        src_boundary = segment_ops.segment_cw_to_t1t2(src_segments)
        # src_segments = torch.cat((torch.stack([a_o['pred_segments'] for a_o in outputs['aux_outputs']], dim=0),
        #                           outputs['pred_segments'].unsqueeze(0)), dim=0)
        # nl, nb, nq = src_segments.shape[:3]
        # src_boundary = segment_ops.segment_cw_to_t1t2(src_segments.flatten(0, 1))

        losses = {}

        # tgt_KK = list()
        # nk = outputs["K_weights"].size(2)
        # for n_i in range(len(src_segments)):
        #     target_segments = targets[n_i]['segments']
        #     obj_boundary = segment_ops.segment_cw_to_t1t2(target_segments)
        #     boundary = torch.clamp(torch.round(obj_boundary * (nk - 1)).int(), 0, nk - 1)
        #     s, e = boundary[..., 0], boundary[..., 1]
        #     ks = torch.arange(nk).to(src_segments).unsqueeze(0)
        #     this_tgt_KK = torch.clamp(torch.logical_and(ks >= s[:, None], ks <= e[:, None]).sum(dim=0), max=1)[:, None]
        #     this_tgt_KK = torch.logical_and(this_tgt_KK, this_tgt_KK.transpose(0, 1))
        #     tgt_KK.append(this_tgt_KK)
        # tgt_KK = torch.stack(tgt_KK, dim=0).float().softmax(dim=-1).detach()

        # nk = outputs["K_weights"].size(2)
        # boundary = torch.clamp(torch.round(src_boundary * (nk - 1)).int(), 0, nk - 1)
        # s, e = boundary[..., 0], boundary[..., 1]
        # ks = torch.arange(nk).to(src_segments)[None, None, :]
        # tgt_KK = torch.logical_and(ks >= s[..., None], ks <= e[..., None]).float().softmax(dim=-1)
        # tgt_KK = torch.bmm(tgt_KK.transpose(1, 2), tgt_KK)
        # tgt_KK = torch.sqrt(tgt_KK + 1.0e-7)
        # tgt_KK = tgt_KK / torch.sum(tgt_KK, dim=-1, keepdim=True).detach()

        tgt_QQ = segment_ops.batched_segment_iou(src_boundary, src_boundary).softmax(dim=-1).detach()
        # tgt_QQ = segment_ops.batched_segment_iou(src_boundary, src_boundary).view(nl, nb, nq, nq).softmax(dim=-1).mean(0).detach()

        # C_weights = outputs["C_weights"].flatten(0, 1)
        C_weights = torch.mean(outputs["C_weights"], dim=0)
        # src_KK = torch.bmm(C_weights.transpose(1, 2), C_weights)
        # src_KK = torch.sqrt(src_KK + 1.0e-7)
        # src_KK = src_KK / torch.sum(src_KK, dim=-1, keepdim=True)
        #
        # src_KK = (src_KK.flatten(0, 1) + 1.0e-7).log()
        # tgt_KK = (tgt_KK.flatten(0, 1) + 1.0e-7).log()
        #
        # loss_KK = F.kl_div(src_KK, tgt_KK, log_target=True, reduction="none").sum(-1).mean()

        src_QQ = torch.sqrt(torch.bmm(C_weights, C_weights.transpose(1, 2)) + 1.0e-7)
        src_QQ = src_QQ / torch.sum(src_QQ, dim=-1, keepdim=True)

        src_QQ = (src_QQ.flatten(0, 1) + 1.0e-7).log()
        tgt_QQ = (tgt_QQ.flatten(0, 1) + 1.0e-7).log()

        loss_QQ = F.kl_div(src_QQ, tgt_QQ, log_target=True, reduction="none").sum(-1).mean()

        losses["loss_mask"] = loss_QQ

        return losses

    def loss_actionness(self, outputs, targets, indices, num_segments):
        """Compute the actionness regression loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'pred_segments' in outputs
        assert 'pred_actionness' in outputs
        src_segments = outputs['pred_segments'].view((-1, 2))
        target_segments = torch.cat([t['segments'] for t in targets], dim=0)

        losses = {}

        iou_mat = segment_ops.segment_iou(
            segment_ops.segment_cw_to_t1t2(src_segments),
            segment_ops.segment_cw_to_t1t2(target_segments))

        gt_iou = iou_mat.max(dim=1)[0]
        pred_actionness = outputs['pred_actionness']
        loss_actionness = F.l1_loss(pred_actionness.view(-1), gt_iou.view(-1).detach())

        losses['loss_iou'] = loss_actionness
        return losses

    def loss_recon(self, outputs, targets, indices, num_segments):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_segments]
        """
        assert 'pred_recon' in outputs

        src_embeds = outputs['pred_recon']
        tgt_embeds = outputs['queries'].detach()

        loss_embed = torch.mean(torch.sum(torch.square(src_embeds - tgt_embeds.unsqueeze(1)), dim=-1))

        losses = {'loss_recon': loss_embed}

        return losses

    def loss_QQ(self, outputs, targets, indices, num_segments):
        """Compute the actionness regression loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'Q_weights' in outputs
        assert 'C_weights' in outputs
        assert 'pred_segments' in outputs

        C_weights = outputs["C_weights"].flatten(0, 1).detach()
        QQ_weights = torch.sqrt(torch.bmm(C_weights, C_weights.transpose(1, 2)) + 1.0e-7)
        target_Q_weights = QQ_weights / torch.sum(QQ_weights, dim=-1, keepdim=True)

        Q_weights = outputs["Q_weights"].flatten(0, 1)

        src_QQ = (Q_weights.flatten(0, 1) + 1.0e-7).log()
        tgt_QQ = (target_Q_weights.flatten(0, 1) + 1.0e-7).log()

        losses = {}

        loss_QQ = F.kl_div(src_QQ, tgt_QQ, log_target=True, reduction="none").sum(-1)
        # loss_QQ = loss_QQ * IoU_weight.unsqueeze(-1)
        # loss_QQ = loss_QQ.sum() / loss_QQ
        loss_QQ = loss_QQ.mean()

        losses['loss_QQ'] = loss_QQ
        return losses

    def loss_KK(self, outputs, targets, indices, num_segments):
        """Compute the actionness regression loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'K_weights' in outputs
        assert 'C_weights' in outputs

        C_weights = torch.mean(outputs["C_weights"], dim=0).detach()
        KK_weights = torch.bmm(C_weights.transpose(1, 2), C_weights)
        KK_weights = torch.sqrt(KK_weights + 1.0e-7)
        target_K_weights = KK_weights / torch.sum(KK_weights, dim=-1, keepdim=True)

        K_weights = torch.mean(outputs["K_weights"], dim=0)

        src_KK = (K_weights.flatten(0, 1) + 1.0e-7).log()
        tgt_KK = (target_K_weights.flatten(0, 1) + 1.0e-7).log()

        losses = {}

        loss_KK = F.kl_div(src_KK, tgt_KK, log_target=True, reduction="none").sum(-1)
        loss_KK = loss_KK.mean()

        losses['loss_KK'] = loss_KK
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

    def get_loss(self, loss, outputs, targets, indices, num_segments, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'segments': self.loss_segments,
            'overlap': self.loss_overlap,
            'mask': self.loss_mask,
            "QQ": self.loss_QQ,
            "KK": self.loss_KK,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_segments, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target segments accross all nodes, for normalization purposes
        num_segments = sum(len(t["labels"]) for t in targets)
        num_segments = torch.as_tensor([num_segments], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_segments)
        num_segments = torch.clamp(num_segments / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_segments, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if 'QQ' in loss or 'KK' in loss or 'mask' in loss:
                        continue

                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False

                    num_segments = sum(len(t["labels"]) for t in targets)
                    num_segments = torch.as_tensor([num_segments], dtype=torch.float,
                                                   device=next(iter(outputs.values())).device)
                    if is_dist_avail_and_initialized():
                        torch.distributed.all_reduce(num_segments)
                    num_segments = torch.clamp(num_segments / get_world_size(), min=1).item()
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_segments, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        self.indices = indices
        return losses


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

    model = DABDETR(
        pos_embed,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True,
        with_segment_refine=True,
        with_act_reg=False)

    matcher = build_matcher(args)
    # losses = ['labels', 'segments']
    losses = ['labels', 'segments', 'mask']

    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_segments': args.seg_loss_coef,
        'loss_iou': args.iou_loss_coef,
        'loss_mask': 1.0,
    }

    if args.use_KK:
        weight_dict["loss_KK"] = args.KK_loss_coef
        losses.append("KK")

    if args.use_QQ:
        weight_dict["loss_QQ"] = args.QQ_loss_coef
        losses.append("QQ")

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # new_dict = dict()
    # for key, value in weight_dict.items():
    #     new_dict[key] = value
    #     new_dict[key + "_one2many"] = value
    # weight_dict = new_dict

    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)

    postprocessor = PostProcess()

    return model, criterion, postprocessor
