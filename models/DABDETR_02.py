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
from .dab_transformer_02 import build_transformer
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

    def __init__(self, position_embedding, transformer, num_classes, num_queries_one2one,
                 num_queries_one2many=0, aux_loss=True, with_segment_refine=True,
                 random_refpoints_xy=False, query_dim=2,
                 two_stage=False, num_queries_enc=40):
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
        self.num_queries_enc = num_queries_enc
        self.num_queries_one2one = num_queries_one2one
        self.num_queries_one2many = num_queries_one2many
        self.num_queries = num_queries_one2one + num_queries_one2many
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.segment_embed = MLP(hidden_dim, hidden_dim, query_dim, 3)

        self.two_stage = two_stage
        self.query_dim = query_dim
        self.refpoint_embed = nn.Embedding(self.num_queries, query_dim)
        self.random_refpoints_xy = random_refpoints_xy
        if random_refpoints_xy:
            self.refpoint_embed.weight.data[:self.num_queries_one2one, :1].uniform_(0, 1)
            self.refpoint_embed.weight.data[:self.num_queries_one2one, :1] = \
                inverse_sigmoid(self.refpoint_embed.weight.data[:self.num_queries_one2one, :1])
            self.refpoint_embed.weight.data[:self.num_queries_one2one, :1].requires_grad = False

        if self.num_queries_one2many > 0:
            # uniform_points = torch.linspace(0.0, 1.0, self.num_queries_one2many, dtype=torch.float32)
            # self.refpoint_embed.weight.data[self.num_queries_one2one:, :1] = uniform_points.unsqueeze(-1)

            self.refpoint_embed.weight.data[self.num_queries_one2one:, :1].uniform_(0, 1)
            self.refpoint_embed.weight.data[self.num_queries_one2one:, :1] = \
                inverse_sigmoid(self.refpoint_embed.weight.data[self.num_queries_one2one:, :1])
            self.refpoint_embed.weight.data[self.num_queries_one2one:, :1].requires_grad = False

            # self.refpoint_embed.weight.data[self.num_queries_one2one:, :].uniform_(0, 1)
            # self.refpoint_embed.weight.data[self.num_queries_one2one:, :] = \
            #     inverse_sigmoid(self.refpoint_embed.weight.data[self.num_queries_one2one:, :])
            # self.refpoint_embed.weight.data[self.num_queries_one2one:, :].requires_grad = False

        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(2048, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )])
        self.position_embedding = position_embedding
        self.aux_loss = aux_loss
        self.with_segment_refine = with_segment_refine

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.segment_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.segment_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # num_pred = transformer.decoder.num_layers
        if with_segment_refine:
            # hack implementation for segment refinement
            self.transformer.decoder.segment_embed = self.segment_embed

        if two_stage:
            # hack implementation for two-stage
            self.transformer.encoder_class_embed = copy.deepcopy(self.class_embed)
            self.transformer.encoder_segment_embed = copy.deepcopy(self.segment_embed)

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
        hs, reference, enc_outputs_class, enc_outputs_coord, Q_weights, K_weights, C_weights = \
            self.transformer(src, mask, embedweight, pos)

        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.segment_embed(hs)
        tmp += reference_before_sigmoid
        outputs_coord_all = tmp.sigmoid()
        # before = segment_ops.segment_cw_to_t1t2(reference)
        # after = segment_ops.segment_cw_to_t1t2(outputs_coord_all)

        outputs_class_all = self.class_embed(hs)

        Q_weights_all = Q_weights
        K_weights_all = K_weights
        C_weights_all = C_weights

        # if self.two_stage:
        #     outputs_class = outputs_class_all[:, :, :self.num_queries_enc + self.num_queries_one2one]
        #     outputs_coord = outputs_coord_all[:, :, :self.num_queries_enc + self.num_queries_one2one]
        #
        #     Q_weights = Q_weights_all[:, :, :self.num_queries_enc + self.num_queries_one2one,
        #                 :self.num_queries_enc + self.num_queries_one2one]
        #     K_weights = K_weights_all
        #     C_weights = C_weights_all[:, :, :self.num_queries_enc + self.num_queries_one2one]
        # else:
        outputs_class = outputs_class_all[:, :, :self.num_queries_one2one]
        outputs_coord = outputs_coord_all[:, :, :self.num_queries_one2one]

        Q_weights = Q_weights_all[:, :, :self.num_queries_one2one, :self.num_queries_one2one]
        K_weights = K_weights_all
        C_weights = C_weights_all[:, :, :self.num_queries_one2one]

        out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_coord[-1],
               'Q_weights': Q_weights, 'K_weights': K_weights, 'C_weights': C_weights}

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord.sigmoid()
            out["enc_outputs"] = {"pred_logits": enc_outputs_class, "pred_segments": enc_outputs_coord}

        if self.num_queries_one2many > 0:
            # if self.two_stage:
            #     outputs_class_one2many = outputs_class_all[:, :, self.num_queries_enc + self.num_queries_one2one:]
            #     outputs_coord_one2many = outputs_coord_all[:, :, self.num_queries_enc + self.num_queries_one2one:]
            #
            #     Q_weights_one2many = Q_weights_all[:, :, self.num_queries_enc + self.num_queries_one2one:,
            #                          self.num_queries_enc + self.num_queries_one2one:]
            #     K_weights_one2many = K_weights_all
            #     C_weights_one2many = C_weights_all[:, :, self.num_queries_enc + self.num_queries_one2one:]
            # else:
            outputs_class_one2many = outputs_class_all[:, :, self.num_queries_one2one:]
            outputs_coord_one2many = outputs_coord_all[:, :, self.num_queries_one2one:]

            Q_weights_one2many = Q_weights_all[:, :, self.num_queries_one2one:, self.num_queries_one2one:]
            K_weights_one2many = K_weights_all
            C_weights_one2many = C_weights_all[:, :, self.num_queries_one2one:]

            out['pred_logits_one2many'] = outputs_class_one2many[-1]
            out['pred_segments_one2many'] = outputs_coord_one2many[-1]
            out['Q_weights_one2many'] = Q_weights_one2many
            out['K_weights_one2many'] = K_weights_one2many
            out['C_weights_one2many'] = C_weights_one2many

            if self.two_stage:
                out["enc_outputs_one2many"] = {"pred_logits": enc_outputs_class, "pred_segments": enc_outputs_coord}
                # out['pred_logits_one2many'] = enc_outputs_class
                # out['pred_segments_one2many'] = enc_outputs_coord

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
            if self.num_queries_one2many > 0:
                out['aux_outputs_one2many'] = self._set_aux_loss(outputs_class_one2many, outputs_coord_one2many)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_segments': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


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

        self.cost_giou = 1.0
        self.cost_bbox = 0.0
        self.cost_class = 0.0

        self.enc_ratios = [1, 2, 4, 8]

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
        IoUs = torch.diag(IoUs)
        # max_IoUs = torch.max(IoUs, dim=-1, keepdims=True)[0]
        # squared_IoUs = torch.square(IoUs)
        # max_squared_IoUs = torch.max(squared_IoUs, dim=-1, keepdims=True)[0]
        # IoUs = ((squared_IoUs / (max_squared_IoUs + 1.0e-12)) * max_IoUs).unsqueeze(-1).detach()

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes[idx] = target_classes_o
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        # target_classes_onehot[idx] = target_classes_onehot[idx]
        target_classes_onehot[idx] = target_classes_onehot[idx] * IoUs
        target_classes_onehot[idx] = target_classes_onehot[idx] * IoUs.unsqueeze(-1).detach()

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
        # loss_iou = 1 - torch.diag(segment_ops.generalized_segment_iou(
        #     segment_ops.segment_cw_to_t1t2(src_segments),
        #     segment_ops.segment_cw_to_t1t2(target_segments)))
        losses['loss_iou'] = loss_iou.sum() / num_segments
        return losses

    def loss_global(self, outputs, targets, indices, num_segments):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_segments]
        """
        assert 'pred_global_logits' in outputs
        src_logits = outputs['pred_global_logits']

        target_classes = torch.stack([torch.mode(t["labels"])[0] for t in targets])

        loss = F.cross_entropy(src_logits, target_classes)

        losses = {"loss_global": loss}

        return losses

    def loss_actionness(self, outputs, targets, indices, num_segments):
        """Compute the losses related to the segmentes, the L1 regression loss and the IoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'pred_actionness' in outputs
        src_logits = outputs['pred_actionness']

        # tgt_actionness = list()
        # nb, nk = src_logits.shape[:2]
        # ks = torch.arange(nk, device=src_logits.device).unsqueeze(0)
        # for n_i in range(nb):
        #     target_segments = targets[n_i]["segments"]
        #     target_segments = segment_ops.segment_cw_to_t1t2(target_segments)
        #     target_segments = torch.clamp(torch.round(target_segments * (nk - 1)).int(), 0, nk - 1)
        #     s, e = target_segments[..., 0], target_segments[..., 1]
        #     this_target = torch.logical_and(ks >= s.unsqueeze(-1), ks <= e.unsqueeze(-1)).float()
        #     this_target = torch.sum(this_target, dim=0).clamp(0, 1)
        #     tgt_actionness.append(this_target)
        # tgt_actionness = torch.stack(tgt_actionness, dim=0).unsqueeze(-1).detach()

        QK = outputs['C_weights']
        K_masks = QK.mean(dim=(0, 2))
        K_masks = K_masks / torch.sum(K_masks, dim=1, keepdims=True)
        nk = K_masks.size(1)
        top_k = round(nk / 2)
        top_k_values = torch.topk(K_masks, top_k, dim=-1)[0][:, -1]
        tgt_actionness = (K_masks > top_k_values.unsqueeze(-1)).float().unsqueeze(-1).detach()

        gamma = 2
        alpha = self.focal_alpha
        prob = src_logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(src_logits, tgt_actionness, reduction="none")
        p_t = prob * tgt_actionness + (1 - prob) * (1 - tgt_actionness)
        loss_actionness = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * tgt_actionness + (1 - alpha) * (1 - tgt_actionness)
            loss_actionness = alpha_t * loss_actionness

        loss_actionness = loss_actionness.mean()
        losses = {'loss_actionness': loss_actionness}
        return losses

    def loss_QK(self, outputs, targets, indices, num_segments):
        """Compute the losses related to the segmentes, the L1 regression loss and the IoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'C_weights' in outputs
        assert 'pred_segments' in outputs

        EPS = 1.0e-12

        split = 0
        # src_segments = outputs['pred_segments']
        # src_boundary = segment_ops.segment_cw_to_t1t2(src_segments)
        # tgt_QQ = segment_ops.batched_segment_iou(src_boundary, src_boundary).softmax(dim=-1).detach()
        #
        # C_weights = torch.mean(outputs["C_weights"], dim=0)
        #
        # src_QQ = torch.sqrt(torch.bmm(C_weights, C_weights.transpose(1, 2)) + EPS)
        # src_QQ = (src_QQ / torch.sum(src_QQ, dim=-1, keepdim=True))
        #
        # src_QQ = (src_QQ.flatten(0, 1) + EPS).log()
        # tgt_QQ = (tgt_QQ.flatten(0, 1) + EPS).log()
        #
        # loss_QK = F.kl_div(src_QQ, tgt_QQ, log_target=True, reduction="none").sum(-1).mean()
        split = 0
        # Prev Main
        src_prob = outputs["pred_logits"].sigmoid()
        src_bbox = outputs["pred_segments"]
        # src_prob = torch.cat((torch.stack([a_o['pred_logits'] for a_o in outputs['aux_outputs']], dim=0),
        #                       outputs['pred_logits'].unsqueeze(0)), dim=0).flatten(0, 1)
        # src_bbox = torch.cat((torch.stack([a_o['pred_segments'] for a_o in outputs['aux_outputs']], dim=0),
        #                       outputs['pred_segments'].unsqueeze(0)), dim=0).flatten(0, 1)
        tgt_prob = src_prob
        tgt_bbox = src_bbox

        IoUs = segment_ops.batched_segment_iou(segment_ops.segment_cw_to_t1t2(src_bbox),
                                               segment_ops.segment_cw_to_t1t2(tgt_bbox))
        # IoUs = segment_ops.generalized_segment_iou(segment_ops.segment_cw_to_t1t2(src_bbox),
        #                                            segment_ops.segment_cw_to_t1t2(tgt_bbox))

        # Compute the classification cost.
        # alpha, gamma = 0.25, 2.0
        # out_prob = torch.bmm(src_prob, tgt_prob.transpose(1, 2)) * torch.sqrt(IoUs)
        # neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        # pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        # cost_class = pos_cost_class - neg_cost_class
        # cost_class = torch.bmm(src_prob, tgt_prob.transpose(1, 2))
        cost_class = torch.clamp(torch.max(tgt_prob, dim=-1)[0].unsqueeze(-2) -
                                 torch.max(src_prob, dim=-1)[0].unsqueeze(-1), min=0.0)

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(src_bbox, tgt_bbox, p=1)
        cost_giou = -IoUs
        # Final cost matrix
        C = -(self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou) / \
            (self.cost_bbox + self.cost_class + self.cost_giou)

        tgt_QQ = C.softmax(dim=-1).detach()
        # L, N, Q = outputs["Q_weights"].shape[:3]
        # tgt_QQ = C.view(L, N, Q, Q).mean(0).softmax(-1).detach()

        src_QQ = torch.mean(outputs["C_weights"], dim=0)
        src_QQ = F.normalize(src_QQ, p=2.0, dim=-1)
        src_QQ = torch.bmm(src_QQ, src_QQ.transpose(1, 2)).softmax(dim=-1)
        # L, N, Q, K = outputs["C_weights"].shape
        # src_QQ = outputs["C_weights"].flatten(0, 1)
        # src_QQ = F.normalize(src_QQ, p=2.0, dim=-1)
        # src_QQ = torch.bmm(src_QQ, src_QQ.transpose(1, 2))
        # src_QQ = src_QQ.view(L, N, Q, Q).mean(0).softmax(-1)

        # N, Q, Q = src_QQ.shape
        src_QQ = (src_QQ.flatten(0, 1) + EPS).log()
        tgt_QQ = (tgt_QQ.flatten(0, 1) + EPS).log()

        loss_QQ = F.kl_div(src_QQ, tgt_QQ, log_target=True, reduction="none").sum(-1).mean()
        # loss_QQ = F.kl_div(src_QQ, tgt_QQ, log_target=True, reduction="none")
        # loss_QQ = loss_QQ.view(N, Q, Q)
        # mask = (torch.arange(Q).unsqueeze(-1) <= torch.arange(Q).unsqueeze(0)).float().to(loss_QQ.device)
        # loss_QQ = (loss_QQ * mask).sum(-1).mean()

        loss_QK = loss_QQ
        split = 0
        # src_prob = torch.cat((torch.stack([a_o['pred_logits'] for a_o in outputs['aux_outputs']], dim=0),
        #                       outputs['pred_logits'].unsqueeze(0)), dim=0).flatten(0, 1)
        # after_bbox = torch.cat((torch.stack([a_o['pred_segments'] for a_o in outputs['aux_outputs']], dim=0),
        #                         outputs['pred_segments'].unsqueeze(0)), dim=0)
        # before_bbox = outputs['references']
        # # after_bbox = outputs['pred_segments']
        # # before_bbox = outputs['references'][0]
        #
        # after_bbox = segment_ops.segment_cw_to_t1t2(after_bbox)
        # before_bbox = segment_ops.segment_cw_to_t1t2(before_bbox)
        #
        # s_delta = torch.stack((torch.minimum(after_bbox[..., 0], before_bbox[..., 0]),
        #                        torch.maximum(after_bbox[..., 0], before_bbox[..., 0])), dim=-1)
        # e_delta = torch.stack((torch.minimum(after_bbox[..., 1], before_bbox[..., 1]),
        #                        torch.maximum(after_bbox[..., 1], before_bbox[..., 1])), dim=-1)
        # a_IoU = segment_ops.batched_segment_iou(after_bbox, after_bbox)
        # b_IoU = segment_ops.batched_segment_iou(before_bbox, before_bbox)
        # # s_IoU = segment_ops.batched_segment_iou(s_delta, s_delta)
        # # e_IoU = segment_ops.batched_segment_iou(e_delta, e_delta)
        # # a_IoU = segment_ops.generalized_segment_iou(after_bbox, after_bbox)
        # # b_IoU = segment_ops.generalized_segment_iou(before_bbox, before_bbox)
        # s_IoU = segment_ops.generalized_segment_iou(s_delta, s_delta)
        # e_IoU = segment_ops.generalized_segment_iou(e_delta, e_delta)
        # # s_IoU = (s_IoU + 1.0) / 2.0
        # # e_IoU = (e_IoU + 1.0) / 2.0
        #
        # C = (s_IoU + e_IoU) / 2.0
        # # C = (a_IoU + b_IoU) / 2.0
        # # C = a_IoU
        # C = C.mean(0)
        #
        # tgt_QQ = C.softmax(dim=-1).detach()
        #
        # src_QQ = outputs["C_weights"].mean(0)
        # # src_QQ = outputs["C_weights"].flatten(0, 1)
        # src_QQ = F.normalize(src_QQ, p=2.0, dim=-1)
        # src_QQ = torch.bmm(src_QQ, src_QQ.transpose(1, 2)).softmax(dim=-1)
        #
        # src_QQ = (src_QQ.flatten(0, 1) + EPS).log()
        # tgt_QQ = (tgt_QQ.flatten(0, 1) + EPS).log()
        #
        # loss_QQ = F.kl_div(src_QQ, tgt_QQ, log_target=True, reduction="none").sum(-1).mean()
        # loss_QK = loss_QQ
        split = 0
        # bs, num_queries = outputs["pred_logits"].shape[:2]
        #
        # # We flatten to compute the cost matrices in a batch
        # out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        # out_bbox = outputs["pred_segments"].flatten(0, 1)  # [batch_size * num_queries, 4]
        #
        # # Also concat the target labels and boxes
        # tgt_ids = torch.cat([v["labels"] for v in targets if len(v["labels"])])
        # tgt_bbox = torch.cat([v["segments"] for v in targets if len(v["segments"])])
        #
        # IoUs = segment_ops.segment_iou(segment_ops.segment_cw_to_t1t2(out_bbox),
        #                                segment_ops.segment_cw_to_t1t2(tgt_bbox))
        #
        # # Compute the classification cost.
        # alpha = 0.25
        # gamma = 2.0
        # out_prob = out_prob[:, tgt_ids] * torch.sqrt(IoUs)
        # neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        # pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        # cost_class = pos_cost_class - neg_cost_class
        #
        # # Compute the L1 cost between boxes
        # cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # cost_giou = -IoUs
        #
        # # Final cost matrix
        # C = -(self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou) / \
        #     (self.cost_bbox + self.cost_class + self.cost_giou)
        # C = C.view(bs, num_queries, -1)
        #
        # tgt_QQ = list()
        # sizes = [len(v["segments"]) for v in targets]
        # for i, c in enumerate(C.split(sizes, -1)):
        #     c = F.normalize(c[i], p=2.0, dim=-1)
        #     this_tgt_QQ = torch.matmul(c, c.transpose(0, 1))
        #     tgt_QQ.append(this_tgt_QQ)
        # tgt_QQ = torch.stack(tgt_QQ).softmax(dim=-1).detach()
        #
        # src_QQ = torch.mean(outputs["C_weights"], dim=0)
        # src_QQ = F.normalize(src_QQ, p=2.0, dim=-1)
        # src_QQ = torch.bmm(src_QQ, src_QQ.transpose(1, 2)).softmax(dim=-1)
        #
        # src_QQ = (src_QQ.flatten(0, 1) + EPS).log()
        # tgt_QQ = (tgt_QQ.flatten(0, 1) + EPS).log()
        #
        # loss_QK = F.kl_div(src_QQ, tgt_QQ, log_target=True, reduction="none").sum(-1).mean()
        split = 0
        # Q_segments = outputs['pred_segments']
        # # K_segments = outputs['enc_outputs']['pred_segments']
        # nk = outputs['enc_outputs']['pred_segments'].size(1) // 2
        # K_segments = outputs['enc_outputs']['pred_segments'][:, nk:]
        #
        # IoUs = segment_ops.batched_segment_iou(segment_ops.segment_cw_to_t1t2(Q_segments),
        #                                        segment_ops.segment_cw_to_t1t2(K_segments))
        #
        # tgt_QK = IoUs.softmax(dim=-1).detach()
        #
        # src_QK = torch.mean(outputs["C_weights"], dim=0)
        #
        # src_QK = (src_QK.flatten(0, 1) + EPS).log()
        # tgt_QK = (tgt_QK.flatten(0, 1) + EPS).log()
        #
        # loss_QK = F.kl_div(src_QK, tgt_QK, log_target=True, reduction="none").sum(-1).mean()
        split = 0
        # src_prob = outputs["pred_logits"]
        # src_bbox = outputs["pred_segments"]
        # tgt_prob = outputs['enc_outputs']['pred_logits']
        # tgt_bbox = outputs['enc_outputs']['pred_segments']
        #
        # src_prob = src_prob.sigmoid()
        # tgt_prob = tgt_prob.sigmoid()
        #
        # IoUs = segment_ops.batched_segment_iou(segment_ops.segment_cw_to_t1t2(src_bbox),
        #                                        segment_ops.segment_cw_to_t1t2(tgt_bbox))
        #
        # # Compute the classification cost.
        # alpha, gamma = 0.25, 2.0
        # out_prob = torch.bmm(src_prob, tgt_prob.transpose(1, 2)) * torch.sqrt(IoUs)
        # neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        # pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        # cost_class = pos_cost_class - neg_cost_class
        #
        # # Compute the L1 cost between boxes
        # cost_bbox = torch.cdist(src_bbox, tgt_bbox, p=1)
        # cost_giou = -IoUs
        # # Final cost matrix
        # C = -(5.0 * cost_bbox + 2.0 * cost_class + 2.0 * cost_giou) / 9.0
        #
        # tgt_QK = C.softmax(dim=-1).detach()
        #
        # src_QK = torch.mean(outputs["C_weights"], dim=0)
        #
        # src_QK = (src_QK.flatten(0, 1) + EPS).log()
        # tgt_QK = (tgt_QK.flatten(0, 1) + EPS).log()
        #
        # loss_QK = F.kl_div(src_QK, tgt_QK, log_target=True, reduction="none").sum(-1).mean()
        split = 0
        # nk = outputs['enc_outputs']['pred_segments'].size(1) // 2
        # src_segments = outputs['enc_outputs']['pred_segments'][:, nk:]
        # # src_segments = torch.stack(outputs['enc_outputs']['pred_segments'].split(nk, dim=1))
        # src_boundary = segment_ops.segment_cw_to_t1t2(src_segments)
        #
        # tgt_KK = segment_ops.batched_segment_iou(src_boundary, src_boundary).softmax(dim=-1).detach()
        # # tgt_KK = segment_ops.batched_segment_iou(src_boundary, src_boundary).mean(0).softmax(dim=-1).detach()
        #
        # C_weights = torch.mean(outputs["C_weights"], dim=0)
        #
        # src_KK = torch.sqrt(torch.bmm(C_weights.transpose(1, 2), C_weights) + EPS)
        # src_KK = (src_KK / torch.sum(src_KK, dim=-1, keepdim=True))
        #
        # src_KK = (src_KK.flatten(0, 1) + EPS).log()
        # tgt_KK = (tgt_KK.flatten(0, 1) + EPS).log()
        #
        # loss_KK = F.kl_div(src_KK, tgt_KK, log_target=True, reduction="none").sum(-1).mean()
        # loss_QK = loss_QK + loss_KK
        split = 0
        # nk = outputs["C_weights"].size(3)
        # # src_prob = outputs['enc_outputs']["pred_logits"].sigmoid()
        # # src_bbox = outputs['enc_outputs']['pred_segments']
        # src_prob = outputs['enc_outputs']["pred_logits"][:, -nk:].sigmoid()
        # src_bbox = outputs['enc_outputs']['pred_segments'][:, -nk:]
        # tgt_prob = src_prob
        # tgt_bbox = src_bbox
        #
        # IoUs = segment_ops.batched_segment_iou(segment_ops.segment_cw_to_t1t2(src_bbox),
        #                                        segment_ops.segment_cw_to_t1t2(tgt_bbox))
        #
        # # Compute the classification cost.
        # # alpha, gamma = 0.25, 2.0
        # # out_prob = torch.bmm(src_prob, tgt_prob.transpose(1, 2)) * torch.sqrt(IoUs)
        # # neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        # # pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        # # cost_class = pos_cost_class - neg_cost_class
        # cost_class = torch.bmm(src_prob, tgt_prob.transpose(1, 2))
        #
        # # Compute the L1 cost between boxes
        # cost_bbox = torch.cdist(src_bbox, tgt_bbox, p=1)
        # cost_giou = -IoUs
        # # Final cost matrix
        # C = -(self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou) / \
        #     (self.cost_bbox + self.cost_class + self.cost_giou)
        #
        # # tgt_KK = C.softmax(dim=-1)
        # tgt_KK = C.softmax(dim=-1).detach()
        # # tgt_KK = inverse_sigmoid(C).softmax(dim=-1).detach()
        #
        # src_KK = torch.mean(outputs["C_weights"], dim=0)
        # src_KK = F.normalize(src_KK, p=2.0, dim=-1)
        # src_KK = torch.bmm(src_KK.transpose(1, 2), src_KK).softmax(dim=-1)
        # # src_KK = inverse_sigmoid(torch.bmm(src_KK.transpose(1, 2), src_KK)).softmax(dim=-1)
        # # src_KK = torch.sqrt(torch.bmm(src_KK.transpose(1, 2), src_KK) + EPS)
        # # src_KK = (src_KK / torch.sum(src_KK, dim=-1, keepdim=True))
        #
        # src_KK = (src_KK.flatten(0, 1) + EPS).log()
        # tgt_KK = (tgt_KK.flatten(0, 1) + EPS).log()
        #
        # loss_QK = loss_QK + F.kl_div(src_KK, tgt_KK, log_target=True, reduction="none").sum(-1).mean()
        split = 0
        # nk = outputs['enc_outputs']['pred_segments'].size(1) // 2
        # src_prob = outputs['enc_outputs']["pred_logits"][:, nk:].sigmoid()
        # src_bbox = outputs['enc_outputs']['pred_segments'][:, nk:]
        # bs, num_queries = src_prob.shape[:2]
        #
        # # We flatten to compute the cost matrices in a batch
        # out_prob = src_prob.flatten(0, 1)  # [batch_size * num_queries, num_classes]
        # out_bbox = src_bbox.flatten(0, 1)  # [batch_size * num_queries, 4]
        #
        # # Also concat the target labels and boxes
        # tgt_ids = torch.cat([v["labels"] for v in targets if len(v["labels"])])
        # tgt_bbox = torch.cat([v["segments"] for v in targets if len(v["segments"])])
        #
        # IoUs = segment_ops.segment_iou(segment_ops.segment_cw_to_t1t2(out_bbox),
        #                                segment_ops.segment_cw_to_t1t2(tgt_bbox))
        #
        # # Compute the classification cost.
        # alpha = 0.25
        # gamma = 2.0
        # out_prob = out_prob[:, tgt_ids] * torch.sqrt(IoUs)
        # neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        # pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        # cost_class = pos_cost_class - neg_cost_class
        #
        # # Compute the L1 cost between boxes
        # cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # cost_giou = -IoUs
        #
        # # Final cost matrix
        # C = -(self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou) / \
        #     (self.cost_bbox + self.cost_class + self.cost_giou)
        # C = C.view(bs, num_queries, -1)
        #
        # tgt_KK = list()
        # sizes = [len(v["segments"]) for v in targets]
        # for i, c in enumerate(C.split(sizes, -1)):
        #     c = F.normalize(c[i], p=2.0, dim=-1)
        #     this_tgt_KK = torch.matmul(c, c.transpose(0, 1))
        #     tgt_KK.append(this_tgt_KK)
        # tgt_KK = torch.stack(tgt_KK).softmax(dim=-1).detach()
        #
        # src_KK = torch.mean(outputs["C_weights"], dim=0)
        # src_KK = F.normalize(src_KK, p=2.0, dim=-1)
        # src_KK = torch.bmm(src_KK.transpose(1, 2), src_KK).softmax(dim=-1)
        #
        # src_KK = (src_KK.flatten(0, 1) + EPS).log()
        # tgt_KK = (tgt_KK.flatten(0, 1) + EPS).log()
        #
        # loss_QK = loss_QK + F.kl_div(src_KK, tgt_KK, log_target=True, reduction="none").sum(-1).mean()
        split = 0
        # Prev Main
        nk = outputs['enc_outputs']["pred_logits"].size(1) // outputs["K_weights"].size(0)
        # src_prob = outputs['enc_outputs']["pred_logits"].sigmoid()
        # src_bbox = outputs['enc_outputs']["pred_segments"]
        # src_prob = torch.stack(outputs['enc_outputs']["pred_logits"].split(nk, dim=1))[-1].sigmoid()
        # src_bbox = torch.stack(outputs['enc_outputs']["pred_segments"].split(nk, dim=1))[-1]
        src_prob = torch.stack(outputs['enc_outputs']["pred_logits"].split(nk, dim=1)).flatten(0, 1).sigmoid()
        src_bbox = torch.stack(outputs['enc_outputs']["pred_segments"].split(nk, dim=1)).flatten(0, 1)

        prev_idx = 0
        tgt_KK = list()
        tgt_conf = list()
        K = outputs["C_weights"].size(3)
        for r in self.enc_ratios:
            this_nk = K // r
            this_src_prob = src_prob[:, prev_idx:prev_idx + this_nk]
            this_src_conf = torch.max(this_src_prob, dim=-1)[0]
            this_src_bbox = src_bbox[:, prev_idx:prev_idx + this_nk]

            IoUs = segment_ops.batched_segment_iou(segment_ops.segment_cw_to_t1t2(this_src_bbox),
                                                   segment_ops.segment_cw_to_t1t2(this_src_bbox))
            # IoUs = segment_ops.generalized_segment_iou(segment_ops.segment_cw_to_t1t2(this_src_bbox),
            #                                            segment_ops.segment_cw_to_t1t2(this_src_bbox))

            # Compute the classification cost.
            cost_class = torch.bmm(this_src_prob, this_src_prob.transpose(1, 2))

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(this_src_bbox, this_src_bbox, p=1)
            cost_giou = -IoUs
            # Final cost matrix
            C = -(self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou) / \
                (self.cost_bbox + self.cost_class + self.cost_giou)

            this_src_conf = this_src_conf.unsqueeze(-1) * this_src_conf.unsqueeze(-2)
            # this_src_conf = torch.sqrt(this_src_conf.unsqueeze(-1) * this_src_conf.unsqueeze(-2))
            # this_src_conf = (this_src_conf.unsqueeze(-1) + this_src_conf.unsqueeze(-2)) / 2.0

            if this_nk < K:
                C = F.interpolate(C.unsqueeze(1), size=(K, K), mode="bilinear").squeeze(1)
                # this_src_conf = F.interpolate(this_src_conf.unsqueeze(1), size=K, mode="linear").squeeze(1)
                this_src_conf = F.interpolate(this_src_conf.unsqueeze(1), size=(K, K), mode="bilinear").squeeze(1)

            prev_idx += this_nk
            this_tgt_KK = C
            tgt_KK.append(this_tgt_KK)
            tgt_conf.append(this_src_conf)
        # tgt_KK = torch.stack(tgt_KK).mean(0).softmax(-1).detach()
        L, N = outputs["K_weights"].shape[:2]
        # tgt_KK = torch.stack(tgt_KK, dim=1).view(L, N, len(self.enc_ratios), K, K).mean(dim=(0, 2)).softmax(-1).detach()
        # multi layer
        tgt_conf = torch.stack(tgt_conf, dim=1).view(L, N, len(self.enc_ratios), K, K).softmax(dim=2)
        tgt_KK = (torch.stack(tgt_KK, dim=1).view(L, N, len(self.enc_ratios), K, K) * tgt_conf).mean(0).sum(1).softmax(-1).detach()
        # tgt_KK = (torch.stack(tgt_KK, dim=1).view(L, N, len(self.enc_ratios), K, K)).mean(0).mean(1).softmax(-1).detach()

        # tgt_conf = torch.stack(tgt_conf).softmax(dim=0)
        # tgt_KK = (torch.stack(tgt_KK) * tgt_conf).sum(0).softmax(-1).detach()
        # single layer
        # tgt_conf = torch.stack(tgt_conf, dim=1).softmax(dim=1)
        # tgt_KK = (torch.stack(tgt_KK, dim=1) * tgt_conf).sum(1).softmax(-1).detach()

        src_KK = torch.mean(outputs["C_weights"], dim=0)
        src_KK = F.normalize(src_KK, p=2.0, dim=-1)
        src_KK = torch.bmm(src_KK.transpose(1, 2), src_KK).softmax(dim=-1)
        # L, N, Q, K = outputs["C_weights"].shape
        # src_KK = outputs["C_weights"].flatten(0, 1)
        # src_KK = F.normalize(src_KK, p=2.0, dim=-1)
        # src_KK = torch.bmm(src_KK.transpose(1, 2), src_KK).softmax(dim=-1)
        # src_KK = src_KK.view(L, N, K, K).mean(0).softmax(-1)

        # N, K, K = src_KK.shape
        src_KK = (src_KK.flatten(0, 1) + EPS).log()
        tgt_KK = (tgt_KK.flatten(0, 1) + EPS).log()

        loss_KK = F.kl_div(src_KK, tgt_KK, log_target=True, reduction="none").sum(-1).mean()

        loss_QK = loss_QK + loss_KK
        split = 0

        losses = {}
        losses["loss_QK"] = loss_QK

        return losses

    def loss_QQ(self, outputs, targets, indices, num_segments):
        """Compute the actionness regression loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'Q_weights' in outputs
        assert 'C_weights' in outputs
        assert 'pred_segments' in outputs

        EPS = 1.0e-12
        split = 0
        # src_segments = torch.cat((torch.stack([a_o['pred_segments'] for a_o in outputs['aux_outputs']], dim=0),
        #                           outputs['pred_segments'].unsqueeze(0)), dim=0)
        # src_boundary = segment_ops.segment_cw_to_t1t2(src_segments.flatten(0, 1))
        #
        # tgt_QQ = segment_ops.batched_segment_iou(src_boundary, src_boundary).softmax(dim=-1).detach()
        split = 0
        # Prev Main
        # src_prob = outputs["pred_logits"].sigmoid()
        # src_bbox = outputs["pred_segments"]
        src_prob = torch.cat((torch.stack([a_o['pred_logits'] for a_o in outputs['aux_outputs']], dim=0),
                              outputs['pred_logits'].unsqueeze(0)), dim=0).flatten(0, 1)
        src_bbox = torch.cat((torch.stack([a_o['pred_segments'] for a_o in outputs['aux_outputs']], dim=0),
                              outputs['pred_segments'].unsqueeze(0)), dim=0).flatten(0, 1)
        # src_prob = torch.cat((torch.stack([a_o['pred_logits'] for a_o in outputs['aux_outputs']], dim=0),
        #                       outputs['pred_logits'].unsqueeze(0)), dim=0).flatten(0, 1)
        # src_bbox = outputs['references'].flatten(0, 1)
        tgt_prob = src_prob
        tgt_bbox = src_bbox

        src_prob = src_prob.sigmoid()
        tgt_prob = tgt_prob.sigmoid()

        IoUs = segment_ops.batched_segment_iou(segment_ops.segment_cw_to_t1t2(src_bbox),
                                               segment_ops.segment_cw_to_t1t2(tgt_bbox))
        # IoUs = segment_ops.generalized_segment_iou(segment_ops.segment_cw_to_t1t2(src_bbox),
        #                                            segment_ops.segment_cw_to_t1t2(tgt_bbox))

        # Compute the classification cost.
        # alpha, gamma = 0.25, 2.0
        # out_prob = torch.bmm(src_prob, tgt_prob.transpose(1, 2)) * torch.sqrt(IoUs)
        # neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        # pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        # cost_class = pos_cost_class - neg_cost_class
        cost_class = torch.bmm(src_prob, tgt_prob.transpose(1, 2))

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(src_bbox, tgt_bbox, p=1)
        cost_giou = -IoUs
        # Final cost matrix
        C = -(self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou) / \
            (self.cost_bbox + self.cost_class + self.cost_giou)

        # L, N, Q = outputs["Q_weights"].shape[:3]
        # tgt_QQ = C.view(L, N, Q, Q).mean(0).softmax(-1).detach()
        tgt_QQ = C.softmax(dim=-1).detach()

        # self_QQ = outputs["C_weights"].flatten(0, 1)
        # self_QQ = F.normalize(self_QQ, p=2.0, dim=-1)
        # self_QQ = torch.bmm(self_QQ, self_QQ.transpose(1, 2))
        #
        # tgt_QQ = (C + self_QQ).softmax(dim=-1).detach()

        # Q_weights = outputs["Q_weights"].mean(0)
        Q_weights = outputs["Q_weights"].flatten(0, 1)

        src_QQ = (Q_weights.flatten(0, 1) + EPS).log()
        tgt_QQ = (tgt_QQ.flatten(0, 1) + EPS).log()

        loss_QQ = F.kl_div(src_QQ, tgt_QQ, log_target=True, reduction="none").sum(-1).mean()
        # loss_QQ = F.kl_div(src_QQ, tgt_QQ, log_target=True, reduction="none")
        # loss_QQ = loss_QQ.view(N, Q, Q)
        # mask = (torch.arange(Q).unsqueeze(-1) <= torch.arange(Q).unsqueeze(0)).float().to(loss_QQ.device)
        # loss_QQ = (loss_QQ * mask).sum(-1).mean()
        split = 0
        # src_prob = torch.cat((torch.stack([a_o['pred_logits'] for a_o in outputs['aux_outputs']], dim=0),
        #                       outputs['pred_logits'].unsqueeze(0)), dim=0).flatten(0, 1)
        # src_bbox = torch.cat((torch.stack([a_o['pred_segments'] for a_o in outputs['aux_outputs']], dim=0),
        #                       outputs['pred_segments'].unsqueeze(0)), dim=0).flatten(0, 1)
        # bs, num_queries = src_prob.shape[:2]
        #
        # # We flatten to compute the cost matrices in a batch
        # out_prob = src_prob.flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        # out_bbox = src_bbox.flatten(0, 1)  # [batch_size * num_queries, 4]
        #
        # # Also concat the target labels and boxes
        # tgt_ids = torch.cat([v["labels"] for v in targets if len(v["labels"])])
        # tgt_bbox = torch.cat([v["segments"] for v in targets if len(v["segments"])])
        #
        # IoUs = segment_ops.segment_iou(segment_ops.segment_cw_to_t1t2(out_bbox),
        #                                segment_ops.segment_cw_to_t1t2(tgt_bbox))
        #
        # # Compute the classification cost.
        # alpha = 0.25
        # gamma = 2.0
        # out_prob = out_prob[:, tgt_ids] * torch.sqrt(IoUs)
        # neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        # pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        # cost_class = pos_cost_class - neg_cost_class
        #
        # # Compute the L1 cost between boxes
        # cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # cost_giou = -IoUs
        #
        # # Final cost matrix
        # C = -(self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou) / \
        #     (self.cost_bbox + self.cost_class + self.cost_giou)
        # C = C.view(len(outputs["Q_weights"]), bs // len(outputs["Q_weights"]), num_queries, -1).transpose(0, 1)
        #
        # tgt_QQ = list()
        # sizes = [len(v["segments"]) for v in targets]
        # for i, c in enumerate(C.split(sizes, -1)):
        #     c = F.normalize(c[i], p=2.0, dim=-1)
        #     this_tgt_QQ = torch.bmm(c, c.transpose(1, 2))
        #     tgt_QQ.append(this_tgt_QQ)
        # tgt_QQ = torch.stack(tgt_QQ, dim=1).flatten(0, 1).softmax(dim=-1).detach()
        split = 0
        # tgt_QQ = outputs["C_weights"].flatten(0, 1)
        # tgt_QQ = F.normalize(tgt_QQ, p=2.0, dim=-1)
        # tgt_QQ = torch.bmm(tgt_QQ, tgt_QQ.transpose(1, 2)).softmax(-1).detach()
        #
        # Q_weights = outputs["Q_weights"].flatten(0, 1)
        #
        # src_QQ = (Q_weights.flatten(0, 1) + EPS).log()
        # tgt_QQ = (tgt_QQ.flatten(0, 1) + EPS).log()
        #
        # loss_QQ = F.kl_div(src_QQ, tgt_QQ, log_target=True, reduction="none").sum(-1).mean()
        split = 0

        losses = {}
        losses['loss_QQ'] = loss_QQ

        return losses

    def loss_KK(self, outputs, targets, indices, num_segments):
        """Compute the actionness regression loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
           The target segments are expected in format (center, width), normalized by the video length.
        """
        assert 'K_weights' in outputs
        assert 'C_weights' in outputs

        EPS = 1.0e-12
        split = 0
        # C_weights = torch.mean(outputs["C_weights"], dim=0)
        # KK_weights = torch.bmm(C_weights.transpose(1, 2), C_weights)
        # KK_weights = torch.sqrt(KK_weights)
        # tgt_KK = (KK_weights / torch.sum(KK_weights, dim=-1, keepdim=True)).detach()
        #
        # K_weights = torch.mean(outputs["K_weights"], dim=0)
        #
        # src_KK = (K_weights.flatten(0, 1) + EPS).log()
        # tgt_KK = (tgt_KK.flatten(0, 1) + EPS).log()
        #
        # loss_KK = F.kl_div(src_KK, tgt_KK, log_target=True, reduction="none").sum(-1)
        # loss_KK = loss_KK.mean()
        split = 0
        # # src_segments = outputs['enc_outputs']['pred_segments']
        # nk = outputs['enc_outputs']['pred_segments'].size(1) // 2
        # src_segments = torch.stack(outputs['enc_outputs']['pred_segments'].split(nk, dim=1)).flatten(0, 1)
        # src_boundary = segment_ops.segment_cw_to_t1t2(src_segments)
        #
        # tgt_KK = segment_ops.batched_segment_iou(src_boundary, src_boundary).softmax(dim=-1).detach()
        #
        # # K_weights = outputs["K_weights"].mean(0)
        # K_weights = outputs["K_weights"].flatten(0, 1)
        #
        # src_KK = (K_weights.flatten(0, 1) + EPS).log()
        # tgt_KK = (tgt_KK.flatten(0, 1) + EPS).log()
        #
        # loss_KK = F.kl_div(src_KK, tgt_KK, log_target=True, reduction="none").sum(-1).mean()
        split = 0
        # # nk = outputs['enc_outputs']['pred_segments'].size(1) // 2
        # nk = outputs["C_weights"].size(3)
        # # src_prob = outputs['enc_outputs']['pred_logits'].sigmoid()
        # # src_bbox = outputs['enc_outputs']['pred_segments']
        # src_prob = torch.stack(outputs['enc_outputs']['pred_logits'].split(nk, dim=1)).flatten(0, 1).sigmoid()
        # src_bbox = torch.stack(outputs['enc_outputs']['pred_segments'].split(nk, dim=1)).flatten(0, 1)
        # tgt_prob = src_prob
        # tgt_bbox = src_bbox
        #
        # src_prob = src_prob.sigmoid()
        # tgt_prob = tgt_prob.sigmoid()
        #
        # IoUs = segment_ops.batched_segment_iou(segment_ops.segment_cw_to_t1t2(src_bbox),
        #                                        segment_ops.segment_cw_to_t1t2(tgt_bbox))
        #
        # # Compute the classification cost.
        # # alpha, gamma = 0.25, 2.0
        # # out_prob = torch.bmm(src_prob, tgt_prob.transpose(1, 2)) * torch.sqrt(IoUs)
        # # neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        # # pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        # # cost_class = pos_cost_class - neg_cost_class
        # cost_class = torch.bmm(src_prob, tgt_prob.transpose(1, 2))
        #
        # # Compute the L1 cost between boxes
        # cost_bbox = torch.cdist(src_bbox, tgt_bbox, p=1)
        # cost_giou = -IoUs
        # # Final cost matrix
        # C = -(self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou) / \
        #     (self.cost_bbox + self.cost_class + self.cost_giou)
        #
        # # tgt_KK = C.softmax(dim=-1)
        # tgt_KK = C.softmax(dim=-1).detach()
        # # tgt_KK = inverse_sigmoid(C).softmax(dim=-1).detach()
        #
        # # K_weights = outputs["K_weights"].mean(0)
        # K_weights = outputs["K_weights"].flatten(0, 1)
        # src_KK = (K_weights.flatten(0, 1) + EPS).log()
        # tgt_KK = (tgt_KK.flatten(0, 1) + EPS).log()
        #
        # loss_KK = F.kl_div(src_KK, tgt_KK, log_target=True, reduction="none").sum(-1).mean()
        split = 0
        # nk = outputs['enc_outputs']['pred_segments'].size(1) // 2
        # src_prob = torch.stack(outputs['enc_outputs']['pred_logits'].split(nk, dim=1)).flatten(0, 1).sigmoid()
        # src_bbox = torch.stack(outputs['enc_outputs']['pred_segments'].split(nk, dim=1)).flatten(0, 1)
        # bs, num_queries = src_prob.shape[:2]
        #
        # # We flatten to compute the cost matrices in a batch
        # out_prob = src_prob.flatten(0, 1)  # [batch_size * num_queries, num_classes]
        # out_bbox = src_bbox.flatten(0, 1)  # [batch_size * num_queries, 4]
        #
        # # Also concat the target labels and boxes
        # tgt_ids = torch.cat([v["labels"] for v in targets if len(v["labels"])])
        # tgt_bbox = torch.cat([v["segments"] for v in targets if len(v["segments"])])
        #
        # IoUs = segment_ops.segment_iou(segment_ops.segment_cw_to_t1t2(out_bbox),
        #                                segment_ops.segment_cw_to_t1t2(tgt_bbox))
        #
        # # Compute the classification cost.
        # alpha = 0.25
        # gamma = 2.0
        # out_prob = out_prob[:, tgt_ids] * torch.sqrt(IoUs)
        # neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        # pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        # cost_class = pos_cost_class - neg_cost_class
        #
        # # Compute the L1 cost between boxes
        # cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # cost_giou = -IoUs
        #
        # # Final cost matrix
        # C = -(self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou) / \
        #     (self.cost_bbox + self.cost_class + self.cost_giou)
        # C = C.view(len(outputs["K_weights"]), bs // len(outputs["K_weights"]), num_queries, -1).transpose(0, 1)
        #
        # tgt_KK = list()
        # sizes = [len(v["segments"]) for v in targets]
        # for i, c in enumerate(C.split(sizes, -1)):
        #     c = F.normalize(c[i], p=2.0, dim=-1)
        #     this_tgt_KK = torch.bmm(c, c.transpose(1, 2))
        #     tgt_KK.append(this_tgt_KK)
        # tgt_KK = torch.stack(tgt_KK, dim=1).flatten(0, 1).softmax(dim=-1).detach()
        #
        # K_weights = outputs["K_weights"].flatten(0, 1)
        #
        # src_KK = (K_weights.flatten(0, 1) + EPS).log()
        # tgt_KK = (tgt_KK.flatten(0, 1) + EPS).log()
        #
        # loss_KK = F.kl_div(src_KK, tgt_KK, log_target=True, reduction="none").sum(-1).mean()
        split = 0
        # Prev Main
        nk = outputs['enc_outputs']["pred_logits"].size(1) // outputs["K_weights"].size(0)
        # src_prob = outputs['enc_outputs']["pred_logits"].sigmoid()
        # src_bbox = outputs['enc_outputs']["pred_segments"]
        # src_prob = torch.stack(outputs['enc_outputs']["pred_logits"].split(nk, dim=1))[-1].sigmoid()
        # src_bbox = torch.stack(outputs['enc_outputs']["pred_segments"].split(nk, dim=1))[-1]
        src_prob = torch.stack(outputs['enc_outputs']["pred_logits"].split(nk, dim=1)).flatten(0, 1).sigmoid()
        src_bbox = torch.stack(outputs['enc_outputs']["pred_segments"].split(nk, dim=1)).flatten(0, 1)

        prev_idx = 0
        tgt_KK = list()
        tgt_conf = list()
        K = outputs["C_weights"].size(3)
        for r in self.enc_ratios:
            this_nk = K // r
            this_src_prob = src_prob[:, prev_idx:prev_idx + this_nk]
            this_src_conf = torch.max(this_src_prob, dim=-1)[0]
            this_src_bbox = src_bbox[:, prev_idx:prev_idx + this_nk]

            IoUs = segment_ops.batched_segment_iou(segment_ops.segment_cw_to_t1t2(this_src_bbox),
                                                   segment_ops.segment_cw_to_t1t2(this_src_bbox))
            # IoUs = segment_ops.generalized_segment_iou(segment_ops.segment_cw_to_t1t2(this_src_bbox),
            #                                            segment_ops.segment_cw_to_t1t2(this_src_bbox))

            # Compute the classification cost.
            cost_class = torch.bmm(this_src_prob, this_src_prob.transpose(1, 2))

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(this_src_bbox, this_src_bbox, p=1)
            cost_giou = -IoUs
            # Final cost matrix
            C = -(self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou) / \
                (self.cost_bbox + self.cost_class + self.cost_giou)

            this_src_conf = this_src_conf.unsqueeze(-1) * this_src_conf.unsqueeze(-2)
            # this_src_conf = torch.sqrt(this_src_conf.unsqueeze(-1) * this_src_conf.unsqueeze(-2))
            # this_src_conf = (this_src_conf.unsqueeze(-1) + this_src_conf.unsqueeze(-2)) / 2.0

            if this_nk < K:
                C = F.interpolate(C.unsqueeze(1), size=(K, K), mode="bilinear").squeeze(1)
                # this_src_conf = F.interpolate(this_src_conf.unsqueeze(1), size=K, mode="linear").squeeze(1)
                this_src_conf = F.interpolate(this_src_conf.unsqueeze(1), size=(K, K), mode="bilinear").squeeze(1)

            prev_idx += this_nk
            this_tgt_KK = C
            tgt_KK.append(this_tgt_KK)
            tgt_conf.append(this_src_conf)
        tgt_conf = torch.stack(tgt_conf).softmax(dim=0)
        # tgt_KK = (torch.stack(tgt_KK)).mean(0)
        tgt_KK = (torch.stack(tgt_KK) * tgt_conf).sum(0)
        # multi layer
        tgt_KK = tgt_KK.view(outputs["K_weights"].size(0), outputs["K_weights"].size(1), K, K).mean(0).softmax(-1).detach()
        # tgt_KK = tgt_KK.softmax(-1).detach()
        # single layer
        # tgt_KK = tgt_KK.softmax(-1).detach()

        # tgt_KK = tgt_KK.view(outputs["K_weights"].size(0), outputs["K_weights"].size(1), K, K).mean(0)

        # self_KK = outputs["C_weights"].mean(0)
        # self_KK = F.normalize(self_KK, p=2.0, dim=-1)
        # self_KK = torch.bmm(self_KK.transpose(1, 2), self_KK)
        #
        # tgt_KK = (tgt_KK + self_KK).softmax(dim=-1).detach()

        K_weights = outputs["K_weights"].mean(0)
        # K_weights = outputs["K_weights"].flatten(0, 1)

        # N, K, K = K_weights.shape
        src_KK = (K_weights.flatten(0, 1) + EPS).log()
        tgt_KK = (tgt_KK.flatten(0, 1) + EPS).log()

        loss_KK = F.kl_div(src_KK, tgt_KK, log_target=True, reduction="none").sum(-1).mean()
        # loss_KK = F.kl_div(src_KK, tgt_KK, log_target=True, reduction="none")
        # loss_KK = loss_KK.view(N, K, K)
        # mask = (torch.arange(K).unsqueeze(-1) <= torch.arange(K).unsqueeze(0)).float().to(loss_KK.device)
        # loss_KK = (loss_KK * mask).sum(-1).mean()
        split = 0
        # tgt_KK = outputs["C_weights"].mean(0)
        # tgt_KK = F.normalize(tgt_KK, p=2.0, dim=-1)
        # tgt_KK = torch.bmm(tgt_KK.transpose(1, 2), tgt_KK).softmax(-1).detach()
        #
        # K_weights = outputs["K_weights"].mean(0)
        #
        # src_KK = (K_weights.flatten(0, 1) + EPS).log()
        # tgt_KK = (tgt_KK.flatten(0, 1) + EPS).log()
        #
        # loss_KK = F.kl_div(src_KK, tgt_KK, log_target=True, reduction="none").sum(-1).mean()
        split = 0


        losses = {}
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
            'actionness': self.loss_actionness,
            'QK': self.loss_QK,
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
            if 'QQ' in loss and 'Q_weights' not in outputs:
                continue
            if 'KK' in loss and 'K_weights' not in outputs:
                continue
            if 'QK' in loss and 'C_weights' not in outputs:
                continue
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_segments, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if 'QQ' in loss or 'KK' in loss or 'QK' in loss:
                        continue

                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_segments, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])
            # for target in bin_targets:
            #     target["segments"] = target["segments"].repeat(6, 1)
            #     target["labels"] = target["labels"].repeat(6)
            # num_segments = sum(len(t["labels"]) for t in bin_targets)
            # num_segments = torch.as_tensor([num_segments], dtype=torch.float,
            #                                device=next(iter(outputs.values())).device)
            # if is_dist_avail_and_initialized():
            #     torch.distributed.all_reduce(num_segments)
            # num_segments = torch.clamp(num_segments / get_world_size(), min=1).item()

            # indices = list()
            # num_segments = list()
            # nk = outputs["K_weights"][0].size(1)
            # ks = torch.arange(nk, device=enc_outputs.device)[:, None]
            # for n_i in range(len(targets)):
            #     tgt_segs = torch.round(segment_ops.segment_cw_to_t1t2(targets[n_i]["segments"]).unsqueeze(0) * (nk - 1)).int()
            #     this_indices = torch.nonzero(torch.logical_and(ks >= tgt_segs[..., 0], ks <= tgt_segs[..., 1]))
            #     num_segments.append(len(this_indices))
            #     this_indices = (this_indices[..., 0], this_indices[..., 1])
            #     indices.append(this_indices)
            # num_segments = sum(num_segments)

            # prev_idx = 0
            # nk = outputs["C_weights"].size(3)
            # for l_i in range(len(outputs["K_weights"])):
            #     this_enc_outputs = {k:v[:, prev_idx:prev_idx + nk] for k, v in enc_outputs.items()}
            #     indices = self.matcher(this_enc_outputs, bin_targets)
            #
            #     for loss in self.losses:
            #         if 'QQ' in loss or 'KK' in loss or 'QK' in loss:
            #             continue
            #
            #         kwargs = {}
            #         if loss == 'labels':
            #             # Logging is enabled only for the last layer
            #             kwargs['log'] = False
            #
            #         l_dict = self.get_loss(loss, this_enc_outputs, targets, indices, num_segments, **kwargs)
            #         if l_i < len(outputs["K_weights"]) - 1:
            #             l_dict = {k + f'_enc_{l_i}': v for k, v in l_dict.items()}
            #         else:
            #             l_dict = {k + f'_enc': v for k, v in l_dict.items()}
            #         losses.update(l_dict)
            #
            #     prev_idx += nk

            indices = self.matcher(enc_outputs, bin_targets)

            for loss in self.losses:
                if 'QQ' in loss or 'KK' in loss or 'QK' in loss:
                    continue
                kwargs = {}
                if loss == "labels":
                    # Logging is enabled only for the last layer
                    kwargs["log"] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_segments, **kwargs)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

            # bin_targets = copy.deepcopy(targets)
            # for bt in bin_targets:
            #     bt["labels"] = torch.zeros_like(bt["labels"])
            #
            # prev_idx = 0
            # nk = outputs["enc_outputs"]["pred_logits"].size(1) // outputs["K_weights"].size(0)
            # for l_i in range(len(outputs["K_weights"])):
            #     enc_outputs = {k: v[:, prev_idx:prev_idx + nk] for k, v in outputs["enc_outputs"].items()}
            #
            #     indices = self.matcher(enc_outputs, bin_targets)
            #
            #     for loss in self.losses:
            #         if 'QQ' in loss or 'KK' in loss or 'QK' in loss:
            #             continue
            #         kwargs = {}
            #         if loss == "labels":
            #             # Logging is enabled only for the last layer
            #             kwargs["log"] = False
            #         l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_segments, **kwargs)
            #         if l_i < len(outputs["K_weights"]) - 1:
            #             l_dict = {k + f"_enc_{l_i}": v for k, v in l_dict.items()}
            #         else:
            #             l_dict = {k + "_enc": v for k, v in l_dict.items()}
            #         losses.update(l_dict)
            #     prev_idx += nk

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
        num_queries_one2one=args.num_queries_one2one,
        num_queries_one2many=args.num_queries_one2many,
        aux_loss=args.aux_loss,
        random_refpoints_xy=args.random_refpoints_xy,
        two_stage=args.two_stage,
        num_queries_enc=args.num_queries_enc
    )

    matcher = build_matcher(args)
    losses = ['labels', 'segments']

    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_segments': args.seg_loss_coef,
        'loss_iou': args.iou_loss_coef,
        'loss_mask': 5.0,
    }

    if args.use_QK:
        weight_dict["loss_QK"] = args.QK_loss_coef
        losses.append("QK")

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
        for i in range(args.enc_layers):
            if i < 1 - args.enc_layers:
                aux_weight_dict.update({k + f'_enc_{i}': v for k, v in weight_dict.items()})
            else:
                aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        # aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # new_dict = dict()
    # for key, value in weight_dict.items():
    #     new_dict[key] = value
    #     new_dict[key + "_one2many"] = value
    # weight_dict = new_dict

    if args.num_queries_one2many > 0:
        new_dict = dict()
        for key, value in weight_dict.items():
            new_dict[key] = value
            new_dict[key + "_one2many"] = value
        weight_dict = new_dict

    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha).cuda()

    postprocessor = PostProcess()

    return model, criterion, postprocessor
