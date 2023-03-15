import math
import copy

import torch
import torch.nn.functional as F
from torch import nn

from util import segment_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from models.matcher import build_matcher
from models.position_encoding import build_position_encoding
from .custom_loss import sigmoid_focal_loss
from .transformer import build_transformer
from opts import cfg
import numpy as np


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


class SelfDETR(nn.Module):

    def __init__(self, position_embedding, transformer, num_classes, num_queries,
                 aux_loss=True, with_segment_refine=True, query_dim=2):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.segment_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        self.query_dim = query_dim

        self.refpoint_embed = nn.Embedding(num_queries, query_dim)

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

        if with_segment_refine:
            # hack implementation for segment refinement
            self.transformer.decoder.segment_embed = self.segment_embed

    def forward(self, samples):
        if not isinstance(samples, NestedTensor):
            if isinstance(samples, (list, tuple)):
                samples = NestedTensor(*samples)
            else:
                samples = nested_tensor_from_tensor_list(samples)

        pos = [self.position_embedding(samples)]
        src, mask = samples.tensors, samples.mask

        embedweight = self.refpoint_embed.weight
        hs, reference, memory, Q_weights, K_weights, C_weights = \
            self.transformer(self.input_proj[0](src), mask, embedweight, pos[-1])

        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.segment_embed(hs)
        tmp[..., :self.query_dim] += reference_before_sigmoid
        outputs_coord = tmp.sigmoid()

        outputs_class = self.class_embed(hs)

        out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_coord[-1],
               'Q_weights': Q_weights, 'K_weights': K_weights, 'C_weights': C_weights}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_segments': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_segments, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_segments, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]  # nq
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def loss_segments(self, outputs, targets, indices, num_segments):
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

    def loss_QQ(self, outputs, targets, indices, num_segments):
        assert 'Q_weights' in outputs
        assert 'C_weights' in outputs

        Q_weights = outputs["Q_weights"].flatten(0, 1)

        C_weights = outputs["C_weights"].flatten(0, 1).detach()
        QQ_weights = torch.sqrt(torch.bmm(C_weights, C_weights.transpose(1, 2)) + 1.0e-7)
        target_Q_weights = QQ_weights / torch.sum(QQ_weights, dim=-1, keepdim=True)

        src_QQ = (Q_weights.flatten(0, 1) + 1.0e-7).log()
        tgt_QQ = (target_Q_weights.flatten(0, 1) + 1.0e-7).log()

        losses = {}

        loss_QQ = F.kl_div(src_QQ, tgt_QQ, log_target=True, reduction="none").sum(-1)

        loss_QQ = loss_QQ.mean()

        losses['loss_QQ'] = loss_QQ

        return losses

    def loss_KK(self, outputs, targets, indices, num_segments):
        assert 'K_weights' in outputs
        assert 'C_weights' in outputs

        K_weights = outputs["K_weights"]
        normalized_K_weights = K_weights[0]
        for i in range(len(K_weights) - 1):
            normalized_K_weights = torch.sqrt(
                torch.bmm(normalized_K_weights, K_weights[i + 1].transpose(1, 2)) + 1.0e-7)
            normalized_K_weights = normalized_K_weights / torch.sum(normalized_K_weights, dim=-1, keepdim=True)
        K_weights = normalized_K_weights

        C_weights = torch.mean(outputs["C_weights"], dim=0).detach()
        KK_weights = torch.bmm(C_weights.transpose(1, 2), C_weights)
        KK_weights = torch.sqrt(KK_weights + 1.0e-7)
        target_K_weights = KK_weights / torch.sum(KK_weights, dim=-1, keepdim=True)

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
            "QQ": self.loss_QQ,
            "KK": self.loss_KK,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_segments, **kwargs)

    def forward(self, outputs, targets):
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
                    if 'QQ' in loss or 'KK' in loss:
                        continue

                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_segments, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        self.indices = indices
        return losses


class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_segments = outputs['pred_logits'], outputs['pred_segments']

        assert len(out_logits) == len(target_sizes)

        prob = out_logits.sigmoid()  # [bs, nq, C]

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

    model = SelfDETR(
        pos_embed,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        with_segment_refine=args.seg_refine)

    matcher = build_matcher(args)
    losses = ['labels', 'segments']

    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_segments': args.seg_loss_coef,
        'loss_iou': args.iou_loss_coef}

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
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(num_classes, matcher,
                             weight_dict, losses, focal_alpha=args.focal_alpha)

    postprocessor = PostProcess()

    return model, criterion, postprocessor
