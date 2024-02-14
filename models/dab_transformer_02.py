# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
import math
import copy
import os
from typing import Optional, List
from util.misc import inverse_sigmoid

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention import MultiheadAttention, RelativeAttention
from util.segment_ops import segment_cw_to_t1t2

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def gen_sineembed_for_position(pos_tensor, d_model=256):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(d_model // 2, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / (d_model // 2))
    t_embed = pos_tensor[:, :, 0] * scale
    pos_t = t_embed[:, :, None] / dim_t
    pos_t = torch.stack((pos_t[:, :, 0::2].sin(), pos_t[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 1:
        pos = pos_t
    elif pos_tensor.size(-1) == 2:
        l_embed = pos_tensor[:, :, 1] * scale
        pos_l = l_embed[:, :, None] / dim_t
        pos_l = torch.stack((pos_l[:, :, 0::2].sin(), pos_l[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_t, pos_l), dim=2)
    elif pos_tensor.size(-1) == 3:
        x_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

        y_embed = pos_tensor[:, :, 2] * scale
        pos_y = y_embed[:, :, None] / dim_t
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_t, pos_x, pos_y), dim=2)
    elif pos_tensor.size(-1) == 4:
        l_embed = pos_tensor[:, :, 1] * scale
        pos_l = l_embed[:, :, None] / dim_t
        pos_l = torch.stack((pos_l[:, :, 0::2].sin(), pos_l[:, :, 1::2].cos()), dim=3).flatten(2)

        x_embed = pos_tensor[:, :, 2] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

        y_embed = pos_tensor[:, :, 3] * scale
        pos_y = y_embed[:, :, None] / dim_t
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_t, pos_l, pos_x, pos_y), dim=2)
    elif pos_tensor.size(-1) == 5:
        x_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

        y_embed = pos_tensor[:, :, 2] * scale
        pos_y = y_embed[:, :, None] / dim_t
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

        w_embed = pos_tensor[:, :, 3] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 4] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_t, pos_x, pos_y, pos_w, pos_h), dim=2)
    elif pos_tensor.size(-1) == 6:
        x_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

        y_embed = pos_tensor[:, :, 2] * scale
        pos_y = y_embed[:, :, None] / dim_t
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

        l_embed = pos_tensor[:, :, 3] * scale
        pos_l = l_embed[:, :, None] / dim_t
        pos_l = torch.stack((pos_l[:, :, 0::2].sin(), pos_l[:, :, 1::2].cos()), dim=3).flatten(2)

        w_embed = pos_tensor[:, :, 4] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 5] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_t, pos_x, pos_y, pos_l, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class Transformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_queries_one2one=40, num_queries_one2many=0,
                 num_encoder_layers=2, num_decoder_layers=4, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=2,
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 num_patterns=0,
                 modulate_hw_attn=True,
                 bbox_embed_diff_each_layer=False,
                 num_feature_levels=1,
                 two_stage=False,
                 two_stage_num_proposals=40,
                 proposal_feature_levels=4,
                 proposal_in_stride=1,
                 proposal_tgt_strides=[1, 2, 4, 8],
                 ):

        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim,
                                          keep_query_pos=keep_query_pos, query_scale_type=query_scale_type,
                                          modulate_hw_attn=modulate_hw_attn,
                                          bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']

        self.d_model = d_model
        self.nhead = nhead
        self.enc_layers = num_encoder_layers
        self.dec_layers = num_decoder_layers
        self.num_queries_one2one = num_queries_one2one
        self.num_queries_one2many = num_queries_one2many
        self.num_queries = num_queries_one2one + num_queries_one2many
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, d_model)

        self.num_feature_levels = num_feature_levels
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            # self.enc_outputs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(self.enc_layers)])
            # self.enc_output_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.enc_layers)])
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        self.proposal_feature_levels = proposal_feature_levels
        self.proposal_tgt_strides = proposal_tgt_strides
        self.proposal_min_size = 50

        if two_stage and proposal_feature_levels > 1:
            assert len(proposal_tgt_strides) == proposal_feature_levels

            # self.proposal_in_stride = proposal_in_stride
            # self.enc_output_proj = nn.ModuleList([])
            # for stride in proposal_tgt_strides:
            #     if stride == proposal_in_stride:
            #         self.enc_output_proj.append(nn.Identity())
            #     elif stride > proposal_in_stride:
            #         scale = int(math.log2(stride / proposal_in_stride))
            #         layers = []
            #         for _ in range(scale - 1):
            #             layers += [
            #                 nn.Conv1d(d_model, d_model, kernel_size=2, stride=2),
            #                 nn.LayerNorm(d_model),
            #                 nn.GELU()
            #             ]
            #         layers.append(nn.Conv1d(d_model, d_model, kernel_size=2, stride=2))
            #         self.enc_output_proj.append(nn.Sequential(*layers))
            #     else:
            #         scale = int(math.log2(proposal_in_stride / stride))
            #         layers = []
            #         for _ in range(scale - 1):
            #             layers += [
            #                 nn.ConvTranspose1d(d_model, d_model, kernel_size=2, stride=2),
            #                 nn.LayerNorm(d_model),
            #                 nn.GELU()
            #             ]
            #         layers.append(nn.ConvTranspose1d(d_model, d_model, kernel_size=2, stride=2))
            #         self.enc_output_proj.append(nn.Sequential(*layers))

            self.proposal_in_stride = proposal_in_stride
            self.enc_output_projs = nn.ModuleList([])
            for e_i in range(self.enc_layers):
                enc_output_proj = nn.ModuleList([])
                for stride in proposal_tgt_strides:
                    if stride == proposal_in_stride:
                        enc_output_proj.append(nn.Identity())
                    elif stride > proposal_in_stride:
                        scale = int(math.log2(stride / proposal_in_stride))
                        layers = []
                        for _ in range(scale - 1):
                            layers += [
                                nn.Conv1d(d_model, d_model, kernel_size=2, stride=2),
                                nn.GroupNorm(32, d_model),
                                # LayerNorm1D(d_model),
                                # nn.GELU(),
                                nn.PReLU()
                            ]
                        layers.append(nn.Conv1d(d_model, d_model, kernel_size=2, stride=2))
                        enc_output_proj.append(nn.Sequential(*layers))
                    else:
                        scale = int(math.log2(proposal_in_stride / stride))
                        layers = []
                        for _ in range(scale - 1):
                            layers += [
                                nn.ConvTranspose1d(d_model, d_model, kernel_size=2, stride=2),
                                nn.GroupNorm(32, d_model),
                                # LayerNorm1D(d_model),
                                # nn.GELU(),
                                nn.PReLU()
                            ]
                        layers.append(nn.ConvTranspose1d(d_model, d_model, kernel_size=2, stride=2))
                        enc_output_proj.append(nn.Sequential(*layers))
                self.enc_output_projs.append(enc_output_proj)
            # self.enc_output_projs[1] = self.enc_output_projs[0]

        self.fusion_proj = MLP(d_model * num_encoder_layers, d_model, d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = self.d_model // 2
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device
        )
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack(
            (pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4
        ).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, temporal_shapes, layer_idx=0):
        if self.proposal_feature_levels > 1:
            memory, memory_padding_mask, temporal_shapes = self.expand_encoder_output(
                memory, memory_padding_mask, temporal_shapes, layer_idx
            )
        N_, S_, C_ = memory.shape
        # base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (T_, ) in enumerate(temporal_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur: (_cur + T_)].view(N_, T_, 1)
            valid_T = torch.sum(~mask_flatten_[..., 0], 1)

            grid = torch.linspace(0, T_ - 1, T_, dtype=torch.float32, device=memory.device)

            scale = valid_T.view(N_, 1)
            grid = (grid.unsqueeze(0).expand(N_, -1) + 0.5) / scale
            # w = torch.ones_like(grid) * 0.05
            w = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.stack((grid, w), -1)
            proposals.append(proposal)
            _cur += T_
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 1.0e-8) & (output_proposals < 1.0 - 1.0e-8)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        # output_memory = self.enc_output_norms[layer_idx](self.enc_outputs[layer_idx](output_memory))

        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, T = mask.shape
        valid_T = torch.sum(~mask[:, 0], 1)
        valid_ratio = valid_T.float() / T
        return valid_ratio

    def expand_encoder_output(self, memory, memory_padding_mask, temporal_shapes, layer_idx=0):
        assert temporal_shapes.size(0) == 1, f'Get encoder output of shape {temporal_shapes}, not sure how to expand'

        bs, _, c = memory.shape
        t = temporal_shapes[0]

        _out_memory = memory.permute(0, 2, 1)
        _out_memory_padding_mask = memory_padding_mask.view(bs, t)

        out_memory, out_memory_padding_mask, out_temporal_shapes = [], [], []
        for i in range(self.proposal_feature_levels):
            # mem = self.enc_output_proj[i](_out_memory)
            mem = self.enc_output_projs[layer_idx][i](_out_memory)
            mask = F.interpolate(
                _out_memory_padding_mask[None].float(), size=mem.shape[-1]
            ).to(torch.bool)

            out_memory.append(mem)
            out_memory_padding_mask.append(mask.squeeze(0))
            out_temporal_shapes.append(mem.shape[-1:])

        out_memory = torch.cat([mem.transpose(1, 2) for mem in out_memory], dim=1)
        out_memory_padding_mask = torch.cat([mask for mask in out_memory_padding_mask], dim=1)
        out_spatial_shapes = torch.as_tensor(out_temporal_shapes, dtype=torch.long, device=out_memory.device)
        return out_memory, out_memory_padding_mask, out_spatial_shapes

    def get_reference_points(self, memory, mask_flatten, temporal_shapes, layer_idx=0):
        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, temporal_shapes, layer_idx)

        # hack implementation for two-stage Deformable DETR
        enc_outputs_class = self.encoder_class_embed(output_memory)
        enc_outputs_coord = (self.encoder_segment_embed(output_memory) + output_proposals)

        if self.two_stage_num_proposals > 0:
            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 2))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
        else:
            reference_points = None
        return (reference_points, enc_outputs_class, enc_outputs_coord, output_proposals)

    def forward(self, src, mask, refpoint_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, w = src.shape
        src = src.permute(2, 0, 1)
        pos_embed = pos_embed.permute(2, 0, 1)
        refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, bs, 1)

        memory, K_weights = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # src_flatten = []
        # mask_flatten = []
        # lvl_pos_embed_flatten = []
        # temporal_shapes = []
        # for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
        #     bs, c, t = src.shape
        #     temporal_shape = (t, )
        #     temporal_shapes.append(temporal_shape)
        #     src = src.transpose(1, 2)
        #     pos_embed = pos_embed.transpose(1, 2)
        #     lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
        #     lvl_pos_embed_flatten.append(lvl_pos_embed)
        #     src_flatten.append(src)
        #     mask_flatten.append(mask)
        # src_flatten = torch.cat(src_flatten, 1)
        # mask_flatten = torch.cat(mask_flatten, 1)
        # lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # temporal_shapes = torch.as_tensor(temporal_shapes, dtype=torch.long, device=src_flatten.device)
        # level_start_index = torch.cat((temporal_shapes.new_zeros((1,)), temporal_shapes.prod(1).cumsum(0)[:-1]))
        # valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # prepare input for decoder

        # memory_flatten = torch.cat(memory, dim=0).transpose(0, 1)
        # mask_flatten = mask.repeat((1, self.enc_layers))
        # temporal_shapes = torch.as_tensor([(w, ) for _ in range(self.enc_layers)],
        #                                   dtype=torch.long, device=mask_flatten.device)

        # memory_flatten = memory[-1].transpose(0, 1)
        # mask_flatten = mask
        # temporal_shapes = torch.as_tensor(((w, ), ), dtype=torch.long, device=mask_flatten.device)

        # bs, _, c = memory_flatten.shape
        # if self.two_stage:
        #     (reference_points, enc_outputs_class, enc_outputs_coord_unact, output_proposals) \
        #         = self.get_reference_points(memory_flatten, mask_flatten, temporal_shapes)
        #     # init_reference_out = reference_points
        #     # pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(reference_points)))
        #     # enc_refpoint_embed = inverse_sigmoid(reference_points).transpose(0, 1)
        #     # refpoint_embed = torch.cat((refpoint_embed, enc_refpoint_embed), dim=0)
        # else:
        #     enc_outputs_class, enc_outputs_coord_unact = None, None

        if self.two_stage:
            enc_outputs_class = list()
            enc_outputs_coord_unact = list()
            for i in range(self.enc_layers):
                memory_flatten = memory[i].transpose(0, 1).detach()
                mask_flatten = mask
                temporal_shapes = torch.as_tensor([(w,)], dtype=torch.long, device=mask_flatten.device)

                bs, _, c = memory_flatten.shape
                # self.enc_output_proj = self.enc_output_projs[i]
                (reference_points, this_enc_outputs_class, this_enc_outputs_coord_unact, output_proposals) \
                    = self.get_reference_points(memory_flatten, mask_flatten, temporal_shapes, layer_idx=i)
                enc_outputs_class.append(this_enc_outputs_class)
                enc_outputs_coord_unact.append(this_enc_outputs_coord_unact)
            enc_outputs_class = torch.cat(enc_outputs_class, dim=1)
            enc_outputs_coord_unact = torch.cat(enc_outputs_coord_unact, dim=1)
        else:
            enc_outputs_class, enc_outputs_coord_unact = None, None

        # memory = memory[-1]
        memory = self.fusion_proj(torch.cat(memory, dim=-1))

        # query_embed = gen_sineembed_for_position(refpoint_embed)
        num_queries = refpoint_embed.shape[0]
        if self.num_patterns == 0:
            tgt = torch.zeros(num_queries, bs, self.d_model, device=refpoint_embed.device)
        else:
            tgt = self.patterns.weight[:, None, None, :].repeat(1, num_queries, bs, 1).flatten(0, 1)
            refpoint_embed = refpoint_embed.repeat(self.num_patterns, 1, 1)  # n_q*n_pat, bs, d_model

        if self.num_queries_one2many > 0:
            tgt_mask = torch.ones(dtype=torch.bool, size=(num_queries, num_queries), device=tgt.device)
            # if self.two_stage:
            #     tgt_mask[:self.two_stage_num_proposals + self.num_queries_one2one,
            #     :self.two_stage_num_proposals + self.num_queries_one2one] = False
            #     tgt_mask[self.two_stage_num_proposals + self.num_queries_one2one:,
            #     self.two_stage_num_proposals + self.num_queries_one2one:] = False
            # else:
            tgt_mask[:self.num_queries_one2one, :self.num_queries_one2one] = False
            tgt_mask[self.num_queries_one2one:, self.num_queries_one2one:] = False
            tgt_mask = tgt_mask.detach()
        else:
            tgt_mask = None

        hs, references, Q_weights, C_weights = \
            self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=mask,
                         pos=pos_embed, refpoints_unsigmoid=refpoint_embed)

        return hs, references, enc_outputs_class, enc_outputs_coord_unact, Q_weights, K_weights, C_weights


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, d_model=256):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        inter_outputs = list()
        inter_K_weights = list()
        for layer_id, layer in enumerate(self.layers):
            # rescale the content and pos sim
            pos_scales = self.query_scale(output)
            output, K_weights = layer(output, src_mask=mask,
                                      src_key_padding_mask=src_key_padding_mask,
                                      pos=pos * pos_scales)

            inter_outputs.append(output)
            inter_K_weights.append(K_weights)

        if self.norm is not None:
            output = self.norm(output)

        return inter_outputs, torch.stack(inter_K_weights)


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,
                 d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
                 modulate_hw_attn=False,
                 bbox_embed_diff_each_layer=False,
                 ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)

        self.segment_embed = None
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        if modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 1, 2)

        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
                ):
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]
        inter_Q_weights = []
        inter_C_weights = []

        # import ipdb; ipdb.set_trace()

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :self.query_dim]  # [num_queries, batch_size, 2]
            # obj_boundary = segment_cw_to_t1t2(obj_center)
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center)
            query_pos = self.ref_point_head(query_sine_embed)

            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]

            # apply transformation
            query_sine_embed = query_sine_embed[..., :self.d_model] * pos_transformation

            # modulated HW attentions
            if self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(output).sigmoid()  # nq, bs, 2
                query_sine_embed *= (refHW_cond[..., 0] / obj_center[..., 1]).unsqueeze(-1)

            # if layer_id >= 0 and True:
            #     nk = memory.size(0)
            #     alpha = 8
            #     center = torch.clamp(torch.round(obj_center * (nk - 1)).int(), 0, nk - 1).transpose(0, 1)
            #     boundary = torch.clamp(torch.round(obj_boundary * (nk - 1)).int(), 0, nk - 1).transpose(0, 1)
            #     c, w = center[..., 0], center[..., 1]
            #     s, e = boundary[..., 0], boundary[..., 1]
            #     bw = torch.clamp(torch.round(w / alpha).int(), min=1)
            #     ks = torch.arange(nk)[None, None].to(output)
            #     memory_mask = torch.logical_and(ks >= torch.clamp(s - bw, min=0).unsqueeze(-1),
            #                                     ks <= torch.clamp(e + bw, max=nk - 1).unsqueeze(-1))
            #     # memory_mask = \
            #     #     torch.logical_or(
            #     #         torch.logical_and(ks >= torch.clamp(s - bw, min=0).unsqueeze(-1),
            #     #                           ks <= torch.clamp(s + bw, max=nk - 1).unsqueeze(-1)),
            #     #         torch.logical_and(ks >= torch.clamp(e - bw, min=0).unsqueeze(-1),
            #     #                           ks <= torch.clamp(e + bw, max=nk - 1).unsqueeze(-1)))
            #     memory_mask = torch.logical_not(memory_mask).repeat(layer.nhead, 1, 1).detach()

            output, Q_weights, C_weights = \
                layer(output, memory, tgt_mask=tgt_mask,
                      memory_mask=memory_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask,
                      pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                      is_first=(layer_id == 0))

            # iter update
            if self.segment_embed is not None:
                if self.bbox_embed_diff_each_layer:
                    tmp = self.segment_embed[layer_id](output)
                else:
                    tmp = self.segment_embed(output)

                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()

                # new_reference_points = delta2bbox(inverse_sigmoid(reference_points), tmp).sigmoid()

                # nq = reference_points.size(0)
                # ref = inverse_sigmoid(reference_points)
                # # ref[:nq // 2, ..., 1] += self.static_segment_embed(output)[:nq // 2].squeeze(-1)
                # ref[nq // 2:] += tmp[nq // 2:]
                # new_reference_points = ref.sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                inter_Q_weights.append(Q_weights)
                inter_C_weights.append(C_weights)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.segment_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(ref_points).transpose(1, 2),
                    torch.stack(inter_Q_weights),
                    torch.stack(inter_C_weights),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    reference_points.unsqueeze(0).transpose(1, 2)
                ]

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn = RelativeAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2, K_weights = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)

        # q = k = src.transpose(1, 0)
        # # q = k = self.with_pos_embed(src, pos).transpose(1, 0)
        # src2, K_weights = self.self_attn(q, k, value=src.transpose(1, 0),
        #                                  attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        # src2 = src2.transpose(1, 0)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, K_weights


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False):
        super().__init__()
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            # self.self_attn = ChainAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)

            # self.sa_QK_qcontent_proj = nn.Linear(d_model, d_model)
            # self.sa_QK_qpos_proj = nn.Linear(d_model, d_model)
            # self.sa_QK_kcontent_proj = nn.Linear(d_model, d_model)
            # self.sa_QK_kpos_proj = nn.Linear(d_model, d_model)
            # self.sa_QK_v_proj = nn.Linear(d_model, d_model)
            # self.sa_QK_qpos_sine_proj = nn.Linear(d_model, d_model)
            # self.QK_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)
            #
            # self.norm0 = nn.LayerNorm(d_model)
            # self.dropout0 = nn.Dropout(dropout)
            #
            # self.sa_conv_1 = nn.Conv1d(d_model, d_model, 3, padding=1)
            # self.sa_conv_norm_1 = nn.LayerNorm(d_model)
            # self.sa_activation_1 = _get_activation_fn(activation)
            # self.sa_conv_2 = nn.Conv1d(d_model, d_model, 3, padding=1)
            # self.sa_conv_norm_2 = nn.LayerNorm(d_model)
            # self.sa_activation_2 = _get_activation_fn(activation)
            # self.sa_conv_3 = nn.Conv1d(d_model, d_model, 3, padding=1)
            # self.sa_conv_norm_3 = nn.LayerNorm(d_model)
            # self.sa_conv_dropout1 = nn.Dropout(dropout)
            #
            # self.sa_KQ_qcontent_proj = nn.Linear(d_model, d_model)
            # self.sa_KQ_qpos_proj = nn.Linear(d_model, d_model)
            # self.sa_KQ_kcontent_proj = nn.Linear(d_model, d_model)
            # self.sa_KQ_kpos_proj = nn.Linear(d_model, d_model)
            # self.sa_KQ_v_proj = nn.Linear(d_model, d_model)
            # self.sa_KQ_qpos_sine_proj = nn.Linear(d_model, d_model)
            # self.KQ_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed=None,
                is_first=False):

        if False:
            # ========== Begin of Cross-Attention =============
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.ca_qcontent_proj(tgt)
            k_content = self.ca_kcontent_proj(memory)
            v = self.ca_v_proj(memory)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            k_pos = self.ca_kpos_proj(pos)

            # For the first decoder layer, we concatenate the positional embedding predicted from
            # the object query (the positional embedding) into the original query (key) in DETR.
            if is_first or self.keep_query_pos:
                q_pos = self.ca_qpos_proj(query_pos)
                q = q_content + q_pos
                k = k_content + k_pos
            else:
                q = q_content
                k = k_content

            q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
            query_sine_embed_ = self.ca_qpos_sine_proj(query_sine_embed)
            query_sine_embed_ = query_sine_embed_.view(num_queries, bs, self.nhead, n_model // self.nhead)
            q = torch.cat([q, query_sine_embed_], dim=3).view(num_queries, bs, n_model * 2)
            k = k.view(hw, bs, self.nhead, n_model // self.nhead)
            k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
            k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

            tgt2, C_weights = self.cross_attn(query=q,
                                              key=k,
                                              value=v, attn_mask=memory_mask,
                                              key_padding_mask=memory_key_padding_mask)

            # ========== End of Cross-Attention =============
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder and True:
            q_content = self.sa_qcontent_proj(tgt)
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2, Q_weights = self.self_attn(q, k, value=v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        if True:
            # ========== Begin of Cross-Attention =============
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.ca_qcontent_proj(tgt)
            k_content = self.ca_kcontent_proj(memory)
            v = self.ca_v_proj(memory)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            k_pos = self.ca_kpos_proj(pos)

            # For the first decoder layer, we concatenate the positional embedding predicted from
            # the object query (the positional embedding) into the original query (key) in DETR.
            if is_first or self.keep_query_pos:
                q_pos = self.ca_qpos_proj(query_pos)
                q = q_content + q_pos
                k = k_content + k_pos
            else:
                q = q_content
                k = k_content

            q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
            query_sine_embed_ = self.ca_qpos_sine_proj(query_sine_embed)
            query_sine_embed_ = query_sine_embed_.view(num_queries, bs, self.nhead, n_model // self.nhead)
            q = torch.cat([q, query_sine_embed_], dim=3).view(num_queries, bs, n_model * 2)
            k = k.view(hw, bs, self.nhead, n_model // self.nhead)
            k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
            k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

            tgt2, C_weights = self.cross_attn(query=q,
                                              key=k,
                                              value=v, attn_mask=memory_mask,
                                              key_padding_mask=memory_key_padding_mask)

            # ========== End of Cross-Attention =============
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, Q_weights, C_weights


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries_one2one=args.num_queries_one2one,
        num_queries_one2many=args.num_queries_one2many,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=False,
        return_intermediate_dec=True,
        query_dim=2,
        activation="prelu",
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries_enc,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
