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

    def __init__(self, d_model=512, nhead=8, num_queries=300, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=2,
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 num_patterns=0,
                 modulate_hw_attn=True,
                 bbox_embed_diff_each_layer=False,
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
                                          d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos, query_scale_type=query_scale_type,
                                          modulate_hw_attn=modulate_hw_attn,
                                          bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)

        self._reset_parameters()
        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, refpoint_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, w = src.shape
        src = src.permute(2, 0, 1)
        pos_embed = pos_embed.permute(2, 0, 1)
        refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        memory, K_weights = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # query_embed = gen_sineembed_for_position(refpoint_embed)
        num_queries = refpoint_embed.shape[0]
        if self.num_patterns == 0:
            tgt = torch.zeros(num_queries, bs, self.d_model, device=refpoint_embed.device)
        else:
            tgt = self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1) # n_q*n_pat, bs, d_model
            refpoint_embed = refpoint_embed.repeat(self.num_patterns, 1, 1) # n_q*n_pat, bs, d_model
            # import ipdb; ipdb.set_trace()

        hs, references, Q_weights, C_weights = \
            self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, refpoints_unsigmoid=refpoint_embed)

        return hs, references, memory, Q_weights, K_weights, C_weights


class STTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_queries=300, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=2,
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 num_patterns=0,
                 modulate_hw_attn=True,
                 bbox_embed_diff_each_layer=False,
                 ):

        super().__init__()

        # encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                           dropout, activation, normalize_before)
        # encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        # self.a_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        #
        # encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        # encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        # self.o_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                  dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm, d_model=d_model)

        # decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
        #                                           dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        # decoder_norm = nn.LayerNorm(d_model)
        # self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
        #                                     return_intermediate=return_intermediate_dec,
        #                                     d_model=d_model,
        #                                     query_dim=query_dim, keep_query_pos=keep_query_pos,
        #                                     query_scale_type=query_scale_type,
        #                                     modulate_hw_attn=modulate_hw_attn,
        #                                     bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)

        decoder_layer = TransformerSTDecoderLayer(d_model, nhead, dim_feedforward,
                                                  dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerSTDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec,
                                            d_model=d_model,
                                            query_dim=query_dim, keep_query_pos=keep_query_pos,
                                            query_scale_type=query_scale_type,
                                            modulate_hw_attn=modulate_hw_attn,
                                            bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)

        # decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        # decoder_norm = nn.LayerNorm(d_model)
        # self.S_decoder = TransformerSDecoder(decoder_layer, num_decoder_layers, decoder_norm,
        #                                      return_intermediate=return_intermediate_dec,
        #                                      d_model=d_model, query_dim=3, keep_query_pos=keep_query_pos,
        #                                      query_scale_type=query_scale_type,
        #                                      modulate_hw_attn=False,
        #                                      bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)
        #
        # decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        # decoder_norm = nn.LayerNorm(d_model)
        # self.T_decoder = TransformerTDecoder(decoder_layer, num_decoder_layers, decoder_norm,
        #                                      return_intermediate=return_intermediate_dec,
        #                                      d_model=d_model, query_dim=2, keep_query_pos=keep_query_pos,
        #                                      query_scale_type=query_scale_type,
        #                                      modulate_hw_attn=modulate_hw_attn,
        #                                      bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)

        # self.ref_point_head = MLP(3 * (d_model // 2), d_model, d_model, 2)

        self._reset_parameters()
        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, refpoint_embed, pos_embed):
        # NxCxTxHW to TxHWxNxC
        bs = src.shape[0]
        src = src.permute(2, 0, 1)
        pos_embed = pos_embed.permute(2, 0, 1)
        refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = None
        # mask = mask.flatten(1)

        memory, K_weights = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        num_queries = refpoint_embed.shape[0]
        if self.num_patterns == 0:
            tgt = torch.zeros(num_queries, bs, self.d_model, device=refpoint_embed.device)
        else:
            tgt = self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1) # n_q*n_pat, bs, d_model
            refpoint_embed = refpoint_embed.repeat(self.num_patterns, 1, 1) # n_q*n_pat, bs, d_model

        hs, references, Q_weights, C_weights = \
            self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, refpoints_unsigmoid=refpoint_embed)

        # memory = memory.flatten(0, 1)
        # pos_embed = pos_embed.flatten(0, 1)

        # # query_embed = gen_sineembed_for_position(refpoint_embed)
        # num_queries = refpoint_embed[0].shape[0]
        # if self.num_patterns == 0:
        #     tgt = torch.zeros(num_queries, bs, self.d_model, device=refpoint_embed[0].device)
        #     S_refpoint_embed = refpoint_embed[0]
        # else:
        #     tgt = self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1) # n_q*n_pat, bs, d_model
        #     S_refpoint_embed = refpoint_embed[0].repeat(self.num_patterns, 1, 1) # n_q*n_pat, bs, d_model
        #
        # hs, references, Q_weights, C_weights = \
        #     self.S_decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, refpoints_unsigmoid=S_refpoint_embed)

        # num_queries = refpoint_embed[1].shape[0]
        # if self.num_patterns == 0:
        #     tgt = torch.zeros(num_queries, bs, self.d_model, device=refpoint_embed[1].device)
        #     T_refpoint_embed = refpoint_embed[1]
        # else:
        #     tgt = self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1)  # n_q*n_pat, bs, d_model
        #     T_refpoint_embed = refpoint_embed[1].repeat(self.num_patterns, 1, 1)  # n_q*n_pat, bs, d_model
        #
        # # memory = hs[-1]
        # # tmp = self.S_decoder.segment_embed(hs[-1])
        # # tmp += inverse_sigmoid(references[-1])
        # # # obj_center = tmp.sigmoid().detach()
        # # obj_center = tmp.sigmoid()
        # # query_sine_embed = gen_sineembed_for_position(obj_center, d_model=self.d_model)
        # # pos_embed = self.ref_point_head(query_sine_embed)
        #
        # # memory = memory.transpose(0, 1)
        # # pos_embed = pos_embed.transpose(0, 1)
        #
        # hs, references, Q_weights, C_weights = \
        #     self.T_decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, refpoints_unsigmoid=T_refpoint_embed)

        return hs, references, memory, Q_weights, K_weights, C_weights


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

        inter_K_weights = list()
        for layer_id, layer in enumerate(self.layers):
            # rescale the content and pos sim
            pos_scales = self.query_scale(output)
            output, K_weights = layer(output, src_mask=mask,
                                      src_key_padding_mask=src_key_padding_mask, pos=pos * pos_scales)
            inter_K_weights.append(K_weights)

        if self.norm is not None:
            output = self.norm(output)

        return output, torch.stack(inter_K_weights)


class TransformerSTEncoder(nn.Module):

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

        inter_K_weights = list()
        for layer_id, layer in enumerate(self.layers):
            # rescale the content and pos sim
            pos_scales = self.query_scale(output)
            output, K_weights = layer(output, src_mask=mask,
                                      src_key_padding_mask=src_key_padding_mask, pos=pos * pos_scales)
            inter_K_weights.append(K_weights)

        if self.norm is not None:
            output = self.norm(output)

        return output, torch.stack(inter_K_weights)


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
            obj_boundary = segment_cw_to_t1t2(obj_center)
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
                # import ipdb; ipdb.set_trace()
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
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


class TransformerSDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,
                 d_model=256, query_dim=3, keep_query_pos=False, query_scale_type='cond_elewise',
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

        self.ref_point_head = MLP(query_dim * (d_model // 2), d_model, d_model, 2)

        self.segment_embed = None
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        if modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)

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
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center, d_model=self.d_model)
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
                # refHW_cond = self.ref_anchor_head(output).sigmoid()  # nq, bs, 2
                # query_sine_embed *= (refHW_cond[..., 0] / obj_center[..., 1]).unsqueeze(-1)

                refHW_cond = self.ref_anchor_head(output).sigmoid()  # nq, bs, 2
                # query_sine_embed[..., :self.d_model // 3] *= (refHW_cond[..., 0] / obj_center[..., 3]).unsqueeze(-1)
                query_sine_embed[..., self.d_model // 3:self.d_model // 3 * 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)
                query_sine_embed[..., self.d_model // 3 * 2:] *= (refHW_cond[..., 2] / obj_center[..., 4]).unsqueeze(-1)

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
                # import ipdb; ipdb.set_trace()
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                # reference_points = new_reference_points.detach()
                reference_points = new_reference_points

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


class TransformerTDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,
                 d_model=256, query_dim=3, keep_query_pos=False, query_scale_type='cond_elewise',
                 modulate_hw_attn=True,
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

        self.ref_point_head = MLP(query_dim * (d_model // 2), d_model, d_model, 2)

        self.segment_embed = None
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        if modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)

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
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center, d_model=self.d_model)
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

                # refHW_cond = self.ref_anchor_head(output).sigmoid()  # nq, bs, 2
                # query_sine_embed[..., :self.d_model // 3] *= (refHW_cond[..., 0] / obj_center[..., 3]).unsqueeze(-1)
                # query_sine_embed[..., self.d_model // 3:self.d_model // 3 * 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)
                # query_sine_embed[..., self.d_model // 3 * 2:] *= (refHW_cond[..., 2] / obj_center[..., 4]).unsqueeze(-1)

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
                # import ipdb; ipdb.set_trace()
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
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


class TransformerSTDecoder(nn.Module):

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
            self.query_scale_center = MLP(d_model // 2, d_model // 2, d_model // 2, 2)
            self.query_scale_boundary = MLP(d_model // 2, d_model // 2, d_model // 2, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model // 2, d_model // 2, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model // 2)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))

        self.ref_point_center_head = MLP(query_dim // 2 * d_model // 2, d_model // 2, d_model // 2, 2)
        self.ref_point_boundary_head = MLP(query_dim // 2 * d_model // 2, d_model // 2, d_model // 2, 2)

        self.segment_embed = None
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        if modulate_hw_attn:
            self.ref_anchor_center_head = MLP(d_model // 2, d_model // 2, 1, 2)
            self.ref_anchor_boundary_head = MLP(d_model // 2, d_model // 2, 2, 2)

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
            obj_boundary = segment_cw_to_t1t2(obj_center)
            # get sine embedding for the query vector
            query_sine_embed_center = gen_sineembed_for_position(obj_center, d_model=self.d_model // 2)
            query_sine_embed_boundary = gen_sineembed_for_position(obj_boundary, d_model=self.d_model // 2)
            query_pos_center = self.ref_point_center_head(query_sine_embed_center)
            query_pos_boundary = self.ref_point_boundary_head(query_sine_embed_boundary)

            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation_center, pos_transformation_boundary = 1, 1
                else:
                    pos_transformation_center = self.query_scale_center(output[..., :self.d_model // 2])
                    pos_transformation_boundary = self.query_scale_boundary(output[..., self.d_model // 2:])
            else:
                pos_transformation_center = self.query_scale_center.weight[layer_id]
                pos_transformation_boundary = self.query_scale_boundary.weight[layer_id]

            # apply transformation
            query_sine_embed_center = query_sine_embed_center[..., :self.d_model // 2] * pos_transformation_center
            query_sine_embed_boundary = query_sine_embed_boundary[..., :self.d_model // 2] * pos_transformation_boundary

            # modulated HW attentions
            if self.modulate_hw_attn:
                refHW_cond_center = self.ref_anchor_center_head(output[..., :self.d_model // 2]).sigmoid()  # nq, bs, 2
                query_sine_embed_center *= (refHW_cond_center[..., 0] / obj_center[..., 1]).unsqueeze(-1)

                refHW_cond_boundary = self.ref_anchor_boundary_head(output[..., :self.d_model // 2:]).sigmoid()  # nq, bs, 2
                query_sine_embed_boundary *= (refHW_cond_boundary[..., 0] / (obj_center[..., 1] / 2)).unsqueeze(-1)
                query_sine_embed_boundary *= (refHW_cond_boundary[..., 1] / (obj_center[..., 1] / 2)).unsqueeze(-1)

            query_pos = (query_pos_center, query_pos_boundary)
            query_sine_embed = (query_sine_embed_center, query_sine_embed_boundary)

            nb = output.size(1)
            nq = output.size(0)
            nk = memory.size(0)
            center_memory_mask = torch.ones(dtype=torch.bool, size=(nb, nq, nk), device=output.device)
            boundary_memory_mask = torch.ones(dtype=torch.bool, size=(nb, nq, nk), device=output.device)
            for n_i in range(nb):
                for q_i, (center, boundary) in enumerate(zip(obj_center[:, n_i], obj_boundary[:, n_i])):
                    center = torch.clamp(torch.round(center * (nk - 1)).int(), 0, nk - 1)
                    boundary = torch.clamp(torch.round(boundary * (nk - 1)).int(), 0, nk - 1)
                    c, w = center[..., 0], center[..., 1]
                    s, e = boundary[..., 0], boundary[..., 1]
                    bw = torch.clamp(torch.round(w / 8).int(), min=1)
                    center_memory_mask[n_i, q_i, s:e + 1] = False
                    boundary_memory_mask[n_i, q_i, torch.clamp(s - bw, min=0):torch.clamp(s + bw + 1, max=nk)] = False
                    boundary_memory_mask[n_i, q_i, torch.clamp(e - bw, min=0):torch.clamp(e + bw + 1, max=nk)] = False
            center_memory_mask = center_memory_mask.repeat(layer.nhead, 1, 1).detach()
            boundary_memory_mask = boundary_memory_mask.repeat(layer.nhead, 1, 1).detach()
            memory_mask = (center_memory_mask, boundary_memory_mask)

            # tgt_mask = torch.ones(dtype=torch.bool, size=(nq, nq), device=output.device)
            # tgt_mask[:nq // 6, :nq // 6] = False
            # tgt_mask[nq // 6:, nq // 6:] = False

            output, Q_weights, C_weights = \
                layer(output, memory,
                      tgt_mask=tgt_mask,
                      memory_mask=memory_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask,
                      pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                      is_first=(layer_id == 0))

            # iter update
            if self.segment_embed is not None:
                if self.bbox_embed_diff_each_layer:
                    tmp = self.segment_embed[layer_id](output[..., self.d_model // 2:])
                else:
                    tmp = self.segment_embed(output[..., self.d_model // 2:])
                # import ipdb; ipdb.set_trace()
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
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


class TransformerSTEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.S_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.T_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1_1 = nn.LayerNorm(d_model)
        self.norm1_2 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1_1 = nn.Dropout(dropout)
        self.dropout1_2 = nn.Dropout(dropout)
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

        # src2, K_weights = self.S_self_attn(q.mean(dim=0), k.mean(0), value=src.mean(0),
        #                                    attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        # src = src + self.dropout1_1(src2).unsqueeze(0)

        # T, HW, N, C -> HW, TN, C
        T, HW, N, C = q.shape
        q = q.transpose(0, 1).flatten(1, 2)
        k = k.transpose(0, 1).flatten(1, 2)
        src = src.transpose(0, 1).flatten(1, 2)
        src2, K_weights = self.S_self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1_1(src2)
        src = src.view(HW, T, N, C).transpose(0, 1)

        src = self.norm1_1(src)

        q = k = self.with_pos_embed(src, pos)

        # src2, K_weights = self.T_self_attn(q.mean(1), k.mean(1), value=src.mean(1),
        #                                    attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        # src = src + self.dropout1_2(src2).unsqueeze(1)

        # T, HW, N, C -> T, HWN, C
        q = q.flatten(1, 2)
        k = k.flatten(1, 2)
        src = src.flatten(1, 2)
        src2, K_weights = self.T_self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1_2(src2)
        src = src.view(T, HW, N, C)

        src = self.norm1_2(src)

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
        self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)

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

            q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
            query_sine_embed_ = self.ca_qpos_sine_proj(query_sine_embed)
            query_sine_embed_ = query_sine_embed_.view(num_queries, bs, self.nhead, n_model//self.nhead)
            q = torch.cat([q, query_sine_embed_], dim=3).view(num_queries, bs, n_model * 2)
            k = k.view(hw, bs, self.nhead, n_model//self.nhead)
            k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
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

            q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
            query_sine_embed_ = self.ca_qpos_sine_proj(query_sine_embed)
            query_sine_embed_ = query_sine_embed_.view(num_queries, bs, self.nhead, n_model//self.nhead)
            q = torch.cat([q, query_sine_embed_], dim=3).view(num_queries, bs, n_model * 2)
            k = k.view(hw, bs, self.nhead, n_model//self.nhead)
            k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
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


class TransformerSTDecoderLayer(nn.Module):

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

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

            # self.S_sa_qcontent_proj = nn.Linear(d_model // 2, d_model // 2)
            # self.S_sa_qpos_proj = nn.Linear(d_model, d_model // 2)
            # self.S_sa_kcontent_proj = nn.Linear(d_model // 2, d_model // 2)
            # self.S_sa_kpos_proj = nn.Linear(d_model, d_model // 2)
            # self.S_sa_v_proj = nn.Linear(d_model // 2, d_model // 2)
            # self.S_self_attn = MultiheadAttention(d_model // 2, nhead, dropout=dropout, vdim=d_model // 2)
            #
            # self.T_sa_qcontent_proj = nn.Linear(d_model // 2, d_model // 2)
            # self.T_sa_qpos_proj = nn.Linear(d_model, d_model // 2)
            # self.T_sa_kcontent_proj = nn.Linear(d_model // 2, d_model // 2)
            # self.T_sa_kpos_proj = nn.Linear(d_model, d_model // 2)
            # self.T_sa_v_proj = nn.Linear(d_model // 2, d_model // 2)
            # self.T_self_attn = MultiheadAttention(d_model // 2, nhead, dropout=dropout, vdim=d_model // 2)
            #
            # self.norm1_1 = nn.LayerNorm(d_model // 2)
            # self.dropout1_1 = nn.Dropout(dropout)
            # self.norm1_2 = nn.LayerNorm(d_model // 2)
            # self.dropout1_2 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.S_ca_qcontent_proj = nn.Linear(d_model // 2, d_model // 2)
        self.S_ca_qpos_proj = nn.Linear(d_model // 2, d_model // 2)
        self.S_ca_kcontent_proj = nn.Linear(d_model, d_model // 2)
        self.S_ca_kpos_proj = nn.Linear(d_model, d_model // 2)
        self.S_ca_v_proj = nn.Linear(d_model, d_model // 2)
        self.S_ca_qpos_sine_proj = nn.Linear(d_model // 2, d_model // 2)
        self.S_cross_attn = MultiheadAttention(d_model // 2 * 2, nhead, dropout=dropout, vdim=d_model // 2)

        self.T_ca_qcontent_proj = nn.Linear(d_model // 2, d_model // 2)
        self.T_ca_qpos_proj = nn.Linear(d_model // 2, d_model // 2)
        self.T_ca_kcontent_proj = nn.Linear(d_model, d_model // 2)
        self.T_ca_kpos_proj = nn.Linear(d_model, d_model // 2)
        self.T_ca_v_proj = nn.Linear(d_model, d_model // 2)
        self.T_ca_qpos_sine_proj = nn.Linear(d_model // 2, d_model // 2)
        self.T_cross_attn = MultiheadAttention(d_model // 2 * 2, nhead, dropout=dropout, vdim=d_model // 2)

        self.norm2_1 = nn.LayerNorm(d_model // 2)
        self.norm2_2 = nn.LayerNorm(d_model // 2)
        self.dropout2_1 = nn.Dropout(dropout)
        self.dropout2_2 = nn.Dropout(dropout)

        self.d_model = d_model
        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.dropout = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # self.linear1_1 = nn.Linear(d_model // 2, dim_feedforward // 2)
        # self.linear1_2 = nn.Linear(d_model // 2, dim_feedforward // 2)
        #
        # self.linear2_1 = nn.Linear(dim_feedforward // 2, d_model // 2)
        # self.linear2_2 = nn.Linear(dim_feedforward // 2, d_model // 2)
        #
        # self.norm3_1 = nn.LayerNorm(d_model // 2)
        # self.norm3_2 = nn.LayerNorm(d_model // 2)
        #
        # self.dropout_1 = nn.Dropout(dropout)
        # self.dropout_2 = nn.Dropout(dropout)
        # self.dropout3_1 = nn.Dropout(dropout)
        # self.dropout3_2 = nn.Dropout(dropout)

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

        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder and True:
            combined_query_pos = torch.cat(query_pos, dim=-1)
            q_content = self.sa_qcontent_proj(tgt)
            q_pos = self.sa_qpos_proj(combined_query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(combined_query_pos)
            v = self.sa_v_proj(tgt)

            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2, Q_weights = self.self_attn(q, k, value=v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # if not self.rm_self_attn_decoder and True:
        #     C_tgt = tgt[..., :self.d_model // 2]
        #     C_query_pos = query_pos[0]
        #
        #     q_content = self.S_sa_qcontent_proj(C_tgt)
        #     q_pos = self.S_sa_qpos_proj(C_query_pos)
        #     k_content = self.S_sa_kcontent_proj(C_tgt)
        #     k_pos = self.S_sa_kpos_proj(C_query_pos)
        #     v = self.S_sa_v_proj(C_tgt)
        #
        #     hw, _, _ = k_content.shape
        #
        #     q = q_content + q_pos
        #     k = k_content + k_pos
        #
        #     tgt2, Q_weights = self.S_self_attn(q, k, value=v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        #
        #     C_tgt = C_tgt + self.dropout1_1(tgt2)
        #     C_tgt = self.norm1_1(C_tgt)
        #
        # if not self.rm_self_attn_decoder and True:
        #     L_tgt = tgt[..., self.d_model // 2:]
        #     L_query_pos = query_pos[1]
        #
        #     q_content = self.T_sa_qcontent_proj(L_tgt)
        #     q_pos = self.T_sa_qpos_proj(L_query_pos)
        #     k_content = self.T_sa_kcontent_proj(L_tgt)
        #     k_pos = self.T_sa_kpos_proj(L_query_pos)
        #     v = self.T_sa_v_proj(L_tgt)
        #
        #     hw, _, _ = k_content.shape
        #
        #     q = q_content + q_pos
        #     k = k_content + k_pos
        #
        #     tgt2, Q_weights = self.T_self_attn(q, k, value=v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        #
        #     L_tgt = L_tgt + self.dropout1_2(tgt2)
        #     L_tgt = self.norm1_2(L_tgt)

        if True:
            # ========== Begin of Cross-Attention =============
            # Apply projections here
            # shape: num_queries x batch_size x 256
            C_tgt = tgt[..., :self.d_model // 2]
            C_query_sine_embed = query_sine_embed[0]
            C_query_pos = query_pos[0]
            q_content = self.S_ca_qcontent_proj(C_tgt)
            k_content = self.S_ca_kcontent_proj(memory)
            v = self.S_ca_v_proj(memory)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            k_pos = self.S_ca_kpos_proj(pos)

            # For the first decoder layer, we concatenate the positional embedding predicted from
            # the object query (the positional embedding) into the original query (key) in DETR.
            if is_first or self.keep_query_pos:
                q_pos = self.S_ca_qpos_proj(C_query_pos)
                q = q_content + q_pos
                k = k_content + k_pos
            else:
                q = q_content
                k = k_content

            q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
            query_sine_embed_ = self.S_ca_qpos_sine_proj(C_query_sine_embed)
            query_sine_embed_ = query_sine_embed_.view(num_queries, bs, self.nhead, n_model // self.nhead)
            q = torch.cat([q, query_sine_embed_], dim=3).view(num_queries, bs, n_model * 2)
            k = k.view(hw, bs, self.nhead, n_model // self.nhead)
            k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
            k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

            tgt2, C_C_weights = self.S_cross_attn(query=q,
                                                  key=k,
                                                  value=v, attn_mask=memory_mask[0],
                                                  key_padding_mask=memory_key_padding_mask)

            # ========== End of Cross-Attention =============
            C_tgt = C_tgt + self.dropout2_1(tgt2)
            C_tgt = self.norm2_1(C_tgt)

            # tgt2 = self.linear2_1(self.dropout(self.activation(self.linear1_1(C_tgt))))
            # C_tgt = C_tgt + self.dropout3_1(tgt2)
            # C_tgt = self.norm3_1(C_tgt)

        if True:
            # ========== Begin of Cross-Attention =============
            # Apply projections here
            # shape: num_queries x batch_size x 256
            L_tgt = tgt[..., self.d_model // 2:]
            L_query_sine_embed = query_sine_embed[1]
            L_query_pos = query_pos[1]
            q_content = self.T_ca_qcontent_proj(L_tgt)
            k_content = self.T_ca_kcontent_proj(memory)
            v = self.T_ca_v_proj(memory)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            k_pos = self.T_ca_kpos_proj(pos)

            # For the first decoder layer, we concatenate the positional embedding predicted from
            # the object query (the positional embedding) into the original query (key) in DETR.
            if is_first or self.keep_query_pos:
                q_pos = self.T_ca_qpos_proj(L_query_pos)
                q = q_content + q_pos
                k = k_content + k_pos
            else:
                q = q_content
                k = k_content

            q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
            query_sine_embed_ = self.T_ca_qpos_sine_proj(L_query_sine_embed)
            query_sine_embed_ = query_sine_embed_.view(num_queries, bs, self.nhead, n_model // self.nhead)
            q = torch.cat([q, query_sine_embed_], dim=3).view(num_queries, bs, n_model * 2)
            k = k.view(hw, bs, self.nhead, n_model // self.nhead)
            k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
            k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

            tgt2, L_C_weights = self.T_cross_attn(query=q,
                                                  key=k,
                                                  value=v, attn_mask=memory_mask[1],
                                                  key_padding_mask=memory_key_padding_mask)

            # ========== End of Cross-Attention =============
            L_tgt = L_tgt + self.dropout2_2(tgt2)
            L_tgt = self.norm2_2(L_tgt)

            # tgt2 = self.linear2_2(self.dropout(self.activation(self.linear1_2(L_tgt))))
            # L_tgt = L_tgt + self.dropout3_2(tgt2)
            # L_tgt = self.norm3_2(L_tgt)

        tgt = torch.cat((C_tgt, L_tgt), dim=-1)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, Q_weights, torch.stack((C_C_weights, L_C_weights), dim=-1)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=False,
        return_intermediate_dec=True,
        query_dim=6,
        activation="prelu"
    )

def build_ST_transformer(args):
    return STTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=False,
        return_intermediate_dec=True,
        query_dim=2,
        activation="prelu"
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
