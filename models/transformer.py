# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
# and DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.temporal_deform_attn import DeformAttn
from opts import cfg
from util.nms import dynamic_nms
from util.segment_ops import segment_cw_to_t1t2


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4, use_dab=True):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.use_dab = use_dab

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                        dropout, activation,
                                                        num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, d_model, return_intermediate_dec,
                                                    use_dab=use_dab)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.reference_points = nn.Linear(d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, DeformAttn):
                m._reset_parameters()

        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, T = mask.shape
        valid_T = torch.sum(~mask, 1)
        valid_ratio = valid_T.float() / T
        return valid_ratio    # shape=(bs)

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        '''
        Params:
            srcs: list of Tensor with shape (bs, c, t)
            masks: list of Tensor with shape (bs, t)
            pos_embeds: list of Tensor with shape (bs, c, t)
            query_embed: list of Tensor with shape (nq, 2c)
        Returns:
            hs: list, per layer output of decoder
            init_reference_out: reference points predicted from query embeddings
            inter_references_out: reference points predicted from each decoder layer
            memory: (bs, c, t), final output of the encoder
        '''
        assert query_embed is not None
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        temporal_lens = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, t = src.shape
            temporal_lens.append(t)
            # (bs, c, t) => (bs, t, c)
            src = src.transpose(1, 2)   
            pos_embed = pos_embed.transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        temporal_lens = torch.as_tensor(temporal_lens, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((temporal_lens.new_zeros((1, )), temporal_lens.cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)   # (bs, nlevels)

        # deformable encoder
        memory, K_weights, K_in, K_out = self.encoder(src_flatten, temporal_lens, level_start_index, valid_ratios,
            lvl_pos_embed_flatten if cfg.use_pos_embed else None, 
            mask_flatten)  # shape=(bs, t, c)

        bs, _, c = memory.shape

        if not self.use_dab:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points
        else:
            reference_points = query_embed[..., self.d_model:].sigmoid()
            tgt = query_embed[..., :self.d_model]
            # reference_points = query_embed[..., self.d_model * 3:].sigmoid()
            # tgt = query_embed[..., :self.d_model * 3]
            init_reference_out = reference_points
            query_embed = None
        # decoder
        hs, inter_references, Q_weights, C_weights, Q_in, Q_out, C_in, C_out = \
            self.decoder(tgt, reference_points, memory, lvl_pos_embed_flatten,
                         temporal_lens, level_start_index, valid_ratios, query_embed, mask_flatten)
        inter_references_out = inter_references 
        return hs, init_reference_out, inter_references_out, memory.transpose(1, 2), \
               Q_weights, K_weights, C_weights, K_in, K_out, Q_in, Q_out, C_in, C_out


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        # self.self_attn = DeformAttn(d_model, n_levels, n_heads, n_points)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # pos = None
        # self attention
        # src2, _ = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        q = k = self.with_pos_embed(src, pos)
        K_in = src
        src2, K_weights = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), src.transpose(0, 1))
        K_out = src2.transpose(0, 1)

        # print(torch.argsort(-K_weights[0].detach().cpu(), dim=-1)[:10, :10].numpy())

        src2 = src2.transpose(0, 1)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src, K_weights, K_in, K_out


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, T_ in enumerate(spatial_shapes):
            ref = torch.linspace(0.5, T_ - 0.5, T_, dtype=torch.float32, device=device)  # (t,)
            ref = ref[None] / (valid_ratios[:, None, lvl] * T_)                          # (bs, t)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]          # (N, t, n_levels)
        return reference_points[..., None]                                               # (N, t, n_levels, 1)

    def forward(self, src, temporal_lens, level_start_index, valid_ratios, pos=None, padding_mask=None):
        '''
        src: shape=(bs, t, c)
        temporal_lens: shape=(n_levels). content: [t1, t2, t3, ...]
        level_start_index: shape=(n_levels,). [0, t1, t1+t2, ...]
        valid_ratios: shape=(bs, n_levels).
        '''
        output = src
        # (bs, t, levels, 1)
        inter_K_weights = list()
        inter_K_in = list()
        inter_K_out = list()
        reference_points = self.get_reference_points(temporal_lens, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output, K_weights, K_in, K_out = \
                layer(output, pos, reference_points, temporal_lens, level_start_index, padding_mask)
            inter_K_weights.append(K_weights)
            inter_K_in.append(K_in)
            inter_K_out.append(K_out)
        return output, torch.stack(inter_K_weights), torch.stack(inter_K_in), torch.stack(inter_K_out)


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        # self.cross_attn = DeformAttn(d_model, n_levels, n_heads, n_points)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        # self.self_attn = DeformAttn(d_model, 5, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_pos, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # query_pos = None
        if not cfg.disable_query_self_att or True:
            # self attention
            q = k = self.with_pos_embed(tgt, query_pos)
            # q = k = query_pos
            Q_in = tgt
            tgt2, Q_weights = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))
            tgt2 = tgt2.transpose(0, 1)
            Q_out = tgt2

            # print(F.cross_entropy(Q_weights, Q_weights).sum(-1).mean().detach().cpu().numpy())
            #
            # q = k = tgt
            # _, C_weights = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))
            # q = k = query_pos
            # _, P_weights = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))
            #
            # N, Q, _ = Q_weights.shape
            # Q_C = torch.bmm(F.normalize(Q_weights.flatten(1)).unsqueeze(-2),
            #                 F.normalize(C_weights.flatten(1)).unsqueeze(-1)).mean()
            # Q_P = torch.bmm(F.normalize(Q_weights.flatten(1)).unsqueeze(-2),
            #                 F.normalize(P_weights.flatten(1)).unsqueeze(-1)).mean()
            #
            # print(Q_C.detach().cpu().numpy(), Q_P.detach().cpu().numpy())

            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        else:
            pass
        # cross attention
        # tgt2, _ = self.cross_attn(self.with_pos_embed(tgt, query_pos),
        #                        reference_points,
        #                        src, src_spatial_shapes, level_start_index, src_padding_mask)
        # tgt2, C_weights = self.cross_attn(query_pos,
        #                                   reference_points,
        #                                   src, src_spatial_shapes, level_start_index, src_padding_mask)
        C_in = tgt
        tgt2, C_weights = self.cross_attn(query=self.with_pos_embed(tgt, query_pos).transpose(0, 1),
                                          key=self.with_pos_embed(src, src_pos).transpose(0, 1),
                                          value=src.transpose(0, 1),
                                          key_padding_mask=src_padding_mask)
        tgt2 = tgt2.transpose(0, 1)
        C_out = tgt2
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt, Q_weights, C_weights, Q_in, Q_out, C_in, C_out


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, d_model, return_intermediate=False, use_dab=True):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        # self.S_layers = _get_clones(decoder_layer, num_layers)
        # self.E_layers = _get_clones(decoder_layer, num_layers)
        # self.C_layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model
        self.use_dab = use_dab
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.segment_embed = None
        # self.S_segment_embed = None
        # self.E_segment_embed = None
        self.class_embed = None

        self.query_scale = MLP(d_model, d_model, d_model, 2)
        # self.S_query_scale = MLP(d_model, d_model, d_model, 2)
        # self.E_query_scale = MLP(d_model, d_model, d_model, 2)
        # self.C_query_scale = MLP(d_model, d_model, d_model, 2)
        self.ref_point_head = MLP(2, d_model, d_model, 3)
        # self.S_ref_point_head = MLP(2, d_model, d_model, 3)
        # self.E_ref_point_head = MLP(2, d_model, d_model, 3)
        # self.C_ref_point_head = MLP(2, d_model, d_model, 3)

    def forward(self, tgt, reference_points, src, src_pos, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        '''
        tgt: [bs, nq, C]
        reference_points: [bs, nq, 1 or 2]
        src: [bs, T, C]
        src_valid_ratios: [bs, levels]
        '''
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        inter_Q_weights = []
        inter_C_weights = []
        inter_Q_in = []
        inter_Q_out = []
        inter_C_in = []
        inter_C_out = []
        for lid, layer in enumerate(self.layers):
        # for lid in range(self.num_layers):
            # (bs, nq, 1, 1 or 2) x (bs, 1, num_level, 1) => (bs, nq, num_level, 1 or 2)
            reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None,:, None]
            if self.use_dab:
                raw_query_pos = self.ref_point_head(reference_points_input[:, :, 0, :])
                pos_scale = self.query_scale(output) if lid != 0 else 1
                query_pos = pos_scale * raw_query_pos

            output, Q_weights, C_weights, Q_in, Q_out, C_in, C_out = \
                layer(output, query_pos, reference_points_input,
                      src, src_pos, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # W = torch.clamp(reference_points_input[..., 1] - reference_points_input[..., 0], 0.0, 1.0)
            # C = torch.clamp(reference_points_input[..., 0] + reference_points_input[..., 1] / 2.0, 0.0, 1.0)
            #
            # S_output = output[..., :self.d_model]
            # S_ref_points = torch.stack((reference_points_input[..., 0], W), dim=-1)
            # if self.use_dab:
            #     raw_query_pos = self.S_ref_point_head(reference_points_input[:, :, 0, :])
            #     pos_scale = self.S_query_scale(S_output) if lid != 0 else 1
            #     query_pos = pos_scale * raw_query_pos
            # S_output = self.S_layers[lid](S_output, query_pos, S_ref_points,
            #                               src, src_spatial_shapes, src_level_start_index, src_padding_mask)
            #
            # E_output = output[..., self.d_model:self.d_model * 2]
            # E_ref_points = torch.stack((reference_points_input[..., 1], W), dim=-1)
            # if self.use_dab:
            #     raw_query_pos = self.E_ref_point_head(reference_points_input[:, :, 0, :])
            #     pos_scale = self.E_query_scale(E_output) if lid != 0 else 1
            #     query_pos = pos_scale * raw_query_pos
            # E_output = self.E_layers[lid](E_output, query_pos, E_ref_points,
            #                               src, src_spatial_shapes, src_level_start_index, src_padding_mask)
            #
            # C_output = output[..., self.d_model * 2:]
            # C_ref_points = torch.stack((C, W), dim=-1)
            # if self.use_dab:
            #     raw_query_pos = self.C_ref_point_head(reference_points_input[:, :, 0, :])
            #     pos_scale = self.C_query_scale(C_output) if lid != 0 else 1
            #     query_pos = pos_scale * raw_query_pos
            # C_output = self.C_layers[lid](C_output, query_pos, C_ref_points,
            #                               src, src_spatial_shapes, src_level_start_index, src_padding_mask)
            #
            # output = torch.cat((S_output, E_output, C_output), dim=-1)

            # hack implementation for segment refinement
            if self.segment_embed is not None:
            # if self.S_segment_embed is not None:
                # update the reference point/segment of the next layer according to the output from the current layer
                tmp = self.segment_embed[lid](output)
                if reference_points.shape[-1] == 2:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    # at the 0-th decoder layer
                    # d^(n+1) = delta_d^(n+1)
                    # c^(n+1) = sigmoid( inverse_sigmoid(c^(n)) + delta_c^(n+1))
                    assert reference_points.shape[-1] == 1
                    new_reference_points = tmp
                    new_reference_points[..., :1] = tmp[..., :1] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

                # # S_tmp = self.S_segment_embed[lid](S_output)
                # # E_tmp = self.E_segment_embed[lid](E_output)
                # E_tmp = self.S_segment_embed[lid](S_output)
                # S_tmp = self.E_segment_embed[lid](E_output)
                # new_reference_points = inverse_sigmoid(reference_points)
                # new_reference_points[..., 0] = S_tmp.squeeze(-1) + new_reference_points[..., 0]
                # new_reference_points[..., 1] = E_tmp.squeeze(-1) + new_reference_points[..., 1]
                # new_reference_points = new_reference_points.sigmoid()
                # reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                inter_Q_weights.append(Q_weights)
                inter_C_weights.append(C_weights)
                inter_Q_in.append(Q_in)
                inter_Q_out.append(Q_out)
                inter_C_in.append(C_in)
                inter_C_out.append(C_out)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), \
                   torch.stack(inter_Q_weights), torch.stack(inter_C_weights), \
                   torch.stack(inter_Q_in), torch.stack(inter_Q_out), \
                   torch.stack(inter_C_in), torch.stack(inter_C_out)

        return output, reference_points


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


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "leaky_relu":
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deformable_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.activation,
        return_intermediate_dec=True,
        num_feature_levels=1,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points)


