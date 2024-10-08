# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os.path as osp
import sys
from typing import Iterable
import tqdm
import logging

import torch
import clip

import util.misc as utils
from datasets.tad_eval import TADEvaluator
import pickle

import os
import random
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import copy

# from util.flop_count import flop_count
from thop import profile, clever_format

def train_one_epoch(model: torch.nn.Module,
                    model_ema,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler,
                    device: torch.device, epoch: int, cfg, max_norm: float = 0):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    cnt = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) if k in ['segments', 'labels']
                    else v for k, v in t.items()} for t in targets]

        outputs = model((samples.tensors, samples.mask))

        # loss_dict = criterion(outputs, targets)
        loss_dict = train_hybrid(outputs, targets, criterion)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k]
                     for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss of each type
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        # weighted_loss of each type
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            logging.info("Loss is {}, stopping training".format(loss_value))
            logging.info(str(loss_dict_reduced))
            sys.exit(1)

        losses.backward()
        scheduler.step()
        if (cnt + 1) % cfg.iter_size == 0:
            # scale gradients when iter size is functioning
            if cfg.iter_size != 1:
                for g in optimizer.param_groups:
                    for p in g['params']:
                        p.grad /= cfg.iter_size

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if model_ema is not None:
            model_ema.update(model)

        cnt += 1

    optimizer.zero_grad()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logging.info(f"Averaged stats:{metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_hybrid(outputs, targets, criterion, k_one2many=6, lambda_one2many=1.0):
    # one-to-one-loss
    loss_dict = criterion(outputs, targets)
    multi_targets = copy.deepcopy(targets)
    # repeat the targets
    for target in multi_targets:
        target["segments"] = target["segments"].repeat(k_one2many, 1)
        target["labels"] = target["labels"].repeat(k_one2many)

    outputs_one2many = dict()
    outputs_one2many["pred_logits"] = outputs["pred_logits_one2many"]
    outputs_one2many["pred_segments"] = outputs["pred_segments_one2many"]
    outputs_one2many["Q_weights"] = outputs["Q_weights_one2many"]
    outputs_one2many["K_weights"] = outputs["K_weights_one2many"]
    outputs_one2many["C_weights"] = outputs["C_weights_one2many"]
    outputs_one2many["aux_outputs"] = outputs["aux_outputs_one2many"]
    if "enc_outputs" in outputs:
        outputs_one2many["enc_outputs"] = outputs["enc_outputs_one2many"]
    loss_dict_one2many = criterion(outputs_one2many, multi_targets)
    for key, value in loss_dict_one2many.items():
        if "enc" in key:
            continue
        if key + "_one2many" in loss_dict.keys():
            loss_dict[key + "_one2many"] += value * lambda_one2many
        else:
            loss_dict[key + "_one2many"] = value * lambda_one2many
    return loss_dict

def to_device(t, device):
    if isinstance(t, (list, tuple)):
        return t
    else:
        return t.to(device)


@torch.no_grad()
def test(model, criterion, postprocessor, data_loader, base_ds, device, output_dir, cfg,
         subset='val', epoch=None, test_mode=False, nms_mode="raw"):
    '''
    Run inference and evaluation. Do not compute loss
    test_mode: indicates that we are evaluating specific epoch during testing
    '''
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(
        window_size=1, fmt='{value:.2f}'))

    iou_range = [0.3, 0.4, 0.5, 0.6, 0.7] if cfg.dataset_name == 'thumos14' else [
        num/100 for num in range(50, 100, 5)]
    # iou_range = [num/100 for num in range(50, 100, 5)]
    # logging.info('iou range {}'.format(iou_range))

    # action_evaluator = None
    action_evaluator = TADEvaluator(cfg.dataset_name, subset, base_ds, nms_mode=[nms_mode], iou_range=iou_range, epoch=epoch)

    # raw_res = []
    cnt = 0
    visualize = False
    diversity = False
    # if visualize and (epoch + 1) % 10 == 0:
    # if visualize and (epoch + 1) >= 0:
    if visualize and (epoch + 1) >= 120 and (epoch + 1) % 10 == 0:
        a_i = 0
        attention_dir = os.path.join(output_dir, "attention", "E{:02d}".format(epoch + 1))
        os.makedirs(attention_dir, exist_ok=True)
        # sampled_indices = random.sample(range(len(data_loader)), 3)
        sampled_indices = range(30)
    if diversity:
        K_d_values = list()
        Q_d_values = list()
        C_d_values = list()

    for (samples, targets) in tqdm.tqdm(data_loader, total=len(data_loader)):
        samples = samples.to(device)

        # with torch.no_grad():
        #     queries = ["all actions"]
        #     queries = clip.tokenize(queries).cuda()
        #     queries = (clip_model.encode_text(queries)).float().detach().repeat(len(samples.tensors), 1)

        queries = None

        outputs = model((samples.tensors, samples.mask))
        # outputs = model((samples.tensors, samples.mask), queries=queries)

        # raw_res.append((outputs, targets))
        video_duration = torch.FloatTensor(
            [t["video_duration"] for t in targets]).to(device)
        results = postprocessor(outputs, video_duration, fuse_score=cfg.act_reg)

        res = {target['video_id']: output for target,
               output in zip(targets, results)}
        if action_evaluator is not None:
            action_evaluator.update(res, assign_cls_labels=cfg.binary)

        cnt += 1
        # if visualize and (epoch + 1) % 10 == 0:
        if visualize and (epoch + 1) >= 120 and (epoch + 1) % 10 == 0:
        # if visualize and (epoch + 1) >= 0:
            if cnt - 1 in sampled_indices:
                this_targets = targets[0]["segments"]
                if torch.is_tensor(this_targets):
                    this_targets = this_targets.detach().cpu().numpy()

                # entire_map = outputs["K_weights"][:, 0].detach().cpu().numpy()
                # L, H, W = entire_map.shape
                # for l_i in range(L):
                #     map = entire_map[l_i]
                #     KK_box = np.zeros(dtype=np.float32, shape=(1 + H // 40, W))
                #     for box in this_targets:
                #         s_i = round((box[0] - box[1] / 2) * (W - 1))
                #         e_i = round((box[0] + box[1] / 2) * (W - 1))
                #         KK_box[1:, s_i:e_i + 1] = np.max(map)
                #     map = np.concatenate((map, KK_box), axis=0)
                #     H_labels = ["{}".format(x) for x in range(1, H + 1, 1)] + [""] + ["GT"] * (H // 40)
                #     W_labels = ["{}".format(x) for x in range(1, W + 1, 1)]
                #     # map -= np.min(map)
                #     # map /= np.max(map)
                #     df = pd.DataFrame(map, H_labels, W_labels)
                #     ax = sn.heatmap(df, cbar=True, xticklabels=False, yticklabels=False, square=True)
                #     # ax = sn.heatmap(df, cbar=True, xticklabels=False, yticklabels=False, square=True,
                #     #                 vmin=0.0, vmax=0.25)
                #     plt.savefig(os.path.join(attention_dir, "K_N{:02d}_L{:02d}.png".format(a_i + 1, l_i + 1)))
                #     plt.close()

                # entire_map = outputs["Q_weights"][:, 0].detach().cpu().numpy()
                # L, H, W = entire_map.shape
                # # for l_i in range(L):
                # for l_i in range(1):
                #     map = entire_map[l_i]
                #     H_labels = ["{}".format(x) for x in range(1, H + 1, 1)]
                #     W_labels = ["{}".format(x) for x in range(1, W + 1, 1)]
                #     # map -= np.min(map)
                #     # map /= np.max(map)
                #     df = pd.DataFrame(map, H_labels, W_labels)
                #     ax = sn.heatmap(df, cbar=True, xticklabels=False, yticklabels=False, square=True)
                #     plt.savefig(os.path.join(attention_dir, "Q_N{:02d}_L{:02d}.png".format(a_i + 1, l_i + 1)))
                #     plt.close()

                # entire_map = outputs["C_weights"][:, 0].detach().cpu().numpy()
                # L, H, W = entire_map.shape
                # for l_i in range(L):
                #     map = entire_map[l_i]
                #     QK_box = np.zeros(dtype=np.float32, shape=(1 + H // 40, W))
                #     for box in this_targets:
                #         s_i = round((box[0] - box[1] / 2) * (W - 1))
                #         e_i = round((box[0] + box[1] / 2) * (W - 1))
                #         QK_box[1:, s_i:e_i + 1] = np.max(map)
                #     map = np.concatenate((map, QK_box), axis=0)
                #     H_labels = ["{}".format(x) for x in range(1, H + 1, 1)] + [""] + ["GT"] * (H // 40)
                #     W_labels = ["{}".format(x) for x in range(1, W + 1, 1)]
                #     # map -= np.min(map)
                #     # map /= np.max(map)
                #     df = pd.DataFrame(map, H_labels, W_labels)
                #     ax = sn.heatmap(df, cbar=True, xticklabels=False, yticklabels=False, square=True)
                #     # ax = sn.heatmap(df, cbar=True, xticklabels=False, yticklabels=False, square=True,
                #     #                 vmin=0.0, vmax=0.25)
                #     plt.savefig(os.path.join(attention_dir, "C_N{:02d}_L{:02d}.png".format(a_i + 1, l_i + 1)))
                #     plt.close()

                entire_map = outputs["K_weights"][:, 0].detach().cpu().numpy()
                L, H, W = entire_map.shape
                for l_i in range(L):
                    map = entire_map[l_i]
                    KK_box = np.zeros(dtype=np.float32, shape=(1 + max(H // 40, 1), W))
                    for box in this_targets:
                        s_i = round((box[0] - box[1] / 2) * (W - 1))
                        e_i = round((box[0] + box[1] / 2) * (W - 1))
                        KK_box[1:, s_i:e_i + 1] = np.max(map)
                    map = np.concatenate((map, KK_box), axis=0)
                    H_labels = ["{}".format(x) for x in range(1, H + 1, 1)] + [""] + ["GT"] * max(H // 40, 1)
                    W_labels = ["{}".format(x) for x in range(1, W + 1, 1)]
                    # map -= np.min(map)
                    # map /= np.max(map)
                    df = pd.DataFrame(map, H_labels, W_labels)
                    ax = sn.heatmap(df, cbar=True, xticklabels=False, yticklabels=False, square=True)
                    # ax = sn.heatmap(df, cbar=True, xticklabels=False, yticklabels=False, square=True,
                    #                 vmin=0.0, vmax=0.25)
                    plt.savefig(os.path.join(attention_dir, "K_N{:02d}_L{:02d}.png".format(iter_idx + 1, l_i + 1)))
                    plt.close()

                entire_map = outputs["Q_weights"][:, 0].detach().cpu().numpy()
                L, H, W = entire_map.shape
                # for l_i in range(L):
                for l_i in range(L):
                    map = entire_map[l_i]
                    H_labels = ["{}".format(x) for x in range(1, H + 1, 1)]
                    W_labels = ["{}".format(x) for x in range(1, W + 1, 1)]
                    # map -= np.min(map)
                    # map /= np.max(map)
                    df = pd.DataFrame(map, H_labels, W_labels)
                    ax = sn.heatmap(df, cbar=True, xticklabels=False, yticklabels=False, square=True)
                    plt.savefig(os.path.join(attention_dir, "Q_N{:02d}_L{:02d}.png".format(iter_idx + 1, l_i + 1)))
                    plt.close()

                entire_map = outputs["C_weights"][:, 0].detach().cpu().numpy()
                L, H, W = entire_map.shape
                for l_i in range(L):
                    map = entire_map[l_i]
                    QK_box = np.zeros(dtype=np.float32, shape=(1 + max(H // 40, 1), W))
                    for box in this_targets:
                        s_i = round((box[0] - box[1] / 2) * (W - 1))
                        e_i = round((box[0] + box[1] / 2) * (W - 1))
                        QK_box[1:, s_i:e_i + 1] = np.max(map)
                    map = np.concatenate((map, QK_box), axis=0)
                    H_labels = ["{}".format(x) for x in range(1, H + 1, 1)] + [""] + ["GT"] * max(H // 40, 1)
                    W_labels = ["{}".format(x) for x in range(1, W + 1, 1)]
                    # map -= np.min(map)
                    # map /= np.max(map)
                    df = pd.DataFrame(map, H_labels, W_labels)
                    ax = sn.heatmap(df, cbar=True, xticklabels=False, yticklabels=False, square=True)
                    # ax = sn.heatmap(df, cbar=True, xticklabels=False, yticklabels=False, square=True,
                    #                 vmin=0.0, vmax=0.25)
                    plt.savefig(os.path.join(attention_dir, "C_N{:02d}_L{:02d}.png".format(iter_idx + 1, l_i + 1)))
                    plt.close()

                pred_idx = 0
                nk = outputs["K_weights"][:, 0].size(1)
                enc_preds = outputs["enc_outputs"]["pred_segments"][0].detach().cpu()
                # enc_preds = torch.stack(detr_predictions["enc_outputs"]["pred_segments"][0].split(nk, dim=0)).detach().cpu()
                # L, H, _ = enc_preds.shape
                # for l_i in range(L):
                for l_i in range(4):
                    this_nk = nk // (2 ** l_i)
                    this_enc_preds = enc_preds[pred_idx:pred_idx + this_nk]
                    map = np.zeros(dtype=np.float32, shape=(this_nk, this_nk))
                    # for m_i, box in enumerate(enc_preds[l_i].numpy()):
                    for m_i, box in enumerate(this_enc_preds.numpy()):
                        s_i = round((box[0] - box[1] / 2) * (this_nk - 1))
                        e_i = round((box[0] + box[1] / 2) * (this_nk - 1))
                        map[m_i, s_i:e_i + 1] = 1.0
                    KK_box = np.zeros(dtype=np.float32, shape=(1 + max(this_nk // 40, 1), this_nk))
                    for box in this_targets:
                        s_i = round((box[0] - box[1] / 2) * (this_nk - 1))
                        e_i = round((box[0] + box[1] / 2) * (this_nk - 1))
                        KK_box[1:, s_i:e_i + 1] = np.max(map)
                    map = np.concatenate((map, KK_box), axis=0)
                    H_labels = ["{}".format(x) for x in range(1, this_nk + 1, 1)] + [""] + ["GT"] * max(this_nk // 40,
                                                                                                        1)
                    W_labels = ["{}".format(x) for x in range(1, this_nk + 1, 1)]
                    df = pd.DataFrame(map, H_labels, W_labels)
                    ax = sn.heatmap(df, cbar=True, xticklabels=False, yticklabels=False, square=True)
                    plt.savefig(
                        os.path.join(attention_dir, "EncPred_N{:02d}_L{:02d}.png".format(iter_idx + 1, l_i + 1)))
                    plt.close()

                    # IoUs = segment_ops.batched_segment_iou(segment_ops.segment_cw_to_t1t2(enc_preds[l_i]),
                    #                                        segment_ops.segment_cw_to_t1t2(enc_preds[l_i]))
                    IoUs = segment_ops.batched_segment_iou(segment_ops.segment_cw_to_t1t2(this_enc_preds),
                                                           segment_ops.segment_cw_to_t1t2(this_enc_preds))
                    map = IoUs.softmax(dim=-1).numpy()
                    KK_box = np.zeros(dtype=np.float32, shape=(1 + max(this_nk // 40, 1), this_nk))
                    for box in this_targets:
                        s_i = round((box[0] - box[1] / 2) * (this_nk - 1))
                        e_i = round((box[0] + box[1] / 2) * (this_nk - 1))
                        KK_box[1:, s_i:e_i + 1] = np.max(map)
                    map = np.concatenate((map, KK_box), axis=0)
                    H_labels = ["{}".format(x) for x in range(1, this_nk + 1, 1)] + [""] + ["GT"] * max(this_nk // 40,
                                                                                                        1)
                    W_labels = ["{}".format(x) for x in range(1, this_nk + 1, 1)]
                    df = pd.DataFrame(map, H_labels, W_labels)
                    ax = sn.heatmap(df, cbar=True, xticklabels=False, yticklabels=False, square=True)
                    plt.savefig(os.path.join(attention_dir, "EncIoU_N{:02d}_L{:02d}.png".format(iter_idx + 1, l_i + 1)))
                    plt.close()
                    pred_idx += this_nk

                nk = outputs["K_weights"][:, 0].size(1)
                dec_preds = torch.cat(
                    (torch.stack([a_o['pred_segments'] for a_o in outputs['aux_outputs']], dim=0),
                     outputs['pred_segments'].unsqueeze(0)), dim=0)[:, 0].detach().cpu()
                L, H, _ = dec_preds.shape
                for l_i in range(L):
                    map = np.zeros(dtype=np.float32, shape=(H, nk))
                    for m_i, box in enumerate(dec_preds[l_i].numpy()):
                        s_i = round((box[0] - box[1] / 2) * (nk - 1))
                        e_i = round((box[0] + box[1] / 2) * (nk - 1))
                        map[m_i, s_i:e_i + 1] = 1.0
                    KK_box = np.zeros(dtype=np.float32, shape=(1 + max(H // 40, 1), nk))
                    for box in this_targets:
                        s_i = round((box[0] - box[1] / 2) * (nk - 1))
                        e_i = round((box[0] + box[1] / 2) * (nk - 1))
                        KK_box[1:, s_i:e_i + 1] = np.max(map)
                    map = np.concatenate((map, KK_box), axis=0)
                    H_labels = ["{}".format(x) for x in range(1, H + 1, 1)] + [""] + ["GT"] * max(H // 40, 1)
                    W_labels = ["{}".format(x) for x in range(1, nk + 1, 1)]
                    df = pd.DataFrame(map, H_labels, W_labels)
                    ax = sn.heatmap(df, cbar=True, xticklabels=False, yticklabels=False, square=True)
                    plt.savefig(
                        os.path.join(attention_dir, "DecPred_N{:02d}_L{:02d}.png".format(iter_idx + 1, l_i + 1)))
                    plt.close()

                    IoUs = segment_ops.batched_segment_iou(segment_ops.segment_cw_to_t1t2(dec_preds[l_i]),
                                                           segment_ops.segment_cw_to_t1t2(dec_preds[l_i]))
                    map = IoUs.softmax(dim=-1).numpy()
                    H_labels = ["{}".format(x) for x in range(1, H + 1, 1)]
                    W_labels = ["{}".format(x) for x in range(1, H + 1, 1)]
                    df = pd.DataFrame(map, H_labels, W_labels)
                    ax = sn.heatmap(df, cbar=True, xticklabels=False, yticklabels=False, square=True)
                    plt.savefig(os.path.join(attention_dir, "DecIoU_N{:02d}_L{:02d}.png".format(iter_idx + 1, l_i + 1)))
                    plt.close()

                a_i += 1

        if diversity and cnt <= 10:
            # K_in = outputs["K_in"].detach().cpu().transpose(1, 0).numpy()
            # K_out = outputs["K_out"].detach().cpu().transpose(1, 0).numpy()
            # Q_in = outputs["Q_in"].detach().cpu().transpose(1, 0).numpy()
            # Q_out = outputs["Q_out"].detach().transpose(1, 0).cpu().numpy()
            # C_in = outputs["C_in"].detach().transpose(1, 0).cpu().numpy()
            # C_out = outputs["C_out"].detach().transpose(1, 0).cpu().numpy()

            K_in = outputs["K_weights"].detach().cpu().transpose(1, 0).numpy()
            Q_in = outputs["Q_weights"].detach().cpu().transpose(1, 0).numpy()
            C_in = outputs["C_weights"].detach().cpu().transpose(1, 0).numpy()

            K_out = K_in
            Q_out = Q_in
            C_out = C_in

            tgt_ins = [K_in, Q_in, C_in]
            tgt_outs = [K_out, Q_out, C_out]
            tgt_lists = [K_d_values, Q_d_values, C_d_values]
            for tgt_in, tgt_out, tgt_list in zip(tgt_ins, tgt_outs, tgt_lists):
                tgt_in = np.transpose(tgt_in, (0, 1, 3, 2))
                # N, L, W, D
                # W = tgt_in.shape[2]
                # N, L, W, W, D
                d_in = np.expand_dims(tgt_in, axis=2) - np.expand_dims(tgt_in, axis=3)
                d_out = np.expand_dims(tgt_out, axis=2) - np.expand_dims(tgt_out, axis=3)
                # N, L, W
                d_in = np.sqrt(np.linalg.norm(d_in, ord=1, axis=(3, 4)) * np.linalg.norm(d_in, ord=np.inf, axis=(3, 4)))
                d_out = np.sqrt(np.linalg.norm(d_out, ord=1, axis=(3, 4)) * np.linalg.norm(d_out, ord=np.inf, axis=(3, 4)))
                # N, L
                d_in = np.min(d_in, axis=2)
                d_out = np.min(d_out, axis=2)

                l_out = np.sqrt(np.linalg.norm(tgt_out, ord=1, axis=(2, 3)) * np.linalg.norm(tgt_out, ord=np.inf, axis=(2, 3)))

                # tgt_list.append(d_out / d_in)
                tgt_list.append(d_out / l_out)
                # tgt_list.append(d_out)

        # # flops = flop_count(model, (samples.tensors, ))
        # # for n_i in range(len(samples.tensors)):
        # macs, params = profile(model, inputs=(samples.tensors[-1][None][..., :100], ))
        # # macs, params = clever_format([macs, params], "%.3f")
        # print(macs, params)
        # exit()

    if diversity:
        K_d_values = np.concatenate(K_d_values, axis=0)
        Q_d_values = np.concatenate(Q_d_values, axis=0)
        C_d_values = np.concatenate(C_d_values, axis=0)

        K_d_mean = np.mean(K_d_values, axis=0)
        K_d_vars = np.std(K_d_values, axis=0)

        Q_d_mean = np.mean(Q_d_values, axis=0)
        Q_d_vars = np.std(Q_d_values, axis=0)

        C_d_mean = np.mean(C_d_values, axis=0)
        C_d_vars = np.std(C_d_values, axis=0)

        print("=" * 50)
        print("Epoch {:02d}: Diversity".format(epoch + 1))
        print("KK: {} / {}".format(K_d_mean, K_d_vars))
        print("QQ: {} / {}".format(Q_d_mean, Q_d_vars))
        print("QK: {} / {}".format(C_d_mean, C_d_vars))
        print("=" * 50)

    # accumulate predictions from all videos
    if action_evaluator is not None:
        action_evaluator.synchronize_between_processes()
        action_evaluator.accumulate(cfg.test_slice_overlap)
        # dump detections
        if test_mode:
            save_path = osp.join('outputs', 'detection_{}.json')
            action_evaluator.dump_detection(save_path)
        action_evaluator.summarize()

    stats = {}

    if action_evaluator is not None:
        for k, v in action_evaluator.stats.items():
            for vk, vv in v.items():
                stats[vk + '_' + k] = vv

        mAP_values = ' '.join([f'{k}: {100*v:.2f}'.format(k, v)
                              for k, v in stats.items() if k.startswith('mAP')])
        logging.info(mAP_values)

        stats['stats_summary'] = action_evaluator.stats_summary

    # with open('raw_outputs.pkl', 'wb') as f:
    #     pickle.dump(raw_res, f)

    return stats
