# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021 - 2012. Xiaolong Liu
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
# and DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
'''Entry for training and testing'''

import datetime
import json
import random
import time
from pathlib import Path
import re
import os
import logging
import sys
import os.path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from util.lr_schedulers import LinearWarmupCosineAnnealingLR

from opts import get_args_parser, cfg, update_cfg_with_args, update_cfg_from_file
import util.misc as utils
from datasets import build_dataset
from engine import train_one_epoch, test
from models import build_model
if cfg.tensorboard:
    from tensorboardX import SummaryWriter

class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None, copy_model=True):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        if copy_model:
            self.module = deepcopy(model)
        else:
            self.module = model
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

def main(args):
    from util.logger import setup_logger

    if args.cfg is not None:
        update_cfg_from_file(cfg, args.cfg)

    update_cfg_with_args(cfg, args.opt)

    if cfg.output_dir:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # The actionness regression module requires CUDA support
    # If your machine does not have CUDA enabled, this module will be disabled.
    if cfg.disable_cuda:
        cfg.act_reg = False

    utils.init_distributed_mode(args)

    if not args.eval:
        mode = 'train'
    else:
        mode = 'test'

    # Logs will be saved in log_path
    log_path = os.path.join(cfg.output_dir, mode + '.log')
    setup_logger(log_path)

    logging.info("git:\n  {}\n".format(utils.get_sha()))

    logging.info(' '.join(sys.argv))

    with open(osp.join(cfg.output_dir, mode + '_cmd.txt'), 'w') as f:
        f.write(' '.join(sys.argv) + '\n')
    logging.info(str(args))
    logging.info(str(cfg))

    device = torch.device(args.device)

    best_metric = -1
    best_metric_txt = ''

    # while True:
        # best_metric = -1
        # best_metric_txt = ''

    # fix the seed
    seed = 42
    # seed = random.randint(0, 10000) + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # best_metric = -1
    # best_metric_txt = ''

    if cfg.input_type == 'image':
        # We plan to support image input in the future
        raise NotImplementedError

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, criterion, postprocessors = build_model(cfg)
    model_, _, _ = build_model(cfg)

    model_.load_state_dict(model.state_dict())

    # checkpoint = torch.load("/mnt/hdd0/VAD/ckpt/kinetics_i3d_v1_scale/pretrain/epoch_015.pth.tar")
    # # # checkpoint = torch.load("/mnt/ssd0/VAD/ckpt/kinetics_i3d_v1_base/pretrain/epoch_015.pth.tar")
    # # # checkpoint = torch.load("/mnt/ssd0/VAD/ckpt/kinetics_i3d_LTP_Deform_S8_scale_E15/pretrain/epoch_014.pth.tar")
    # # # checkpoint = torch.load("/mnt/ssd0/VAD/ckpt/kinetics_i3d_v1_S8_scale_deform/pretrain/epoch_019.pth.tar")
    # # # checkpoint = torch.load("/mnt/ssd0/VAD/ckpt/kinetics_slowfast_deformable_IoU/pretrain/epoch_024.pth.tar")
    # # filtered_ckpt = dict()
    # # for k, v in checkpoint['state_dict'].items():
    # #     # if "class_embed" not in k:
    # #     if "class_embed" not in k and "clip_embed" not in k:
    # #     # if "input" not in k and "class_embed" not in k and "clip_embed" not in k:
    # #     # if "class_embed" not in k and "clip_embed" not in k and "input_proj" not in k:
    # #     # if "class_embed" not in k and "query_embed" not in k:
    # #     # if "class_embed" not in k and "refpoint_embed" not in k and "query_embed" not in k:
    # #     # if "class_embed" not in k and "clip_embed" not in k \
    # #     #         and "query_embed" not in k and "refpoint_embed" not in k:
    # #         filtered_ckpt[k] = v
    # # model.load_state_dict(filtered_ckpt, strict=False)
    # model.load_state_dict(checkpoint['state_dict_ema'], strict=False)
    # del checkpoint

    model.to(device)
    model_.to(device)
    model_ema = ModelEma(model_, copy_model=False)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    elif args.multi_gpu:
        model = torch.nn.DataParallel(model)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters())
    logging.info('number of params: {}'.format(n_parameters))

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        # non-backbone, non-offset
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, cfg.lr_backbone_names) and not match_name_keywords(n, cfg.lr_linear_proj_names) and p.requires_grad],
            "lr": cfg.lr,
            "initial_lr": cfg.lr
        },
        # backbone
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.lr_backbone_names) and p.requires_grad],
            "lr": cfg.lr_backbone,
            "initial_lr": cfg.lr_backbone
        },
        # offset
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.lr_linear_proj_names) and p.requires_grad],
            "lr": cfg.lr * cfg.lr_linear_proj_mult,
            "initial_lr": cfg.lr * cfg.lr_linear_proj_mult
        }
    ]

    optimizer = torch.optim.__dict__[cfg.optimizer](param_dicts, lr=cfg.lr,
                                                     weight_decay=cfg.weight_decay)

    output_dir = Path(cfg.output_dir)

    if args.resume == 'latest':
        args.resume = osp.join(cfg.output_dir, 'checkpoint.pth')
    elif args.resume == 'best':
        args.resume = osp.join(cfg.output_dir, 'model_best.pth')

    # if 'model_best.pth' in os.listdir(cfg.output_dir) and not args.resume and not args.eval:
    #     # for many times, my trained models were accidentally overwrittern by new models😂. So I add this to avoid that
    #     logging.error(
    #         'Danger! You are overwriting an existing output dir {}, probably because you forget to change the output_dir option'.format(cfg.output_dir))
    #     confirm = input('confirm: y/n')
    #     if confirm != 'y':
    #         return

    last_epoch = -1

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        last_epoch = checkpoint['epoch']

    dataset_val = build_dataset(subset=cfg.test_set, args=cfg, mode='val')
    if not args.eval:
        dataset_train = build_dataset(subset='train', args=cfg, mode='train')

    if args.distributed:
        if not args.eval:
            sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)

    else:
        if not args.eval:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if not args.eval:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, cfg.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train,
                                       batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)

    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, cfg.lr_step, last_epoch=last_epoch)

    num_iters_per_epoch = len(data_loader_train)
    max_epochs = cfg.epochs + 5 # warm-up epochs
    max_steps = max_epochs * num_iters_per_epoch
    # get warmup params
    warmup_epochs = 5
    warmup_steps = warmup_epochs * num_iters_per_epoch
    # Cosine
    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps,
        max_steps,
    )
    cfg.epochs += warmup_epochs

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_steps, last_epoch=last_epoch)

    data_loader_val = DataLoader(dataset_val, cfg.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    base_ds = dataset_val.video_dict

    if not args.eval and cfg.tensorboard and utils.is_main_process():
        smry_writer = SummaryWriter(output_dir)
    else:
        smry_writer = None

    if args.eval and not args.resume:
        args.resume = osp.join(output_dir, 'model_best.pth')

    # start training from this epoch. You do not to set this option.
    start_epoch = 0
    if args.resume:
        print('loading checkpint {}'.format(args.resume))
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1

        if 'best_metric' in checkpoint:
            best_metric = checkpoint['best_metric']

    if args.eval:
        test_stats = test(model, criterion, postprocessors,
                          data_loader_val, base_ds, device, cfg.output_dir, cfg, subset=cfg.test_set, epoch=checkpoint['epoch'], test_mode=True)

        return

    # test_stats = test(
    #     model_ema.module,
    #     clip_model,
    #     criterion, postprocessors, data_loader_val, base_ds, device, cfg.output_dir, cfg, epoch=epoch
    # )
    # prime_metric = 'mAP_raw'
    # if test_stats[prime_metric] > best_metric:
    #     best_metric = test_stats[prime_metric]
    #     best_metric_txt = test_stats['stats_summary']
    #     logging.info(
    #         'new_best_metric {:.4f}@epoch{}|seed{}'.format(best_metric, epoch, seed))
    # exit()

    logging.info("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        for group in optimizer.param_groups:
            logging.info('lr={}'.format(group['lr']))
        train_stats = train_one_epoch(
            model, model_ema, criterion, data_loader_train, optimizer, lr_scheduler,
            device, epoch, cfg, cfg.clip_max_norm)

        # lr_scheduler.step()

        if cfg.output_dir:
            # save checkpoint every `cfg.ckpt_interval` epochs, also when reducing the learning rate
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) in cfg.lr_step or (epoch + 1) % cfg.ckpt_interval == 0:
                checkpoint_paths.append(
                    output_dir / f'checkpoint{epoch:04}.pth')
            ckpt = {
                'model': model_without_ddp.state_dict(),
                'epoch': epoch,
                'args': args,
                'cfg': cfg,
                'best_metric': best_metric,
            }
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(ckpt, checkpoint_path)

        if (epoch + 1) % cfg.test_interval == 0:
            test_stats = test(
                # model,
                model_ema.module,
                criterion, postprocessors, data_loader_val, base_ds, device, cfg.output_dir, cfg,
                epoch=epoch, nms_mode=cfg.nms_mode,
            )
            prime_metric = "mAP_{}".format(cfg.nms_mode)
            if test_stats[prime_metric] > best_metric:
                best_metric = test_stats[prime_metric]
                best_metric_txt = test_stats['stats_summary']
                logging.info(
                    'new_best_metric {:.4f}@epoch{}|seed{}'.format(best_metric, epoch, seed))
                if cfg.output_dir:
                    ckpt['best_metric'] = best_metric
                    best_ckpt_path = output_dir / 'model_best.pth'
                    utils.save_on_master(ckpt, best_ckpt_path)

        else:
            test_stats = {}

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if cfg.output_dir and utils.is_main_process():
            for k, v in log_stats.items():
                if isinstance(v, np.ndarray):
                    log_stats[k] = v.tolist()
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if smry_writer:
                for k, v in log_stats.items():
                    if re.findall('loss_\S+unscaled', k) or k.endswith('loss') or 'lr' in k or 'AP50' in k or 'AP75' in k or 'AP95' in k or 'mAP' in k or 'AR' in k:
                        smry_writer.add_scalar(k, v, epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    if utils.is_main_process():
        logging.info('Training time {}'.format(total_time_str))
        logging.info(str(
            ['{}:{}'.format(k, v) for k, v in test_stats.items() if 'AP' in k or 'AR' in k]))
        if smry_writer is not None:
            smry_writer.close()
    logging.info('best det result\n{}'.format(best_metric_txt))
    logging.info(log_path)

    # break

    # if best_metric > 0.573:
    #     break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        'TadTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    s_ = time.time()
    main(args)
    logging.info('main takes {:.3f} seconds'.format(time.time() - s_))
