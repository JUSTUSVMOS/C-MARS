import argparse
import datetime
import os
import shutil
import sys
import time
import warnings
from functools import partial

import cv2
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim
import torch.utils.data as data
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from torch.utils.data.dataloader import default_collate

import utils.config as config
# import wandb
from utils.dataset import RefDataset
from engine.engine import train, validate
from model import build_segmenter
from utils.misc import (init_random_seed, set_random_seed, setup_logger,
                        worker_init_fn)


warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

def safe_collate(batch):
    batch = [b for b in batch if b is not None]   # 篩掉 dataset 回傳 None
    return default_collate(batch) if batch else None

def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

class CustomLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1

        if epoch <= 5:
            # Epoch 1-5: Learning rate is constant at 0.0001
            return [0.0001 for base_lr in self.base_lrs]
        elif 6 <= epoch <= 15:
            # Epoch 6-15: Linear decrease from 0.0001 to 0.00005
            return [0.0001 + (0.00005 - 0.0001) * ((epoch - 5) / 10) for base_lr in self.base_lrs]
        elif 16 <= epoch <= 25:
            # Epoch 16-25: Linear decrease from 0.00005 to 0.00001
            return [0.00005 + (0.00001 - 0.00005) * ((epoch - 15) / 10) for base_lr in self.base_lrs]
        elif 26 <= epoch <= 60:
            # Epoch 26-60: Linear decrease from 0.00001 to 0.000001
            return [0.00001 + (0.000001 - 0.00001) * ((epoch - 25) / 35) for base_lr in self.base_lrs]
        else:
            # Beyond epoch 60: Learning rate is constant at 0.000001
            return [0.000001 for base_lr in self.base_lrs]


@logger.catch
def main():
    
    args = get_parser()
    setup_logger(args.output_folder)
    args.manual_seed = init_random_seed(args.manual_seed)
    set_random_seed(args.manual_seed, deterministic=False)

    # 设置设备为 GPU 或 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    
    # 构建模型
    model, param_list = build_segmenter(args)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.warning(f"Total trainable parameters: {total_params:,}")
    # build optimizer & lr scheduler
    optimizer = torch.optim.AdamW(param_list,
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay)
    # scheduler = MultiStepLR(optimizer,
    #                         milestones=args.milestones,
    #                         gamma=args.lr_decay)
    
    scheduler = CustomLRScheduler(optimizer)
    scaler = amp.GradScaler()

    # build dataset
    train_data = RefDataset(lmdb_dir=args.train_lmdb,
                            mask_dir=args.mask_root,
                            dataset=args.dataset,
                            split=args.train_split,
                            mode='train',
                            input_size=args.input_size,
                            word_length=args.word_len)
    val_data = RefDataset(lmdb_dir=args.val_lmdb,
                          mask_dir=args.mask_root,
                          dataset=args.dataset,
                          split=args.val_split,
                          mode='val',
                          input_size=args.input_size,
                          word_length=args.word_len)

    # build dataloader
    init_fn = partial(worker_init_fn,
                      num_workers=args.workers,
                      rank=0,  # 单GPU训练时 rank 设置为 0
                      seed=args.manual_seed)
    train_loader = data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=init_fn,
        drop_last=True,
        collate_fn=safe_collate)          # <── 這行

    val_loader = data.DataLoader(
        val_data,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers_val,
        pin_memory=True,
        drop_last=False,
    collate_fn=safe_collate)          # <── 這行


    best_IoU = 0.0
    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda() if torch.cuda.is_available() else storage)
            args.start_epoch = checkpoint['epoch']
            best_IoU = checkpoint["best_iou"]
            model.load_state_dict(checkpoint['state_dict'])
            print(model)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            raise ValueError(
                "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
                .format(args.resume))

    # start training
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        # train
        train(train_loader, model, optimizer, scheduler, scaler, epoch_log,
              args)

        # evaluation
        iou, prec_dict = validate(val_loader, model, epoch_log, args)

        # save model
        lastname = os.path.join(args.output_folder, "last_model.pth")
        torch.save(
            {
                'epoch': epoch_log,
                'cur_iou': iou,
                'best_iou': best_IoU,
                'prec': prec_dict,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, lastname)
        if iou >= best_IoU:
            best_IoU = iou
            bestname = os.path.join(args.output_folder, "best_model.pth")
            shutil.copyfile(lastname, bestname)

        # update lr
        scheduler.step(epoch_log)
        torch.cuda.empty_cache()

    time.sleep(2)

    logger.info("* Best IoU={} * ".format(best_IoU))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('* Training time {} *'.format(total_time_str))


if __name__ == '__main__':
    main()
    sys.exit(0)
