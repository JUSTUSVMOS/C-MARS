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
import torch.optim
import torch.utils.data as data
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR  # for future use if desired
from torch.utils.data.dataloader import default_collate

import utils.config as config
from utils.dataset import RefDataset
from utils.misc import (
    init_random_seed,
    set_random_seed,
    setup_logger,
    worker_init_fn,
)
from engine.engine import train, validate
from model import build_segmenter

# PERFORMANCE SETTINGS
torch.backends.cudnn.benchmark = True        # Enable cuDNN autotuner for performance
warnings.filterwarnings("ignore")            # Suppress all warnings
cv2.setNumThreads(0)                         # Restrict OpenCV to a single thread


def safe_collate(batch):
    """
    Filter out None entries (in case dataset returns None),
    then use the default collate function.
    """
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if batch else None


def get_config():
    """
    Parse command‐line arguments and load/override the YAML config.
    """
    parser = argparse.ArgumentParser(
        description="Pytorch Referring Expression Segmentation"
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "--opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Optional overrides of config options",
    )
    args = parser.parse_args()

    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


class CustomLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Custom learning‐rate scheduler:
      - Epochs 1–5:    lr = 1e-4
      - Epochs 6–15:   linear decay 1e-4 → 5e-5
      - Epochs 16–25:  linear decay 5e-5 → 1e-5
      - Epochs 26–60:  linear decay 1e-5 → 1e-6
      - After 60:      lr = 1e-6
    """
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1
        if epoch <= 5:
            return [1e-4 for _ in self.base_lrs]
        if epoch <= 15:
            return [1e-4 + (5e-5 - 1e-4) * ((epoch - 5) / 10) for _ in self.base_lrs]
        if epoch <= 25:
            return [5e-5 + (1e-5 - 5e-5) * ((epoch - 15) / 10) for _ in self.base_lrs]
        if epoch <= 60:
            return [1e-5 + (1e-6 - 1e-5) * ((epoch - 25) / 35) for _ in self.base_lrs]
        return [1e-6 for _ in self.base_lrs]


@logger.catch
def main():
    cfg = get_config()
    setup_logger(cfg.output_folder)

    # initialize seeds
    cfg.manual_seed = init_random_seed(cfg.manual_seed)
    set_random_seed(cfg.manual_seed, deterministic=False)

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.device = device

    # build model
    model, param_list = build_segmenter(cfg)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.warning(f"Total trainable parameters: {total_params:,}")

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(
        param_list, lr=cfg.base_lr, weight_decay=cfg.weight_decay
    )
    scheduler = CustomLRScheduler(optimizer)
    scaler = amp.GradScaler()

    # datasets
    train_dataset = RefDataset(
        lmdb_dir=cfg.train_lmdb,
        mask_dir=cfg.mask_root,
        dataset=cfg.dataset,
        split=cfg.train_split,
        mode="train",
        input_size=cfg.input_size,
        word_length=cfg.word_len,
    )
    val_dataset = RefDataset(
        lmdb_dir=cfg.val_lmdb,
        mask_dir=cfg.mask_root,
        dataset=cfg.dataset,
        split=cfg.val_split,
        mode="val",
        input_size=cfg.input_size,
        word_length=cfg.word_len,
    )

    # data loaders
    init_fn = partial(
        worker_init_fn,
        num_workers=cfg.workers,
        rank=0,
        seed=cfg.manual_seed,
    )
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        worker_init_fn=init_fn,
        drop_last=True,
        collate_fn=safe_collate,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size_val,
        shuffle=False,
        num_workers=cfg.workers_val,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        collate_fn=safe_collate,
    )

    best_iou = 0.0

    # optionally resume from checkpoint
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            logger.info(f"=> Loading checkpoint '{cfg.resume}'")
            ckpt = torch.load(
                cfg.resume,
                map_location=lambda storage, loc: storage.cuda()
                if torch.cuda.is_available()
                else storage,
            )
            cfg.start_epoch = ckpt["epoch"]
            best_iou = ckpt["best_iou"]
            model.load_state_dict(ckpt["state_dict"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            logger.info(f"=> Resumed from epoch {ckpt['epoch']}")
        else:
            raise FileNotFoundError(f"No checkpoint at '{cfg.resume}'")

    # training loop
    start_time = time.time()
    for epoch in range(cfg.start_epoch, cfg.epochs):
        current_epoch = epoch + 1

        train(train_loader, model, optimizer, scheduler, scaler, current_epoch, cfg)
        iou, prec_dict = validate(val_loader, model, current_epoch, cfg)

        # save last checkpoint
        last_path = os.path.join(cfg.output_folder, "last_model.pth")
        torch.save(
            {
                "epoch": current_epoch,
                "cur_iou": iou,
                "best_iou": best_iou,
                "prec": prec_dict,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            last_path,
        )
        if iou >= best_iou:
            best_iou = iou
            best_path = os.path.join(cfg.output_folder, "best_model.pth")
            shutil.copyfile(last_path, best_path)

        scheduler.step(current_epoch)

    # report
    elapsed = time.time() - start_time
    logger.info(f"* Best IoU = {best_iou:.4f} *")
    logger.info(f"* Total training time: {datetime.timedelta(seconds=int(elapsed))} *")


if __name__ == "__main__":
    main()
    sys.exit(0)
