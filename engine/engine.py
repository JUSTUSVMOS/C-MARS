'''engine/engine.py

This module provides training, validation, and inference routines for the segmentation model.
'''

import os
import time

import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger

from utils.dataset import tokenize
from utils.misc import AverageMeter, ProgressMeter, trainMetricGPU

# ------------------------ Training ------------------------ #
def train(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    """
    Execute one epoch of training.

    Args:
        train_loader: DataLoader for training data.
        model: PyTorch model to train.
        optimizer: Optimizer for model updates.
        scheduler: Learning rate scheduler.
        scaler: GradScaler for mixed precision.
        epoch: Current epoch number.
        args: Arguments namespace with settings.
    """
    # Initialize meters
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr_meter = AverageMeter('LR', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    iou_meter = AverageMeter('IoU', ':2.2f')
    prec_meter = AverageMeter('Prec@50', ':2.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr_meter, loss_meter, iou_meter, prec_meter],
        prefix=f"Training Epoch [{epoch}/{args.epochs}]"
    )

    model.train()
    time.sleep(2)  # Warm-up delay
    end = time.time()

    loader = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{args.epochs}",
        ncols=120,
        mininterval=args.print_freq
    )

    for idx, (img, txt, tgt) in enumerate(loader, start=1):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Move tensors to device
        img, txt, tgt = (x.to(args.device, non_blocking=True) for x in (img, txt, tgt))
        tgt = tgt.unsqueeze(1)

        # Forward pass with mixed precision
        with amp.autocast():
            pred, tgt_mask, loss = model(img, txt, tgt)

        # Backward and optimization
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()

        # Compute metrics
        iou_val, prec5 = trainMetricGPU(pred, tgt_mask, 0.35, 0.5)
        loss_meter.update(loss.item(), img.size(0))
        iou_meter.update(iou_val.item(), img.size(0))
        prec_meter.update(prec5.item(), img.size(0))
        lr_meter.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        # Display progress
        if idx % args.print_freq == 0:
            progress.display(idx)
            loader.set_postfix(
                Loss=loss_meter.avg,
                IoU=iou_meter.avg,
                Prec50=prec_meter.avg,
                LR=lr_meter.avg,
                Time=batch_time.avg
            )

    # Step scheduler after epoch
    scheduler.step()


# ------------------------ Validation ------------------------ #
@torch.no_grad()
def validate(val_loader, model, epoch, args):
    """
    Execute validation over the dataset.

    Args:
        val_loader: DataLoader for validation data.
        model: PyTorch model to evaluate.
        epoch: Current epoch number.
        args: Arguments namespace with settings.

    Returns:
        mean IoU and precision dictionary.
    """
    def first_element(x):
        if isinstance(x, (list, tuple)):
            return x[0]
        if torch.is_tensor(x) and x.ndim:
            return x[0]
        if isinstance(x, np.ndarray) and x.ndim:
            return x[0]
        return x

    model.eval()
    iou_list = []
    loader = tqdm(val_loader, desc=f"Validation {epoch}/{args.epochs}", ncols=100)

    for imgs, texts, params in loader:
        imgs = imgs.to(args.device, non_blocking=True)
        texts = texts.to(args.device, non_blocking=True)

        # Predict and apply sigmoid
        preds = torch.sigmoid(model(imgs, texts))
        if preds.shape[-2:] != imgs.shape[-2:]:
            preds = F.interpolate(
                preds,
                size=imgs.shape[-2:],
                mode='bicubic',
                align_corners=True
            )

        for pred, mask_path, inv_mat, orig_size in zip(
            preds, params['mask_dir'], params['inverse'], params['ori_size']
        ):
            # Restore original shape and warp back
            h, w = map(int, first_element(orig_size))
            M = np.array(first_element(inv_mat))
            pred_np = pred.squeeze().cpu().numpy()
            pred_warped = cv2.warpAffine(pred_np, M, (w, h), flags=cv2.INTER_CUBIC) > 0.35

            mask = cv2.imread(first_element(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = mask.astype(np.float32) / 255.0

            # Shape tolerance
            if pred_warped.shape != mask.shape:
                if pred_warped.T.shape == mask.shape:
                    pred_warped = pred_warped.T
                elif mask.T.shape == pred_warped.shape:
                    mask = mask.T
                else:
                    mask = cv2.resize(mask, pred_warped.shape[::-1], interpolation=cv2.INTER_NEAREST)

            # Zero-mask rule
            if mask.sum() == 0:
                iou_val = 1.0 if pred_warped.sum() == 0 else 0.0
            else:
                inter = np.logical_and(pred_warped, mask)
                union = np.logical_or(pred_warped, mask)
                iou_val = inter.sum() / (union.sum() + 1e-6)

            iou_list.append(iou_val)

    ious_tensor = torch.tensor(iou_list, device=args.device)
    precision = {f'Pr@{t}0': (ious_tensor > t/10).float().mean().item() for t in range(5, 10)}

    return ious_tensor.mean().item(), precision


# ------------------------ Inference ------------------------ #
@torch.no_grad()
def inference(test_loader, model, args):
    """
    Run inference on test data and compute metrics.

    Args:
        test_loader: DataLoader for test data.
        model: PyTorch model to evaluate.
        args: Arguments namespace with settings.

    Returns:
        mean IoU and empty dict for consistency.
    """
    def first_element(x):
        if isinstance(x, (list, tuple)):
            return x[0]
        if torch.is_tensor(x) and x.ndim:
            return x[0]
        if isinstance(x, np.ndarray) and x.ndim:
            return x[0]
        return x

    model.eval()
    time.sleep(2)  # Warm-up delay
    iou_list = []
    loader = tqdm(test_loader, desc='Inference', ncols=100)

    for img, params in loader:
        img = img.to(args.device, non_blocking=True)

        mask_path = first_element(params['mask_dir'])
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            # Skip missing files
            continue
        mask = mask_raw.astype(np.float32) / 255.0

        inv_mat = np.array(first_element(params['inverse']))
        h, w = map(int, first_element(params['ori_size']))
        sentences = first_element(params['sents'])
        num_sent = len(sentences)

        # Batch pairing (repeat image N times)
        img_batch = img.repeat(num_sent, 1, 1, 1)
        txt_batch = tokenize(sentences, args.word_len, True).to(args.device)

        preds = torch.sigmoid(model(img_batch, txt_batch))
        if preds.shape[-2:] != img.shape[-2:]:
            preds = F.interpolate(
                preds,
                size=img.shape[-2:],
                mode='bicubic',
                align_corners=True
            )

        for pred in preds:
            pred_np = pred.squeeze().cpu().numpy()
            pred_warped = cv2.warpAffine(pred_np, inv_mat, (w, h), flags=cv2.INTER_CUBIC) > 0.35

            # Shape alignment
            if pred_warped.shape != mask.shape:
                if pred_warped.T.shape == mask.shape:
                    pred_warped = pred_warped.T
                elif mask.T.shape == pred_warped.shape:
                    mask = mask.T
                else:
                    mask = cv2.resize(mask, (pred_warped.shape[1], pred_warped.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Compute IoU with zero-mask rule
            if mask.sum() == 0:
                iou_val = 1.0 if pred_warped.sum() == 0 else 0.0
            else:
                inter = np.logical_and(pred_warped, mask)
                union = np.logical_or(pred_warped, mask)
                iou_val = inter.sum() / (union.sum() + 1e-6)
            iou_list.append(iou_val)

    # Report metrics
    ious_tensor = torch.tensor(iou_list, device=args.device)
    logger.info('=> Metric Calculation <=')
    logger.info(f"IoU={ious_tensor.mean()*100:.2f}%")
    for t in range(5, 10):
        logger.info(f"Pr@{t}0: {(ious_tensor > t/10).float().mean()*100:.2f}%")

    return ious_tensor.mean().item(), {}
