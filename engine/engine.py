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
from utils.misc import AverageMeter, trainMetricGPU

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
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr_meter   = AverageMeter('LR', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    iou_meter  = AverageMeter('IoU', ':2.2f')
    prec_meter = AverageMeter('Prec@50', ':2.2f')

    model.train()
    end = time.time()
    total_steps = len(train_loader)
    from utils.misc import ProgressMeter
    progress = ProgressMeter(
        total_steps,
        [batch_time, data_time, lr_meter, loss_meter, iou_meter, prec_meter],
        prefix=f"Training Epoch [{epoch}/{args.epochs}]"
    )

    for idx, (img, txt, tgt) in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)
        img, txt, tgt = (x.to(args.device, non_blocking=True) for x in (img, txt, tgt))
        tgt = tgt.unsqueeze(1)

        with amp.autocast():
            pred, tgt_mask, loss = model(img, txt, tgt)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()

        iou_val, prec5 = trainMetricGPU(pred, tgt_mask, 0.35, 0.5)
        loss_meter.update(loss.item(), img.size(0))
        iou_meter.update(iou_val.item(), img.size(0))
        prec_meter.update(prec5.item(), img.size(0))
        lr_meter.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            progress.display(idx)
            print(f"Step [{idx}/{total_steps}] "
                  f"Loss={loss_meter.avg:.4f} "
                  f"IoU={iou_meter.avg:.2f} "
                  f"Prec50={prec_meter.avg:.2f} "
                  f"LR={lr_meter.avg:.6f}")

    scheduler.step()

# ------------------------ Validation ------------------------ #
@torch.no_grad()
def validate(val_loader, model, epoch, args):
    """
    Execute validation over the dataset.
    Returns overall IoU, mean IoU (excl zero-case) and zero-case accuracy for ref-zom, or mean IoU otherwise, plus precision dict.
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
    acc_list = []
    cum_I, cum_U = 0.0, 0.0
    loader = tqdm(val_loader, desc=f"Validation {epoch}/{args.epochs}", ncols=100)

    for imgs, texts, params in loader:
        imgs = imgs.to(args.device, non_blocking=True)
        texts = texts.to(args.device, non_blocking=True)
        preds = torch.sigmoid(model(imgs, texts))
        if preds.shape[-2:] != imgs.shape[-2:]:
            preds = F.interpolate(preds, size=imgs.shape[-2:], mode='bicubic', align_corners=True)

        for pred, mask_path, inv_mat, orig_size in zip(preds, params['mask_dir'], params['inverse'], params['ori_size']):
            h, w = map(int, first_element(orig_size))
            M = np.array(first_element(inv_mat))
            pred_np = pred.squeeze().cpu().numpy()
            pred_warped = (cv2.warpAffine(pred_np, M, (w, h), flags=cv2.INTER_CUBIC) > 0.35)

            mask = cv2.imread(first_element(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = mask.astype(np.float32) / 255.0
            if pred_warped.shape != mask.shape:
                mask = cv2.resize(mask, pred_warped.shape[::-1], interpolation=cv2.INTER_NEAREST)

            # zero-case handling
            if mask.sum() == 0:
                acc_list.append(1.0 if pred_warped.sum() == 0 else 0.0)
                continue

            inter = np.logical_and(pred_warped, mask).sum()
            union = np.logical_or(pred_warped, mask).sum()
            iou_list.append(inter / (union + 1e-6))
            cum_I += inter
            cum_U += union

    precision = {}
    if iou_list:
        ious_tensor = torch.tensor(iou_list, device=args.device)
        mean_iou = ious_tensor.mean().item()
        precision = {f'Pr@{t}0': (ious_tensor > t/10).float().mean().item() for t in range(5, 10)}
    else:
        mean_iou = 0.0

    overall_iou = cum_I / (cum_U + 1e-6) if cum_U > 0 else 0.0
    mean_acc = float(np.mean(acc_list)) if acc_list else 0.0

    if getattr(args, 'dataset', '') == 'ref-zom':
        logger.info(f"Overall IoU   = {overall_iou*100:.2f}%")
        logger.info(f"Mean IoU      = {mean_iou*100:.2f}%")
        logger.info(f"Zero-case acc = {mean_acc*100:.2f}%")
        return overall_iou, precision
    else:
        logger.info(f"Mean IoU      = {mean_iou*100:.2f}%")
        return mean_iou, precision

# ------------------------ Inference ------------------------ #
@torch.no_grad()
def inference(test_loader, model, args):
    """
    Run inference on test data and compute metrics.
    Returns overall IoU, mean IoU (excl zero-case) and zero-case accuracy for ref-zom, or mean IoU otherwise.
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
    acc_list = []
    cum_I, cum_U = 0.0, 0.0
    loader = tqdm(test_loader, desc='Inference', ncols=100)

    for imgs, texts, params in loader:
        imgs = imgs.to(args.device, non_blocking=True)
        texts = texts.to(args.device, non_blocking=True)
        preds = torch.sigmoid(model(imgs, texts))
        if preds.shape[-2:] != imgs.shape[-2:]:
            preds = F.interpolate(preds, size=imgs.shape[-2:], mode='bicubic', align_corners=True)

        for pred, mask_path, inv_mat, orig_size in zip(preds, params['mask_dir'], params['inverse'], params['ori_size']):
            h, w = map(int, first_element(orig_size))
            M = np.array(first_element(inv_mat))
            pred_np = pred.squeeze().cpu().numpy()
            pred_warped = (cv2.warpAffine(pred_np, M, (w, h), flags=cv2.INTER_CUBIC) > 0.35)

            mask = cv2.imread(first_element(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = mask.astype(np.float32) / 255.0
            if pred_warped.shape != mask.shape:
                mask = cv2.resize(mask, pred_warped.shape[::-1], interpolation=cv2.INTER_NEAREST)

            # zero-case handling
            if mask.sum() == 0:
                acc_list.append(1.0 if pred_warped.sum() == 0 else 0.0)
                continue

            inter = np.logical_and(pred_warped, mask).sum()
            union = np.logical_or(pred_warped, mask).sum()
            iou_list.append(inter / (union + 1e-6))
            cum_I += inter
            cum_U += union

    precision = {}
    if iou_list:
        ious_tensor = torch.tensor(iou_list, device=args.device)
        mean_iou = ious_tensor.mean().item()
        precision = {f'Pr@{t}0': (ious_tensor > t/10).float().mean().item() for t in range(5, 10)}
    else:
        mean_iou = 0.0

    overall_iou = cum_I / (cum_U + 1e-6) if cum_U > 0 else 0.0
    mean_acc = float(np.mean(acc_list)) if acc_list else 0.0

    logger.info('=> Inference Metrics <=')
    if getattr(args, 'dataset', '') == 'ref-zom':
        logger.info(f"Overall IoU   = {overall_iou*100:.2f}%")
        logger.info(f"Mean IoU      = {mean_iou*100:.2f}%")
        logger.info(f"Zero-case acc = {mean_acc*100:.2f}%")
        return overall_iou, precision
    else:
        logger.info(f"Mean IoU      = {mean_iou*100:.2f}%")
        return mean_iou, precision
