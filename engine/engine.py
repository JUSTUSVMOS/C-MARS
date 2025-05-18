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


def compute_iou(pred_bin: np.ndarray, mask_bin: np.ndarray, is_zero_case: bool):
    """
    Compute intersection, union, and zero-case accuracy.
    Returns:
        inter (float): intersection pixel count
        union (float): union pixel count
        acc0  (float): zero-case accuracy, or NaN for non-zero-case
    """
    if is_zero_case:
        # For zero-case (no object), return accuracy only
        return 0.0, 0.0, float(pred_bin.sum() == 0)
    inter = np.logical_and(pred_bin, mask_bin).sum()
    union = np.logical_or(pred_bin, mask_bin).sum()
    return inter, union, np.nan


# ------------------------ Training ------------------------ #
def train(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time  = AverageMeter('Data',  ':2.2f')
    lr_meter   = AverageMeter('LR',    ':1.6f')
    loss_meter = AverageMeter('Loss',  ':2.4f')
    iou_meter  = AverageMeter('IoU',   ':2.2f')
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
        assert img is not None, f"batch {idx} img is None"
        data_time.update(time.time() - end)
        img, txt, tgt = [x.to(args.device, non_blocking=True) for x in (img, txt, tgt)]
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
    model.eval()
    cum_I = cum_U = 0.0
    inter_list, union_list, acc0_list = [], [], []
    zero_case_count = 0

    loader = tqdm(val_loader, desc=f"Validation {epoch}/{args.epochs}", ncols=100)
    for imgs, texts, params in loader:
        imgs = imgs.to(args.device, non_blocking=True)
        texts = texts.to(args.device, non_blocking=True)
        preds = torch.sigmoid(model(imgs, texts))
        if preds.shape[-2:] != imgs.shape[-2:]:
            preds = F.interpolate(preds, size=imgs.shape[-2:], mode='bicubic', align_corners=True)

        src_types = params.get('source_type', [''] * preds.shape[0])
        if isinstance(src_types, (str, int)):
            src_types = [src_types] * preds.shape[0]

        for pred, mpath, inv, sz, src in zip(
            preds, params['mask_dir'], params['inverse'], params['ori_size'], src_types):

            # Robust source type determination
            if isinstance(src, (list, tuple)) and len(src) > 0:
                src_str = src[0]
            else:
                src_str = src
            is_zero = str(src_str).lower() == 'zero'

            pred_np = pred.squeeze().cpu().numpy()

            # Recover affine matrix
            if isinstance(inv, torch.Tensor):
                mat = inv[0].cpu().numpy()
            elif isinstance(inv, list):
                mat = inv[0]
            else:
                mat = inv
            M_inv = np.asarray(mat, dtype=np.float32)
            assert M_inv.shape == (2, 3)

            # Recover original size
            if isinstance(sz, torch.Tensor):
                raw = sz[0] if sz.ndim > 1 else sz
                sz0 = raw.cpu().numpy()
            elif isinstance(sz, (list, tuple)):
                raw = sz[0]
                sz0 = raw.cpu().numpy() if isinstance(raw, torch.Tensor) else np.array(raw)
            else:
                arr = np.array(sz)
                sz0 = arr[0] if arr.ndim > 1 else arr
            h, w = int(sz0[0]), int(sz0[1])

            # Warp prediction back to original size and threshold
            pred_warp = cv2.warpAffine(pred_np, M_inv, (w, h), cv2.INTER_CUBIC) > 0.35

            mask_path = mpath[0] if isinstance(mpath, (list, tuple)) else mpath
            mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_raw is None:
                continue
            mask_bin = (mask_raw.astype(np.float32) / 255.0) > 0.5
            if pred_warp.shape != mask_bin.shape:
                mask_bin = cv2.resize(mask_bin.astype(np.uint8), pred_warp.shape[::-1], cv2.INTER_NEAREST).astype(bool)

            inter, union, acc0 = compute_iou(pred_warp, mask_bin, is_zero)

            if is_zero:
                zero_case_count += 1
                if not np.isnan(acc0):
                    acc0_list.append(acc0)
                # Skip adding inter/union for zero-case
                continue

            # Accumulate metrics for non-zero-case
            if not np.isnan(inter):
                inter_list.append(inter)
                union_list.append(union)
                cum_I += inter
                cum_U += union

    mean_iou = np.mean([i/(u+1e-6) for i, u in zip(inter_list, union_list)]) if inter_list else 0.0
    overall   = cum_I / (cum_U + 1e-6) if cum_U > 0 else 0.0
    mean_acc0 = float(np.mean(acc0_list)) if acc0_list else 0.0

    if getattr(args, 'dataset', '') == 'ref-zom':
        logger.info(f"[Val] Overall IoU    = {overall*100:.2f}%")
        logger.info(f"[Val] Mean IoU       = {mean_iou*100:.2f}%")
        logger.info(f"[Val] Zero-case acc  = {mean_acc0*100:.2f}%")
        logger.info(f"[Val] Zero-case count= {zero_case_count}")
        return overall, {}
    else:
        logger.info(f"[Val] Mean IoU       = {mean_iou*100:.2f}%")
        return mean_iou, {}

# ------------------------ Inference ------------------------ #
@torch.no_grad()
def inference(test_loader, model, args):
    model.eval()
    cum_I = cum_U = 0.0
    inter_list, union_list, acc0_list = [], [], []

    loader = tqdm(test_loader, desc='Inference', ncols=100)
    for img, params in loader:
        img = img.to(args.device, non_blocking=True)
        mask_path = params['mask_dir'][0]
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            continue
        mask_bin = (mask_raw.astype(np.float32) / 255.0) > 0.5

        mat = params['inverse'][0]
        M_inv = np.asarray(mat, dtype=np.float32)
        sz_arr = np.array(params['ori_size'][0])
        h, w = int(sz_arr[0]), int(sz_arr[1])

        sents = params['sents']
        src_types = params.get('source_type', [''] * len(sents))
        for sent, src in zip(sents, src_types):
            text = tokenize(sent, args.word_len, True).to(args.device)
            pred = torch.sigmoid(model(img, text)).squeeze(0)
            if pred.shape[-2:] != img.shape[-2:]:
                pred = F.interpolate(pred.unsqueeze(0), size=img.shape[-2:], mode='bicubic', align_corners=True).squeeze(0)

            pred_np   = pred.cpu().numpy()
            pred_warp = cv2.warpAffine(pred_np, M_inv, (w, h), cv2.INTER_CUBIC) > 0.35

            is_zero = str(src).lower() == 'zero'
            inter, union, acc0 = compute_iou(pred_warp, mask_bin, is_zero)

            if is_zero:
                if not np.isnan(acc0):
                    acc0_list.append(acc0)
                # Skip inter/union accumulation for zero-case
                continue

            if not np.isnan(inter):
                inter_list.append(inter)
                union_list.append(union)
                cum_I += inter
                cum_U += union

    mean_iou  = float(np.sum(inter_list) / (np.sum(union_list) + 1e-6)) if union_list else 0.0
    overall   = cum_I / (cum_U + 1e-6) if cum_U > 0 else 0.0
    mean_acc0 = float(np.mean(acc0_list)) if acc0_list else 0.0

    logger.info("=> Inference Metrics <=")
    if getattr(args, 'dataset', '') == 'ref-zom':
        logger.info(f"[Test] Overall IoU   = {overall*100:.2f}%")
        logger.info(f"[Test] Mean IoU      = {mean_iou*100:.2f}%")
        logger.info(f"[Test] Zero-case acc = {mean_acc0*100:.2f}%")
        return overall, {}
    else:
        logger.info(f"[Test] Mean IoU      = {mean_iou*100:.2f}%")
        return mean_iou, {}
