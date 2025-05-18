'''
engine/engine.py

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
    Args:
        pred_bin (np.ndarray): Predicted binary mask.
        mask_bin (np.ndarray): Ground truth binary mask.
        is_zero_case (bool): Whether this is a zero-case (no object) sample.
    Returns:
        inter (float): Intersection pixel count.
        union (float): Union pixel count.
        acc0 (float): Zero-case accuracy (if applicable), or NaN otherwise.
    """
    if is_zero_case:
        # For zero-case (no object), only accuracy is meaningful
        return 0.0, 0.0, float(pred_bin.sum() == 0)
    inter = np.logical_and(pred_bin, mask_bin).sum()
    union = np.logical_or(pred_bin, mask_bin).sum()
    return inter, union, np.nan


# ------------------------ Training ------------------------ #
def train(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    """
    Run one epoch of training.
    """
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
    """
    Run validation over the dataset.
    Logs and returns overall IoU, mean IoU (non-zero-case), zero-case accuracy,
    and prints precision@50-90 for non-zero-case samples.
    """
    model.eval()
    cum_I = cum_U = 0.0
    inter_list, union_list, acc0_list = [], [], []
    zero_case_count = 0

    loader = tqdm(val_loader, desc=f"Validation {epoch}/{args.epochs}", ncols=100)
    for batch in loader:
        if len(batch) == 4:
            imgs, texts, mask_ts, params = batch
        else:
            imgs, texts, params = batch
            mask_ts = None

        imgs  = imgs.to(args.device, non_blocking=True)
        texts = texts.to(args.device, non_blocking=True)

        preds = torch.sigmoid(model(imgs, texts))
        if preds.shape[-2:] != imgs.shape[-2:]:
            preds = F.interpolate(preds, size=imgs.shape[-2:], mode='bicubic', align_corners=True)

        src_types = params.get('source_type', [''] * preds.size(0))
        if isinstance(src_types, (str, int)):
            src_types = [src_types] * preds.size(0)

        for i, pred in enumerate(preds):
            pred_np = pred.squeeze().cpu().numpy()

            src = src_types[i]
            is_zero = str(src).lower() == 'zero'

            inv = params['inverse'][i]
            M_inv = (inv.cpu().numpy() if torch.is_tensor(inv) else np.asarray(inv, np.float32))
            sz = params['ori_size'][i]
            h, w = int(sz[0]), int(sz[1])

            pred_warp = cv2.warpAffine(pred_np, M_inv, (w, h), cv2.INTER_CUBIC) > 0.35

            if mask_ts is not None:
                m_np = mask_ts[i].cpu().numpy()
                mask_bin = cv2.warpAffine(m_np, M_inv, (w, h), cv2.INTER_NEAREST) > 0.5
            else:
                mask_path = params['mask_dir'][i]
                mask_raw  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_raw is None:
                    continue
                mask_bin = (mask_raw.astype(np.float32)/255.0) > 0.5
                if pred_warp.shape != mask_bin.shape:
                    mask_bin = cv2.resize(mask_bin.astype(np.uint8),
                                           pred_warp.shape[::-1],
                                           cv2.INTER_NEAREST).astype(bool)

            inter, union, acc0 = compute_iou(pred_warp, mask_bin, is_zero)

            if is_zero:
                zero_case_count += 1
                if not np.isnan(acc0):
                    acc0_list.append(acc0)
                continue

            if not np.isnan(inter):
                inter_list.append(inter)
                union_list.append(union)
                cum_I += inter
                cum_U += union

    mean_iou  = np.mean([i/(u+1e-6) for i,u in zip(inter_list, union_list)]) if inter_list else 0.0
    overall   = cum_I / (cum_U + 1e-6) if cum_U > 0 else 0.0
    mean_acc0 = float(np.mean(acc0_list)) if acc0_list else 0.0

    logger.info(f"[Val] Overall IoU    = {overall*100:.2f}%")
    logger.info(f"[Val] Mean IoU       = {mean_iou*100:.2f}%")
    if getattr(args, 'dataset', '') == 'ref-zom':
        logger.info(f"[Val] Zero-case acc  = {mean_acc0*100:.2f}%")
        logger.info(f"[Val] Zero-case count= {zero_case_count}")

    # ---- Print Pr@50~Pr@90 for non-zero cases ----
    if inter_list and union_list:
        ious = np.array([i / (u + 1e-6) for i, u in zip(inter_list, union_list)])
        for t in range(5, 10):
            pr = (ious > t / 10).mean()
            logger.info(f"[Val] Pr@{t}0 = {pr * 100:.2f}%")
    else:
        for t in range(5, 10):
            logger.info(f"[Val] Pr@{t}0 = 0.00%")

    if getattr(args, 'dataset', '') == 'ref-zom':
        return overall, {}
    else:
        return mean_iou, {}

# ------------------------ Inference (with metrics) ------------------------ #
@torch.no_grad()
def inference(test_loader, model, args):
    """
    Run inference on test data and compute metrics.
    Logs and returns overall IoU, mean IoU (non-zero-case), zero-case accuracy,
    and precision@90/80/70/60/50.
    """
    model.eval()
    cum_I = cum_U = 0.0
    iou_list = []
    acc0_list = []
    zero_case_count = 0

    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
    pr_counts = {t: 0 for t in thresholds}
    total_non_zero = 0

    loader = tqdm(test_loader, desc='Inference', ncols=100)
    for imgs, txts, mask_ts, params in loader:
        B = imgs.size(0)
        imgs   = imgs.to(args.device, non_blocking=True)
        txts   = txts.to(args.device, non_blocking=True)

        # Forward once per batch
        preds = torch.sigmoid(model(imgs, txts))
        preds = preds.squeeze(1).cpu().numpy()

        for i in range(B):
            pred_np = preds[i]
            # Recover affine matrix and original size
            inv = params['inverse'][i]
            M_inv = inv  if isinstance(inv, np.ndarray) else inv.cpu().numpy()
            h, w  = params['ori_size'][i]
            pred_warp = cv2.warpAffine(pred_np, M_inv, (w, h), cv2.INTER_CUBIC) > 0.35

            src = params['source_type'][i]
            is_zero = str(src).lower() == 'zero'

            # GT mask
            m_np = mask_ts[i].numpy()
            gt = cv2.warpAffine(m_np, M_inv, (w, h), cv2.INTER_NEAREST) > 0.5

            # Compute
            inter, union, acc0 = compute_iou(pred_warp, gt, is_zero)
            if is_zero:
                zero_case_count += 1
                acc0_list.append(acc0)
            else:
                total_non_zero += 1
                iou = inter/(union+1e-6)
                iou_list.append(iou)
                cum_I += inter
                cum_U += union
                for t in thresholds:
                    if iou >= t:
                        pr_counts[t] += 1

    # Summary
    mean_iou  = float(np.mean(iou_list)) if iou_list else 0.0
    overall   = cum_I / (cum_U + 1e-6)   if cum_U > 0 else 0.0
    mean_acc0 = float(np.mean(acc0_list)) if acc0_list else 0.0
    pr = {f"PR@{int(t*100)}": pr_counts[t]/total_non_zero if total_non_zero>0 else 0.0
          for t in thresholds}

    logger.info("=> Inference Metrics <=")
    logger.info(f"[Test] overall_IoU   = {overall*100:.2f}%")
    logger.info(f"[Test] mean_IoU      = {mean_iou*100:.2f}%")
    logger.info(f"[Test] zero-case acc = {mean_acc0*100:.2f}%")
    for name, val in pr.items():
        logger.info(f"[Test] {name} = {val*100:.2f}%")

    return mean_iou, pr
