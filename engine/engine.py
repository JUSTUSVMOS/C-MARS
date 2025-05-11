# engine/engine.py
import os, time, cv2, numpy as np, torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger

from utils.dataset import tokenize
from utils.misc import AverageMeter, ProgressMeter, trainMetricGPU

# ──────────────────────────────  Train  ────────────────────────────── #
def train(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time  = AverageMeter('Data',  ':2.2f')
    lr_meter   = AverageMeter('Lr',    ':1.6f')
    loss_m     = AverageMeter('Loss',  ':2.4f')
    iou_m      = AverageMeter('IoU',   ':2.2f')
    pr_m       = AverageMeter('Prec@50', ':2.2f')
    prog = ProgressMeter(len(train_loader),
                         [batch_time, data_time, lr_meter, loss_m, iou_m, pr_m],
                         prefix=f"Training: Epoch=[{epoch}/{args.epochs}]")

    model.train();  time.sleep(2);  end = time.time()
    train_loader = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}",
                        ncols=120, mininterval=120)

    for i, (img, txt, tgt) in enumerate(train_loader):
        data_time.update(time.time() - end)
        img, txt, tgt = (x.to(args.device, non_blocking=True) for x in (img, txt, tgt))
        tgt = tgt.unsqueeze(1)

        with amp.autocast():
            pred, tgt, loss = model(img, txt, tgt)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer);  scaler.update()

        iou, pr5 = trainMetricGPU(pred, tgt, 0.35, 0.5)
        loss_m.update(loss.item(), img.size(0))
        iou_m .update(iou .item(), img.size(0))
        pr_m  .update(pr5.item(), img.size(0))
        lr_meter.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end);  end = time.time()

        if (i + 1) % args.print_freq == 10:
            prog.display(i + 1)
            train_loader.set_postfix(Loss=loss_m.avg, IoU=iou_m.avg,
                                     Prec50=pr_m.avg, Lr=lr_meter.avg,
                                     Time=batch_time.avg)

# ────────────────────────────── Validate ────────────────────────────── #
@torch.no_grad()
def validate(val_loader, model, epoch, args):
    def first(x):
        if isinstance(x, (list, tuple)):       return x[0]
        if torch.is_tensor(x) and x.ndim:      return x[0]
        if isinstance(x, np.ndarray) and x.ndim: return x[0]
        return x

    model.eval()
    iou_acc = []

    tbar = tqdm(val_loader, desc=f"Validation {epoch}/{args.epochs}", ncols=100)
    for imgs, texts, param in tbar:
        imgs  = imgs.to(args.device, non_blocking=True)
        texts = texts.to(args.device, non_blocking=True)

        preds = torch.sigmoid(model(imgs, texts))
        if preds.shape[-2:] != imgs.shape[-2:]:
            preds = F.interpolate(preds, size=imgs.shape[-2:], mode='bicubic',
                                  align_corners=True)

        for pred, md, M, osz in zip(preds,
                                    param['mask_dir'],
                                    param['inverse'],
                                    param['ori_size']):
            h, w = map(int, first(osz))
            M    = np.array(first(M))
            pred = pred.squeeze().cpu().numpy()
            pred = cv2.warpAffine(pred, M, (w, h), flags=cv2.INTER_CUBIC) > 0.35

            mask = cv2.imread(first(md), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = mask/255.

            # shape 容錯
            if pred.shape != mask.shape:
                if pred.T.shape == mask.shape:
                    pred = pred.T
                elif mask.T.shape == pred.shape:
                    mask = mask.T
                else:
                    mask = cv2.resize(mask, pred.shape[::-1],
                                      interpolation=cv2.INTER_NEAREST)

            # zero-mask rule
            if mask.sum() == 0:
                iou = 1.0 if pred.sum() == 0 else 0.0
            else:
                inter = np.logical_and(pred, mask)
                union = np.logical_or(pred,  mask)
                iou   = inter.sum() / (union.sum()+1e-6)
            iou_acc.append(iou)

    ious_t = torch.tensor(iou_acc, device=args.device)
    prec = {f'Pr@{t}0': (ious_t > t/10).float().mean().item()
            for t in range(5,10)}
    return ious_t.mean().item(), prec

    return ious.mean().item(), prec

# ────────────────────────────── Inference ───────────────────────────── #
# engine/engine.py
import os, time, cv2, numpy as np, torch
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
from utils.dataset import tokenize
from utils.misc import AverageMeter, ProgressMeter, trainMetricGPU

# ---------- train / validate 與原先一致，略 ---------- #
# （此處保留你的 train()、validate()，未做任何改動）

# ---------------------------- Inference ---------------------------- #
@torch.no_grad()
def inference(test_loader, model, args):
    def first(x):
        if isinstance(x,(list,tuple)):      return x[0]
        if torch.is_tensor(x) and x.ndim:   return x[0]
        if isinstance(x,np.ndarray) and x.ndim: return x[0]
        return x

    ious=[];  model.eval();  time.sleep(2)
    tbar=tqdm(test_loader, desc='Inference', ncols=100)

    for img, param in tbar:
        img = img.to(args.device, non_blocking=True)

        mask_path = first(param['mask_dir'])
        mask_raw  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:      # 缺檔跳過
            continue
        mask_raw = mask_raw / 255.

        M_inv = np.array(first(param['inverse']))
        h, w  = map(int, first(param['ori_size']))
        sents = first(param['sents'])
        N     = len(sents)

        # ----- batch pairing (影像重複 N 份) -----
        img_batch = img.repeat(N,1,1,1)                       # N×3×H×W
        txt_batch = tokenize(sents, args.word_len, True).to(args.device)

        preds = torch.sigmoid(model(img_batch, txt_batch))    # N×1×h×w
        if preds.shape[-2:] != img.shape[-2:]:
            preds = F.interpolate(preds, size=img.shape[-2:], mode='bicubic',
                                  align_corners=True)

        for pred in preds:
            pred = pred.squeeze().cpu().numpy()
            pred = cv2.warpAffine(pred, M_inv, (w,h),
                                   flags=cv2.INTER_CUBIC) > 0.35

            mask = mask_raw.copy()
            # ---- shape align ----
            if pred.shape != mask.shape:
                if pred.T.shape == mask.shape: pred = pred.T
                elif mask.T.shape == pred.shape: mask = mask.T
                else: mask = cv2.resize(mask, (pred.shape[1], pred.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)

            # ---- IoU with zero-rule ----
            if mask.sum() == 0:
                iou = 1.0 if pred.sum()==0 else 0.0
            else:
                inter = np.logical_and(pred, mask)
                union = np.logical_or(pred,  mask)
                iou   = inter.sum() / (union.sum()+1e-6)
            ious.append(iou)

    # --------- report ---------
    ious_t=torch.tensor(ious, device=args.device)
    logger.info('=> Metric Calculation <=')
    logger.info(f"IoU={ious_t.mean()*100:.2f}")
    for t in range(5,10):
        logger.info(f"Pr@{t}0: {(ious_t>t/10).float().mean()*100:.2f}")
    return ious_t.mean().item(), {}
