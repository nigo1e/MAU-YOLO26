"""
YOLO26 stable training entry.

This file is a standalone copy of train.py with three training-stability
features collected in one place:
1. label smoothing for classification targets,
2. configurable gradient clipping,
3. explicit learning-rate/momentum warmup settings.

The original train.py is not modified.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss


# Stability knobs ----------------------------------------------------------------
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM = 10.0
WARMUP_EPOCHS = 3.0
WARMUP_MOMENTUM = 0.8
WARMUP_BIAS_LR = 0.1


def _smooth_targets(target_scores: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply BCE-style label smoothing to YOLO class assignment scores."""
    if eps <= 0:
        return target_scores
    eps = min(max(float(eps), 0.0), 1.0)
    smoothed = target_scores.clone()
    fg_mask = target_scores.sum(-1, keepdim=True).gt(0)
    if fg_mask.any():
        # Smooth only foreground anchors and preserve their assignment weight.
        smooth_mass = target_scores.sum(-1, keepdim=True) * eps / target_scores.shape[-1]
        smoothed = torch.where(fg_mask, target_scores * (1.0 - eps) + smooth_mass, smoothed)
    return smoothed


def enable_label_smoothing(eps: float = LABEL_SMOOTHING) -> None:
    """Patch YOLO detection losses so their classification targets are smoothed."""
    if eps <= 0:
        return

    for loss_cls in (v8DetectionLoss,):
        if getattr(loss_cls, "_stable_label_smoothing_patched", False):
            continue

        original_method = loss_cls.get_assigned_targets_and_loss

        def smoothed_get_assigned_targets_and_loss(self, preds, batch, _original=original_method):
            assigned, loss, loss_detach = _original(self, preds, batch)
            smoothing = getattr(self.hyp, "stable_label_smoothing", eps)
            if smoothing <= 0:
                return assigned, loss, loss_detach

            # Recompute only the classification BCE with smoothed targets, then
            # keep the original bbox and DFL losses from the framework.
            pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
            pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()

            # The parent method already did the assignment, but it does not
            # expose target_scores. Re-run the same lightweight assignment path
            # to obtain targets for a smoothed cls loss without editing core files.
            from ultralytics.utils.tal import make_anchors

            anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
            dtype = pred_scores.dtype
            batch_size = pred_scores.shape[0]
            imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            gt_bboxes = gt_bboxes.clone()
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt(0.0)

            pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

            _, _, target_scores, _, _ = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )
            target_scores_sum = max(target_scores.sum(), 1)
            smoothed_scores = _smooth_targets(target_scores.to(dtype), smoothing)
            cls_loss = self.bce(pred_scores, smoothed_scores).sum() / target_scores_sum * self.hyp.cls
            loss = torch.stack((loss[0], cls_loss, loss[2]))
            loss_detach = loss.detach()
            return assigned, loss, loss_detach

        loss_cls.get_assigned_targets_and_loss = smoothed_get_assigned_targets_and_loss
        loss_cls._stable_label_smoothing_patched = True

    LOGGER.info(f"Stable training: label smoothing enabled, eps={eps}.")


def enable_gradient_clipping(max_norm: float = GRAD_CLIP_NORM) -> None:
    """Patch BaseTrainer.optimizer_step so grad clipping max_norm is configurable."""
    if getattr(BaseTrainer, "_stable_gradient_clipping_patched", False):
        BaseTrainer._stable_grad_clip_norm = max_norm
        return

    def optimizer_step_with_configurable_clip(self):
        self.scaler.unscale_(self.optimizer)
        clip_norm = getattr(self.args, "grad_clip_norm", getattr(BaseTrainer, "_stable_grad_clip_norm", max_norm))
        if clip_norm and clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(clip_norm))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    BaseTrainer.optimizer_step = optimizer_step_with_configurable_clip
    BaseTrainer._stable_grad_clip_norm = max_norm
    BaseTrainer._stable_gradient_clipping_patched = True
    LOGGER.info(f"Stable training: gradient clipping enabled, max_norm={max_norm}.")


def train() -> SimpleNamespace:
    """Start YOLO26 training with stability settings."""
    enable_label_smoothing(LABEL_SMOOTHING)
    enable_gradient_clipping(GRAD_CLIP_NORM)

    model = YOLO("ultralytics/cfg/models/26/yolo26.yaml")

    results = model.train(
        data="coco128.yaml",
        epochs=100,
        imgsz=640,
        optimizer="MuSGD",
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=WARMUP_MOMENTUM,
        warmup_bias_lr=WARMUP_BIAS_LR,
    )

    return results


if __name__ == "__main__":
    train()
