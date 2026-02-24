"""Panoptic Cascade Mask R-CNN with DINOv2 ViT-B/14 backbone.

Replaces the DINO ResNet-50 backbone with DINOv2 ViT-B/14 + SimpleFeaturePyramid,
keeping all CUPS heads (CustomCascadeROIHeads, CustomSemSegFPNHead, DropLoss).

The backbone swap is done by:
  1. Building the standard model from cfg (creates ResNet backbone + correct heads)
  2. Replacing model.backbone with DINOv2 + SimpleFeaturePyramid

Both backbones produce the same output shape (p2-p6 at 256 channels), so the
heads are configured identically regardless of which backbone was used during init.
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import torch
import torch.nn as nn
from detectron2.config import get_cfg

from cups.model.modeling import PanopticFPNWithTTA
from cups.model.modeling.meta_arch import build_model

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def panoptic_cascade_mask_r_cnn_vitb(
    num_clusters_things: int = 300,
    num_clusters_stuffs: int = 100,
    confidence_threshold: float = 0.4,
    class_weights: Tuple[float, ...] | None = None,
    use_tta: bool = False,
    tta_detection_threshold: float = 0.5,
    tta_scales: Tuple[float, ...] = (0.75, 1.0, 1.25, 1.5),
    default_size: Tuple[int, int] = (512, 1024),
    use_drop_loss: bool = True,
    drop_loss_iou_threshold: float = 0.2,
    freeze_backbone: bool = True,
    dinov2_model_name: str = "dinov2_vitb14_reg",
) -> nn.Module:
    """Build Panoptic Cascade Mask R-CNN with DINOv2 ViT-B/14 + SimpleFeaturePyramid.

    Same interface as panoptic_cascade_mask_r_cnn() but uses DINOv2 ViT-B/14
    instead of DINO ResNet-50. All CUPS augmentations (DropLoss, Copy-Paste,
    etc.) work unchanged.

    Args:
        num_clusters_things: Number of thing pseudo-classes.
        num_clusters_stuffs: Number of stuff pseudo-classes.
        confidence_threshold: Confidence threshold for object proposals.
        class_weights: Semantic class weights. Default None.
        use_tta: If True, TTA model is returned.
        tta_detection_threshold: Detection threshold for TTA.
        tta_scales: Scales for TTA.
        default_size: Default image size (H, W).
        use_drop_loss: If True, DropLoss is used.
        drop_loss_iou_threshold: IoU threshold for DropLoss.
        freeze_backbone: If True, freeze DINOv2 backbone parameters.
        dinov2_model_name: DINOv2 model name for torch.hub.

    Returns:
        model: Panoptic Cascade Mask R-CNN with DINOv2 ViT-B/14 backbone.
    """
    # ── Step 1: Build standard cfg (same as ResNet path) ──
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(
        os.path.join(os.path.dirname(__file__), "Panoptic-Cascade-Mask-R-CNN.yaml")
    )

    # Auto-detect device
    if torch.cuda.is_available():
        pass  # default "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        cfg.MODEL.DEVICE = "mps"
    else:
        cfg.MODEL.DEVICE = "cpu"

    if cfg.MODEL.DEVICE in ("cpu", "mps") and cfg.MODEL.RESNETS.NORM == "SyncBN":
        cfg.MODEL.RESNETS.NORM = "BN"
        cfg.MODEL.FPN.NORM = "BN"

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_clusters_things
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_clusters_stuffs
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.MODEL.SEM_SEG_HEAD.CLASS_WEIGHT = class_weights
    cfg.TEST.INSTANCE_SCORE_THRESH = tta_detection_threshold
    cfg.MODEL.ROI_HEADS.USE_DROPLOSS = use_drop_loss
    cfg.MODEL.ROI_HEADS.DROPLOSS_IOU_THRESH = drop_loss_iou_threshold
    if use_tta:
        cfg.TEST.AUG.MIN_SIZES = tuple(
            int(default_size[0] * scale) for scale in tta_scales
        )
    cfg.freeze()

    # ── Step 2: Build model with ResNet backbone (heads configured correctly) ──
    log.info("Building PanopticFPN with ResNet backbone (temporary)...")
    model = build_model(cfg)

    # ── Step 3: Replace backbone with DINOv2 + SimpleFeaturePyramid ──
    from .backbone_dinov2_vit import build_dinov2_vitb_fpn_backbone

    log.info(f"Replacing backbone with DINOv2 ViT-B/14 ({dinov2_model_name})...")
    dinov2_backbone = build_dinov2_vitb_fpn_backbone(
        out_channels=256,
        freeze=freeze_backbone,
        model_name=dinov2_model_name,
    )

    # Move to same device as model
    device = next(model.parameters()).device
    dinov2_backbone = dinov2_backbone.to(device)

    # Delete old ResNet backbone and replace
    del model.backbone
    model.backbone = dinov2_backbone

    # Log param counts
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(
        f"DINOv2 ViT-B/14 backbone: {backbone_params / 1e6:.1f}M params "
        f"(frozen={freeze_backbone})"
    )
    log.info(
        f"Total: {total_params / 1e6:.1f}M params, "
        f"trainable: {trainable_params / 1e6:.1f}M params"
    )

    # ── Step 4: TTA wrapper ──
    if use_tta:
        model = PanopticFPNWithTTA(cfg, model)
        log.info("TTA model is used.")

    return model
