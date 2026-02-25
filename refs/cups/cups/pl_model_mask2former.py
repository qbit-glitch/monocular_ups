"""PyTorch Lightning module for Mask2Former training within the CUPS pipeline.

Subclasses UnsupervisedModel to reuse validation, metrics, augmentations,
and logging — overriding only training_step and configure_optimizers.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.optim
from detectron2.utils.events import EventStorage
from torch import Tensor
from yacs.config import CfgNode

from cups.model.model import (
    filter_predictions,
    prediction_to_label_format,
)
from cups.model.model_mask2former import Mask2FormerWrapper, build_mask2former
from cups.pl_model_pseudo import UnsupervisedModel

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Mask2FormerUnsupervisedModel(UnsupervisedModel):
    """Lightning module for Mask2Former trained on pseudo-labels.

    Inherits validation, PQ metrics, and augmentation pipeline from
    UnsupervisedModel. Overrides training to work with Mask2Former's
    loss format and optimizer configuration.
    """

    def training_step(self, batch: List[Dict[str, Any]], batch_index: int) -> Dict[str, Tensor]:
        """Training step adapted for Mask2Former.

        Mask2Former returns losses with different keys than Cascade R-CNN:
        - loss_ce (classification cross-entropy per aux layer)
        - loss_mask_* (binary cross-entropy mask loss per aux layer)
        - loss_dice_* (dice loss per aux layer)
        """
        # Make storage object (required by Detectron2 internals, even if unused)
        if self.storage is None:
            self.storage = EventStorage(0)
            self.storage.__enter__()

        # Copy-paste augmentation (same logic as base class)
        if self.copy_paste_augmentation is not None:
            if self.global_step < self.hparams.config.AUGMENTATION.NUM_STEPS_STARTUP:
                batch_aug = self.copy_paste_augmentation(batch, deepcopy(batch))
            else:
                if self.prediction_temp is not None:
                    batch_aug = self.copy_paste_augmentation(self.prediction_temp, deepcopy(batch))
                else:
                    batch_aug = deepcopy(batch)
        else:
            batch_aug = deepcopy(batch)

        # Apply photometric + resolution jitter augmentations
        batch_aug = self.photometric_augmentation(self.resolution_jitter_augmentation(batch_aug))

        # Forward pass — Mask2Former returns loss dict
        self.model.train()
        loss_dict = self.model(batch_aug)

        # Compute total loss (guard against empty dict edge case)
        if not loss_dict:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            loss: Tensor = sum(loss_dict.values())

        # Log losses
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        for key, value in loss_dict.items():
            self.log("losses/" + key, value, sync_dist=True)

        # Make inference prediction for copy-paste augmentation seed
        self.model.eval()
        with torch.no_grad():
            prediction = self.model([{"image": sample["image"]} for sample in batch])
            self.prediction_temp = prediction_to_label_format(
                prediction,
                [sample["image"] for sample in batch],
                confidence_threshold=self.hparams.config.AUGMENTATION.CONFIDENCE,
            )
            self.prediction_temp = filter_predictions(self.prediction_temp, batch)
        self.model.train()

        # Log visualizations periodically
        if ((self.global_step) % self.hparams.config.TRAINING.LOG_MEDIA_N_STEPS) == 0:
            self.log_visualizations(batch, prediction)

        return {"loss": loss}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Optimizer with differential learning rates for Mask2Former.

        - Backbone (Swin-L): frozen or very low LR
        - Pixel decoder: base LR
        - Transformer decoder: base LR
        - Class embed (reinitialized): higher LR
        """
        m2f_config = self.hparams.config.MODEL.MASK2FORMER
        base_lr = self.hparams.config.TRAINING.ADAMW.LEARNING_RATE
        backbone_lr_mult = m2f_config.BACKBONE_LR_MULTIPLIER
        weight_decay = m2f_config.WEIGHT_DECAY

        # Group parameters
        backbone_params = []
        decoder_params = []
        class_embed_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "pixel_level_module.encoder" in name:
                backbone_params.append(param)
            elif "class_embed" in name or "class_predictor" in name:
                class_embed_params.append(param)
            else:
                decoder_params.append(param)

        param_groups = []
        if backbone_params and backbone_lr_mult > 0:
            param_groups.append({
                "params": backbone_params,
                "lr": base_lr * backbone_lr_mult,
                "weight_decay": weight_decay,
            })
        if decoder_params:
            param_groups.append({
                "params": decoder_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
            })
        class_embed_lr_mult = m2f_config.CLASS_EMBED_LR_MULTIPLIER
        if class_embed_params:
            param_groups.append({
                "params": class_embed_params,
                "lr": base_lr * class_embed_lr_mult,
                "weight_decay": weight_decay,
            })

        optimizer = torch.optim.AdamW(
            params=param_groups,
            lr=base_lr,
            weight_decay=weight_decay,
            betas=self.hparams.config.TRAINING.ADAMW.BETAS,
        )
        log.info(
            f"AdamW optimizer: backbone_lr={base_lr * backbone_lr_mult}, "
            f"decoder_lr={base_lr}, class_embed_lr={base_lr * class_embed_lr_mult}, "
            f"weight_decay={weight_decay}"
        )
        return optimizer


def build_model_mask2former(
    config: CfgNode,
    thing_pseudo_classes: Optional[Tuple[int, ...]] = None,
    stuff_pseudo_classes: Optional[Tuple[int, ...]] = None,
    thing_classes: Set[int] = set(),
    stuff_classes: Set[int] = set(),
    copy_paste_augmentation: Optional[nn.Module] = nn.Identity(),
    photometric_augmentation: nn.Module = nn.Identity(),
    resolution_jitter_augmentation: nn.Module = nn.Identity(),
    class_weights: Optional[Tuple[float, ...]] = None,
    class_names: Optional[List[str]] = None,
    classes_mask: Optional[List[bool]] = None,
) -> Mask2FormerUnsupervisedModel:
    """Build a Mask2Former model wrapped in the CUPS Lightning module.

    Args:
        config: CUPS config with MODEL.MASK2FORMER section.
        thing_pseudo_classes: Pseudo-label thing class IDs (from dataset distribution).
        stuff_pseudo_classes: Pseudo-label stuff class IDs (from dataset distribution).
        thing_classes: GT thing class IDs for validation PQ metric.
        stuff_classes: GT stuff class IDs for validation PQ metric.
        copy_paste_augmentation: Copy-paste augmentation module.
        photometric_augmentation: Photometric augmentation module.
        resolution_jitter_augmentation: Resolution jitter augmentation module.
        class_weights: Unused for Mask2Former (uses no_object_weight instead).
        class_names: Class names for logging.
        classes_mask: Validation class mask.

    Returns:
        Mask2FormerUnsupervisedModel ready for Lightning training.
    """
    assert thing_pseudo_classes is not None and stuff_pseudo_classes is not None, \
        "thing_pseudo_classes and stuff_pseudo_classes must be provided."

    m2f_config = config.MODEL.MASK2FORMER
    num_things = len(thing_pseudo_classes)
    num_stuffs = len(stuff_pseudo_classes)
    num_classes = num_things + num_stuffs

    log.info(f"Building Mask2Former: {num_classes} classes ({num_stuffs} stuff + {num_things} things)")

    # Build the Mask2Former model
    model: nn.Module = build_mask2former(
        num_classes=num_classes,
        num_stuff_classes=num_stuffs,
        pretrained=m2f_config.PRETRAINED,
        num_queries=m2f_config.NUM_QUERIES,
        freeze_backbone=(m2f_config.BACKBONE_LR_MULTIPLIER == 0.0),
        no_object_weight=m2f_config.NO_OBJECT_WEIGHT,
        confidence_threshold=config.MODEL.INFERENCE_CONFIDENCE_THRESHOLD,
    )

    # Wrap in Lightning module
    lightning_model = Mask2FormerUnsupervisedModel(
        model=model,
        num_thing_pseudo_classes=num_things,
        num_stuff_pseudo_classes=num_stuffs,
        config=config,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
        copy_paste_augmentation=copy_paste_augmentation,
        photometric_augmentation=photometric_augmentation,
        resolution_jitter_augmentation=resolution_jitter_augmentation,
        class_names=class_names,
        classes_mask=classes_mask,
    )

    return lightning_model
