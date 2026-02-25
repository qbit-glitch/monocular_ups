"""PyTorch Lightning module for Mask2Former self-training (Stage-3).

EMA teacher-student framework: the teacher (EMA copy of the student) generates
panoptic pseudo-labels on-the-fly from raw images, and the student trains on them.

Analogous to pl_model_self.py (Cascade Mask R-CNN) but adapted for the
HuggingFace Mask2Former architecture.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.optim
from detectron2.structures import BitMasks, Boxes, Instances
from detectron2.utils.events import EventStorage
from torch import Tensor
from yacs.config import CfgNode

from cups.augmentation import RandomCrop
from cups.data.utils import get_bounding_boxes, instances_to_masks
from cups.model.model_mask2former import Mask2FormerWrapper, build_mask2former
from cups.pl_model_mask2former import Mask2FormerUnsupervisedModel

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Mask2FormerSelfSupervisedModel(Mask2FormerUnsupervisedModel):
    """Self-supervised (Stage-3) training for Mask2Former.

    Inherits validation, metrics, and augmentation pipeline from
    Mask2FormerUnsupervisedModel.  Overrides training_step to use an
    EMA teacher that generates pseudo-labels instead of fixed pseudo-labels.
    """

    def __init__(
        self,
        model: Mask2FormerWrapper,
        teacher_model: Mask2FormerWrapper,
        num_thing_pseudo_classes: int,
        num_stuff_pseudo_classes: int,
        config: CfgNode,
        thing_classes: Set[int],
        stuff_classes: Set[int],
        copy_paste_augmentation: Optional[nn.Module] = nn.Identity(),
        photometric_augmentation: nn.Module = nn.Identity(),
        resolution_jitter_augmentation: nn.Module = nn.Identity(),
        class_names: Optional[List[str]] = None,
        classes_mask: Optional[List[bool]] = None,
        ema_momentum: float = 0.999,
    ) -> None:
        super().__init__(
            model=model,
            num_thing_pseudo_classes=num_thing_pseudo_classes,
            num_stuff_pseudo_classes=num_stuff_pseudo_classes,
            config=config,
            thing_classes=thing_classes,
            stuff_classes=stuff_classes,
            copy_paste_augmentation=copy_paste_augmentation,
            photometric_augmentation=photometric_augmentation,
            resolution_jitter_augmentation=resolution_jitter_augmentation,
            class_names=class_names,
            classes_mask=classes_mask,
        )
        # EMA teacher (deep copy of student, no gradients)
        self.teacher_model: Mask2FormerWrapper = teacher_model
        self.ema_momentum: float = ema_momentum
        # Crop module for self-training augmentation
        if config.DATA.DATASET == "kitti":
            self.crop_module: nn.Module = RandomCrop(
                resolution_max=368, resolution_min=288, long_side_scale=3.369
            )
        else:
            self.crop_module = RandomCrop()
        # Self-training round counter
        self.round: int = 1

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: List[Dict[str, Any]], batch_index: int) -> Dict[str, Tensor]:
        """Self-training step: teacher pseudo-labels → student trains.

        1. Teacher generates panoptic predictions (eval, no grad).
        2. Predictions are converted to Detectron2-format pseudo-labels.
        3. Augmentations are applied (copy-paste, photometric, crop, jitter).
        4. Student forward pass on augmented pseudo-labels → loss.
        """
        # Ensure Detectron2 EventStorage exists
        if self.storage is None:
            self.storage = EventStorage(0)
            self.storage.__enter__()

        # --- 1. Teacher generates predictions ---
        self.teacher_model.eval()
        with torch.no_grad():
            predictions = self.teacher_model(batch)

        # --- 2. Convert predictions to Detectron2-format pseudo-labels ---
        pseudo_labels = self.make_pseudo_labels(
            predictions, batch, self.hparams.stuff_pseudo_classes
        )

        # --- 3. Apply augmentations ---
        if self.copy_paste_augmentation is not None:
            pseudo_labels = self.copy_paste_augmentation(pseudo_labels, pseudo_labels)
        pseudo_labels = self.photometric_augmentation(pseudo_labels)
        pseudo_labels = self.crop_module(pseudo_labels)
        pseudo_labels = self.resolution_jitter_augmentation(pseudo_labels)

        # Sanitize sem_seg: set out-of-range class IDs to ignore (255)
        num_classes = self.model.num_classes
        for sample in pseudo_labels:
            sem = sample["sem_seg"]
            invalid = (sem < 0) | ((sem >= num_classes) & (sem != 255))
            if invalid.any():
                sem[invalid] = 255
            sample["sem_seg"] = sem

        # --- 4. Student forward pass ---
        self.model.train()
        loss_dict = self.model(pseudo_labels)

        if not loss_dict:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            loss: Tensor = sum(loss_dict.values())

        # --- 5. Logging ---
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        for key, value in loss_dict.items():
            self.log("losses/" + key, value, sync_dist=True)

        # Periodic visualization
        if (self.global_step % self.hparams.config.TRAINING.LOG_MEDIA_N_STEPS) == 0:
            self.model.eval()
            with torch.no_grad():
                prediction = self.model([{"image": s["image"]} for s in pseudo_labels])
            self.model.train()
            self.log_visualizations(pseudo_labels, prediction)

        return {"loss": loss}

    # ------------------------------------------------------------------
    # Pseudo-label generation (mirrors SelfSupervisedModel.make_pseudo_labels)
    # ------------------------------------------------------------------

    def make_pseudo_labels(
        self,
        predictions: List[Dict[str, Any]],
        images: List[Dict[str, Any]],
        stuff_classes: Tuple[int, ...],
    ) -> List[Dict[str, Any]]:
        """Convert teacher predictions to Detectron2-format pseudo-labels.

        The prediction format from Mask2FormerWrapper._forward_inference()
        matches the format from PanopticFPNWithTTA, so this method is
        functionally identical to SelfSupervisedModel.make_pseudo_labels().

        Args:
            predictions: Teacher output — list of dicts with 'panoptic_seg'
                         (Tuple[Tensor, List[Dict]]) and 'sem_seg' (Tensor).
            images: Raw batch — list of dicts with 'image' (Tensor).
            stuff_classes: Tuple of stuff pseudo-class indices.

        Returns:
            Pseudo-labels in Detectron2 format (image + sem_seg + instances).
        """
        pseudo_labels: List[Dict[str, Any]] = []

        for sample, image in zip(predictions, images):
            # Build lookup tables for semantic and instance IDs
            max_seg_id = sample["panoptic_seg"][0].amax().item() + 1
            device = sample["panoptic_seg"][0].device

            weight_semantic = torch.ones(max_seg_id, device=device, dtype=torch.long) * 255
            weight_instance = torch.zeros(max_seg_id, device=device, dtype=torch.long)
            object_semantics: List[int] = []

            for obj in sample["panoptic_seg"][1]:
                if obj["isthing"]:
                    weight_semantic[obj["id"]] = 0
                    weight_instance[obj["id"]] = weight_instance.amax() + 1
                    object_semantics.append(obj["category_id"])
                else:
                    weight_semantic[obj["id"]] = obj["category_id"]

            # Instance map from segment IDs
            instance = torch.embedding(
                indices=sample["panoptic_seg"][0],
                weight=weight_instance.view(-1, 1),
            ).squeeze()

            # Confidence-thresholded semantic segmentation
            sem_raw = sample["sem_seg"]  # (C, H, W)
            max_class_scores = sem_raw.amax(dim=(1, 2), keepdim=True)
            class_threshold = (
                max_class_scores
                * self.hparams.config.SELF_TRAINING.SEMANTIC_SEGMENTATION_THRESHOLD
            )
            sem_filtered = torch.where(sem_raw > class_threshold, sem_raw, 0.0)
            sem_pseudo = sem_filtered.argmax(dim=0)
            sem_pseudo[sem_filtered.sum(dim=0) == 0] = 255

            # Build output dict
            img_tensor = image["image"].squeeze()
            if instance.amax() > 0.0:
                instance_masks = instances_to_masks(instance)
                pseudo_labels.append({
                    "image": img_tensor,
                    "sem_seg": sem_pseudo.long(),
                    "instances": Instances(
                        image_size=tuple(img_tensor.shape[1:]),
                        gt_masks=BitMasks(instance_masks),
                        gt_boxes=Boxes(get_bounding_boxes(instance)),
                        gt_classes=torch.tensor(object_semantics, device=device),
                    ),
                })
            else:
                pseudo_labels.append({
                    "image": img_tensor,
                    "sem_seg": sem_pseudo.long(),
                    "instances": Instances(
                        image_size=tuple(img_tensor.shape[1:]),
                        gt_masks=BitMasks(torch.zeros(0, *img_tensor.shape[1:]).bool()),
                        gt_boxes=Boxes(torch.zeros(0, 4).long()),
                        gt_classes=torch.zeros(0).long(),
                    ),
                })

        return pseudo_labels

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Update teacher model via exponential moving average."""
        for student_param, teacher_param in zip(
            self.model.parameters(), self.teacher_model.parameters()
        ):
            teacher_param.data.mul_(self.ema_momentum).add_(
                (1.0 - self.ema_momentum) * student_param.data
            )

    def on_train_epoch_end(self) -> None:
        """Close the Detectron2 EventStorage at epoch end."""
        if self.storage is not None:
            self.storage.__exit__(None, None, None)
            self.storage = None


# ======================================================================
# Builder function
# ======================================================================

def build_model_self_mask2former(
    config: CfgNode,
    thing_pseudo_classes: Optional[Tuple[int, ...]] = None,
    stuff_pseudo_classes: Optional[Tuple[int, ...]] = None,
    thing_classes: Set[int] = set(),
    stuff_classes: Set[int] = set(),
    copy_paste_augmentation: Optional[nn.Module] = nn.Identity(),
    photometric_augmentation: nn.Module = nn.Identity(),
    resolution_jitter_augmentation: nn.Module = nn.Identity(),
    class_names: Optional[List[str]] = None,
    classes_mask: Optional[List[bool]] = None,
    ema_momentum: float = 0.999,
) -> Mask2FormerSelfSupervisedModel:
    """Build Mask2Former for Stage-3 self-training from a Stage-2 checkpoint.

    Loads the Stage-2 checkpoint, infers class counts, creates student + teacher
    (EMA deep copy), and wraps in the Lightning module.

    Args:
        config: CUPS config with MODEL.CHECKPOINT pointing to Stage-2 ckpt.
        thing_pseudo_classes: Pseudo-label thing class IDs (optional, inferred from ckpt).
        stuff_pseudo_classes: Pseudo-label stuff class IDs (optional, inferred from ckpt).
        thing_classes: GT thing class IDs for validation PQ.
        stuff_classes: GT stuff class IDs for validation PQ.
        copy_paste_augmentation: Copy-paste augmentation module.
        photometric_augmentation: Photometric augmentation module.
        resolution_jitter_augmentation: Resolution jitter augmentation module.
        class_names: Class names for logging.
        classes_mask: Validation class mask.
        ema_momentum: EMA momentum for teacher update (default 0.999).

    Returns:
        Mask2FormerSelfSupervisedModel ready for Lightning training.
    """
    m2f_config = config.MODEL.MASK2FORMER

    # ------------------------------------------------------------------
    # 1. Load checkpoint and infer class counts
    # ------------------------------------------------------------------
    state_dict = None
    if config.MODEL.CHECKPOINT is not None:
        log.info(f"Loading Stage-2 checkpoint from {config.MODEL.CHECKPOINT}")
        checkpoint = torch.load(config.MODEL.CHECKPOINT, map_location="cpu", weights_only=False)

        if "state_dict" in checkpoint:
            raw_sd = checkpoint["state_dict"]
            # Strip Lightning's "model." prefix, skip teacher keys if present
            state_dict = {}
            for k, v in raw_sd.items():
                if k.startswith("teacher_model."):
                    continue
                if k.startswith("model."):
                    state_dict[k[len("model."):]] = v
                else:
                    state_dict[k] = v
        else:
            state_dict = checkpoint

        # Infer num_classes from class_predictor weight shape (num_classes+1, hidden)
        class_pred_key = None
        for k in state_dict:
            if "class_predictor" in k and "weight" in k:
                class_pred_key = k
                break

        if class_pred_key is not None:
            num_classes = state_dict[class_pred_key].shape[0] - 1
            log.info(f"Inferred {num_classes} classes from checkpoint key {class_pred_key}")
        else:
            assert thing_pseudo_classes is not None and stuff_pseudo_classes is not None, \
                "Cannot infer num_classes from checkpoint. Provide thing/stuff_pseudo_classes."
            num_classes = len(thing_pseudo_classes) + len(stuff_pseudo_classes)

    else:
        assert thing_pseudo_classes is not None and stuff_pseudo_classes is not None, \
            "Either MODEL.CHECKPOINT or thing/stuff_pseudo_classes must be provided."
        num_classes = len(thing_pseudo_classes) + len(stuff_pseudo_classes)

    # Determine stuff/thing split
    if thing_pseudo_classes is not None and stuff_pseudo_classes is not None:
        num_things = len(thing_pseudo_classes)
        num_stuffs = len(stuff_pseudo_classes)
    else:
        # Try to read from Lightning checkpoint hyper_parameters
        hparams = checkpoint.get("hyper_parameters", {}) if config.MODEL.CHECKPOINT else {}
        if "num_stuff_pseudo_classes" in hparams and "num_thing_pseudo_classes" in hparams:
            num_stuffs = hparams["num_stuff_pseudo_classes"]
            num_things = hparams["num_thing_pseudo_classes"]
            log.info(f"Inferred stuff/thing split from checkpoint hparams: "
                     f"{num_stuffs} stuff, {num_things} things")
        else:
            raise ValueError(
                "Cannot infer stuff/thing split from checkpoint alone. "
                "Provide thing_pseudo_classes and stuff_pseudo_classes, or use "
                "a Lightning checkpoint that contains hyper_parameters."
            )

    log.info(
        f"Building Mask2Former self-training: {num_classes} classes "
        f"({num_stuffs} stuff + {num_things} things)"
    )

    # ------------------------------------------------------------------
    # 2. Build student model
    # ------------------------------------------------------------------
    student: Mask2FormerWrapper = build_mask2former(
        num_classes=num_classes,
        num_stuff_classes=num_stuffs,
        pretrained=m2f_config.PRETRAINED,
        num_queries=m2f_config.NUM_QUERIES,
        freeze_backbone=(m2f_config.BACKBONE_LR_MULTIPLIER == 0.0),
        no_object_weight=m2f_config.NO_OBJECT_WEIGHT,
        confidence_threshold=config.MODEL.INFERENCE_CONFIDENCE_THRESHOLD,
    )

    # Load checkpoint weights
    if state_dict is not None:
        missing, unexpected = student.load_state_dict(state_dict, strict=False)
        log.info(f"Checkpoint loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            log.warning(f"Missing keys (first 10): {missing[:10]}")
        if unexpected:
            log.warning(f"Unexpected keys (first 10): {unexpected[:10]}")

    # ------------------------------------------------------------------
    # 3. Create teacher (deep copy, frozen)
    # ------------------------------------------------------------------
    teacher: Mask2FormerWrapper = copy.deepcopy(student)
    for param in teacher.parameters():
        param.requires_grad = False
    log.info(f"Teacher created (EMA momentum={ema_momentum})")

    # ------------------------------------------------------------------
    # 4. Wrap in Lightning module
    # ------------------------------------------------------------------
    model = Mask2FormerSelfSupervisedModel(
        model=student,
        teacher_model=teacher,
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
        ema_momentum=ema_momentum,
    )

    return model
