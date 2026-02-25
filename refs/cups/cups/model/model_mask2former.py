"""Mask2Former wrapper for CUPS unsupervised panoptic segmentation.

Loads a HuggingFace pre-trained Mask2Former and wraps it so that:
- Training: accepts CUPS Detectron2-format dicts, returns a loss dict.
- Inference: accepts image-only dicts, returns CUPS panoptic output format.

This allows Mask2Former to be used as a drop-in replacement for
Cascade Mask R-CNN within the existing CUPS Lightning training loop.

Class ID conventions:
    CUPS Detectron2 format (dataset output):
        sem_seg: 0=thing_marker, 1..N_stuff=stuff_classes, 255=ignore
        instances.gt_classes: 0..N_things-1 (index into thing_pseudo_classes)

    Mask2Former unified space:
        0..N_stuff-1 = stuff classes
        N_stuff..N_total-1 = thing classes

    CUPS panoptic output (what prediction_to_standard_format expects):
        stuff segments: category_id = 1-based (1..N_stuff)
        thing segments: category_id = 0-based index into thing_pseudo_classes
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# ImageNet normalization used by Swin / Mask2Former
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


class Mask2FormerWrapper(nn.Module):
    """CUPS-compatible wrapper around HuggingFace Mask2FormerForUniversalSegmentation.

    In training mode:
        Input: List[Dict] with keys 'image', 'sem_seg', 'instances' (Detectron2 format).
        Output: Dict[str, Tensor] of losses.

    In eval mode:
        Input: List[Dict] with key 'image'.
        Output: List[Dict] with keys 'panoptic_seg' (Tuple[Tensor, List[Dict]]),
                'sem_seg' (Tensor).
    """

    def __init__(
        self,
        num_classes: int,
        num_stuff_classes: int,
        pretrained: str = "facebook/mask2former-swin-tiny-coco-panoptic",
        num_queries: int = 100,
        freeze_backbone: bool = True,
        no_object_weight: float = 0.1,
        confidence_threshold: float = 0.5,
        overlap_thresh: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_thing_classes = num_classes - num_stuff_classes
        self.confidence_threshold = confidence_threshold
        self.overlap_thresh = overlap_thresh

        # Build set of M2F class IDs for stuff (0..N_stuff-1) and things (N_stuff..N_total-1)
        self.m2f_stuff_ids = set(range(num_stuff_classes))
        self.m2f_thing_ids = set(range(num_stuff_classes, num_classes))

        # Load pre-trained model, reinitialize class head for our num_classes
        log.info(f"Loading Mask2Former from {pretrained} with {num_classes} classes...")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pretrained,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        log.info(f"Mask2Former loaded. Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")

        # Processor for post-processing (inference only)
        self.processor = Mask2FormerImageProcessor(
            ignore_index=255,
            reduce_labels=False,
            do_resize=False,
            do_normalize=False,
        )

        # Set no-object weight in the loss
        self.model.config.no_object_weight = no_object_weight

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()

        # Register ImageNet normalization buffers
        self.register_buffer("pixel_mean", IMAGENET_MEAN.view(3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", IMAGENET_STD.view(3, 1, 1), persistent=False)

    def _freeze_backbone(self) -> None:
        """Freeze the Swin backbone (pixel_level_module.encoder)."""
        frozen = 0
        for name, param in self.model.named_parameters():
            if "pixel_level_module.encoder" in name:
                param.requires_grad = False
                frozen += 1
        log.info(f"Frozen {frozen} backbone parameters.")

    def _normalize_image(self, image: Tensor) -> Tensor:
        """Normalize a CUPS image tensor to Mask2Former input format.

        CUPS format: (C, H, W), float32, values [0, 1], RGB order.
        Mask2Former format: (C, H, W), float32, ImageNet-normalized, RGB order.
        """
        return (image - self.pixel_mean) / self.pixel_std

    def _cups_to_mask2former_labels(
        self,
        batched_inputs: List[Dict[str, Any]],
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """Convert CUPS Detectron2-format batch to Mask2Former training inputs.

        CUPS sem_seg convention:
            0 = thing marker (skip — things come from instances)
            1..N_stuff = stuff class (1-based index)
            255 = ignore

        CUPS instances.gt_classes convention:
            0..N_things-1 = thing class (0-based index)

        Mask2Former class convention (unified):
            0..N_stuff-1 = stuff classes
            N_stuff..N_total-1 = thing classes

        Returns:
            pixel_values: (B, 3, H, W) normalized images
            mask_labels: list of (M_i, H, W) binary masks per sample
            class_labels: list of (M_i,) class IDs per sample (in M2F unified space)
        """
        pixel_values_list = []
        mask_labels_list = []
        class_labels_list = []

        for sample in batched_inputs:
            image = sample["image"]  # (C, H, W), [0, 1], RGB
            sem_seg = sample["sem_seg"]  # (H, W), long
            instances = sample["instances"]  # Detectron2 Instances

            H, W = image.shape[1], image.shape[2]

            # Normalize image
            pixel_values_list.append(self._normalize_image(image))

            masks = []
            classes = []

            # --- Extract stuff segments from sem_seg ---
            # sem_seg values: 0=thing_marker, 1..N_stuff=stuff, 255=ignore
            unique_classes = sem_seg.unique()
            for cls_id in unique_classes:
                cls_id_item = cls_id.item()
                if cls_id_item == 0 or cls_id_item == 255:
                    continue  # skip thing marker and ignore
                # This is a stuff class (1-based) -> M2F class (0-based)
                mask = (sem_seg == cls_id)
                if mask.sum() > 0:
                    masks.append(mask.float())
                    classes.append(cls_id_item - 1)  # 1-based -> 0-based M2F stuff ID

            # --- Extract thing instances ---
            if len(instances) > 0:
                instance_masks = instances.gt_masks.tensor  # (N, H, W) bool
                instance_classes = instances.gt_classes  # (N,) long, 0..N_things-1
                for i in range(len(instances)):
                    mask = instance_masks[i].float()
                    if mask.sum() > 0:
                        masks.append(mask)
                        # 0-based thing index -> M2F unified space (offset by N_stuff)
                        classes.append(instance_classes[i].item() + self.num_stuff_classes)

            # Stack into tensors
            if len(masks) > 0:
                mask_labels = torch.stack(masks, dim=0)  # (M, H, W)
                class_labels = torch.tensor(classes, dtype=torch.long, device=image.device)
            else:
                # No valid segments — provide empty tensors
                mask_labels = torch.zeros(0, H, W, dtype=torch.float32, device=image.device)
                class_labels = torch.zeros(0, dtype=torch.long, device=image.device)

            mask_labels_list.append(mask_labels)
            class_labels_list.append(class_labels)

        # Stack pixel values into batch
        pixel_values = torch.stack(pixel_values_list, dim=0)

        return pixel_values, mask_labels_list, class_labels_list

    def _panoptic_postprocess(
        self,
        outputs: Any,
        target_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Any]]:
        """Convert Mask2Former outputs to CUPS panoptic format.

        Remaps M2F unified class IDs back to CUPS conventions:
            M2F 0..N_stuff-1  -> CUPS stuff: category_id = label_id + 1 (1-based)
            M2F N_stuff..N-1  -> CUPS thing: category_id = label_id - N_stuff (0-based)

        Returns list of dicts, each with:
            'panoptic_seg': Tuple[Tensor(H,W), List[Dict]] (Detectron2 format)
            'sem_seg': Tensor(num_classes, H, W) class logits
        """
        # Use HuggingFace post-processing for panoptic segmentation
        # Fuse stuff classes so multiple segments of same stuff merge into one
        results = self.processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=target_sizes,
            threshold=self.confidence_threshold,
            overlap_mask_area_threshold=self.overlap_thresh,
            label_ids_to_fuse=self.m2f_stuff_ids,
        )

        processed = []
        for idx, result in enumerate(results):
            hf_seg = result["segmentation"].long()  # (H, W) tensor, segment IDs
            hf_segments = result["segments_info"]  # list of dicts

            # --- Normalize segment IDs to CUPS Detectron2 convention ---
            # HuggingFace post_process_panoptic_segmentation returns:
            #   Newer transformers (>=4.40): -1=background, 0-based segment IDs
            #   Older transformers: 0=background, 1-based segment IDs
            # CUPS expects: 0=background, positive (1-based) segment IDs
            if hf_seg.numel() > 0 and hf_seg.min().item() < 0:
                # Newer HF: shift -1→0 (bg), 0→1, 1→2, ...
                hf_seg = hf_seg + 1
                for seg in hf_segments:
                    seg["id"] = seg["id"] + 1

            # Handle edge case: no segments above threshold (random init class head)
            # All pixels are background (0). Provide trivial empty output.
            H, W = target_sizes[idx]
            if len(hf_segments) == 0:
                hf_seg = torch.zeros(H, W, dtype=torch.long, device=hf_seg.device)

            # Convert HF segments_info to Detectron2 format with correct category_id
            d2_segments = []
            for seg in hf_segments:
                label_id = seg["label_id"]
                is_thing = label_id in self.m2f_thing_ids

                if is_thing:
                    # M2F thing class -> CUPS 0-based thing index
                    category_id = label_id - self.num_stuff_classes
                else:
                    # M2F stuff class -> CUPS 1-based stuff index
                    category_id = label_id + 1

                d2_segments.append({
                    "id": seg["id"],
                    "isthing": is_thing,
                    "category_id": category_id,
                    "score": seg.get("score", 1.0),
                    "area": (hf_seg == seg["id"]).sum().item(),
                })

            # Build semantic logits from mask queries
            # outputs.class_queries_logits: (B, num_queries, num_classes+1)
            # outputs.masks_queries_logits: (B, num_queries, H/4, W/4)
            class_logits = outputs.class_queries_logits[idx]  # (Q, C+1)
            mask_logits = outputs.masks_queries_logits[idx]  # (Q, H/4, W/4)

            # Upsample masks to target size
            H, W = target_sizes[idx]
            mask_logits_up = F.interpolate(
                mask_logits.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(0)  # (Q, H, W)

            # Compute per-class semantic logits: sum mask_prob * class_prob over queries
            mask_probs = mask_logits_up.sigmoid()  # (Q, H, W)
            class_probs = class_logits[:, :-1].softmax(dim=-1)  # (Q, C) — exclude no-object
            sem_seg = torch.einsum("qc,qhw->chw", class_probs, mask_probs)  # (C, H, W)

            processed.append({
                "panoptic_seg": (hf_seg, d2_segments),
                "sem_seg": sem_seg,
            })

        return processed

    def forward(
        self, batched_inputs: List[Dict[str, Any]]
    ) -> Any:
        """Forward pass.

        Training: returns Dict[str, Tensor] of losses.
        Inference: returns List[Dict] with 'panoptic_seg' and 'sem_seg'.
        """
        if self.training:
            return self._forward_train(batched_inputs)
        else:
            return self._forward_inference(batched_inputs)

    def _forward_train(self, batched_inputs: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        """Training forward: convert CUPS format -> HF format, compute losses."""
        pixel_values, mask_labels, class_labels = self._cups_to_mask2former_labels(batched_inputs)

        outputs = self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

        # HF Mask2FormerForUniversalSegmentationOutput does not expose per-component
        # losses (loss_ce, loss_mask, loss_dice per aux layer). The total loss includes
        # all components with their default weights (class=2.0, mask=5.0, dice=5.0).
        return {"loss_mask2former": outputs.loss}

    def _forward_inference(self, batched_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Inference forward: predict panoptic segmentation in CUPS format."""
        # Normalize images
        pixel_values_list = []
        target_sizes = []
        for sample in batched_inputs:
            image = sample["image"]  # (C, H, W)
            pixel_values_list.append(self._normalize_image(image))
            target_sizes.append((image.shape[1], image.shape[2]))

        pixel_values = torch.stack(pixel_values_list, dim=0)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)

        return self._panoptic_postprocess(outputs, target_sizes)


def build_mask2former(
    num_classes: int,
    num_stuff_classes: int,
    pretrained: str = "facebook/mask2former-swin-tiny-coco-panoptic",
    num_queries: int = 100,
    freeze_backbone: bool = True,
    no_object_weight: float = 0.1,
    confidence_threshold: float = 0.5,
) -> Mask2FormerWrapper:
    """Build a Mask2Former model for CUPS training.

    Args:
        num_classes: Total number of pseudo-label classes (stuff + things).
        num_stuff_classes: Number of stuff classes (for ID mapping).
        pretrained: HuggingFace model ID.
        num_queries: Number of mask queries (100 for Swin-T, 200 for Swin-L).
        freeze_backbone: If True, freeze Swin backbone.
        no_object_weight: Weight for the no-object class in CE loss.
        confidence_threshold: Score threshold for inference.

    Returns:
        CUPS-compatible Mask2Former wrapper.
    """
    return Mask2FormerWrapper(
        num_classes=num_classes,
        num_stuff_classes=num_stuff_classes,
        pretrained=pretrained,
        num_queries=num_queries,
        freeze_backbone=freeze_backbone,
        no_object_weight=no_object_weight,
        confidence_threshold=confidence_threshold,
    )
