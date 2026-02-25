# Design: Mask2Former (Swin-L) Stage-2 Training via CUPS Pipeline

**Date**: 2026-02-25
**Goal**: Fine-tune pre-trained Mask2Former Swin-L (COCO panoptic) on k=100 overclustered pseudo-labels, integrated into the CUPS PyTorch Lightning training loop.
**Target**: PQ > 27.8 on Cityscapes val (beating CUPS CVPR 2025 SOTA)
**Compute**: RTX A6000 Pro 48GB (Anydesk machine)

## Architecture

```
Input Image (B, 3, H, W)
    |
    v
Mask2Former (Swin-L, pre-trained COCO panoptic, ~216M params)
  - Swin-L backbone (197M) -- frozen or 0.1x LR
  - MSDeformAttn Pixel Decoder (10M) -- fine-tuned
  - Masked Attention Transformer Decoder (9M, 200 queries) -- fine-tuned
  - class_embed: 256 -> k+1 (REINITIALIZED from scratch)
  - mask_embed: 256 -> 256 (kept from COCO)
    |
    v (training)
  Hungarian matching -> loss_ce + loss_mask + loss_dice (x10 aux layers)
    |
    v (inference)
  post_process_panoptic_segmentation() -> panoptic_seg + segments_info
    |
    v
CUPS Lightning Wrapper (Mask2FormerUnsupervisedModel)
  - Reuses: PQ validation, copy-paste aug, photometric aug, resolution jitter
  - Adapts: training_step, validation_step for Mask2Former I/O format
```

## What Changes vs. What's Reused

| Component | Action | Details |
|-----------|--------|---------|
| `cups/config.py` | Extend | Add `BACKBONE_TYPE: "mask2former_swinl"` + M2F hyperparams |
| `cups/pl_model_mask2former.py` | New | Lightning module for Mask2Former training/val |
| `cups/model/model_mask2former.py` | New | HF Mask2Former wrapper + format adapter |
| `train.py` | Extend | Route to M2F builder when backbone_type matches |
| CUPS augmentations | Reuse | Copy-paste, photometric, resolution jitter |
| CUPS PQ metric | Reuse | `PanopticQualitySemanticMatching` on panoptic maps |
| `PseudoLabelDataset` | Reuse | Same pseudo-label format |
| CUPS Cascade R-CNN heads | Not used | Replaced by Mask2Former decoder |

## Data Format Adapter

CUPS PseudoLabelDataset -> adapter -> Mask2Former input:
1. Extract stuff segments from sem_seg (one mask per unique stuff class)
2. Instance masks already have per-instance binary masks + class IDs
3. Concatenate stuff + instance masks -> mask_labels (M, H, W)
4. Concatenate class IDs -> class_labels (M,)
5. Normalize image (CUPS [0,255] BGR -> Mask2Former normalized RGB)

## Training Recipe

| Param | Value |
|-------|-------|
| Checkpoint | `facebook/mask2former-swin-large-coco-panoptic` |
| Num classes | k (raw clusters, e.g. 100) |
| Backbone LR | 0 (frozen) |
| Decoder LR | 1e-4 (AdamW) |
| Weight decay | 0.05 |
| Batch size | 2 x 8 accum = 16 effective |
| Precision | bf16-mixed |
| Steps | 4000 |
| Grad clip | max_norm=0.01 |
| Queries | 200 |
| Loss weights | class=2.0, mask=5.0, dice=5.0, no_object=0.1 |
| Copy-paste | Yes, startup=500 |
| Resolution jitter | 384-704 (11 scales) |
| Validation | Every 200 steps |

## New Files

```
cups/model/model_mask2former.py              # HF Mask2Former wrapper
cups/pl_model_mask2former.py                 # Lightning module
refs/cups/configs/train_cityscapes_mask2former_swinl_k100.yaml
scripts/run_stage2_mask2former_k100.sh
```
