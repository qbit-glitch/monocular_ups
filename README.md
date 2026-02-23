# Unsupervised Panoptic Segmentation

Fully unsupervised panoptic segmentation on Cityscapes using overclustered CAUSE-TR semantics, depth-guided instances, and CUPS Cascade Mask R-CNN with self-training. **No ground-truth labels used anywhere in the pipeline.**

## Pipeline Overview

```
Stage 1: Pseudo-Label Generation (monocular images only)
──────────────────────────────────────────────────────
  DINOv2 ViT-B/14 features
      │
      ├─→ CAUSE-TR decoder → 27-class semantic map
      │       │
      │       └─→ K-means overclustering (k=300) → 19-class map (mIoU=60.7%)
      │
      ├─→ SPIdepth → monocular depth
      │       │
      │       └─→ Depth-gradient instance splitting
      │
      └─→ Connected components of thing classes → instance masks
              │
              └─→ Panoptic pseudo-labels (PQ=25.6, PQ_st=33.1, PQ_th=15.2)

Stage 2: Supervised Training on Pseudo-Labels
─────────────────────────────────────────────
  Cascade Mask R-CNN (DINOv2 ViT-B/14 or ResNet-50 backbone)
      │
      └─→ Train on pseudo-labels with copy-paste augmentation
              │
              └─→ PQ ≈ 23-25%

Stage 3: Self-Training with TTA Teacher
────────────────────────────────────────
  EMA teacher (TTA) → generates improved pseudo-labels online
      │
      └─→ Student retrains on teacher predictions (3 rounds)
              │
              └─→ PQ ≈ 27-29% (target: beat CUPS 27.8)
```

## Key Results

| Method | mIoU | PQ | PQ_stuff | PQ_things |
|--------|------|----|----------|-----------|
| Input pseudo-labels (overclustered k=300 + CC) | 60.7% | 25.6 | 33.1 | 15.2 |
| Stage-2 v4 (ResNet-50, step 4000/8000) | 35.3% | 22.5 | — | 12.6 |
| CUPS (Hahn et al., CVPR 2025) | — | 27.8 | — | — |

## Repository Structure

```
├── train.py                    # Stage-2: Supervised training on pseudo-labels
├── train_hybrid.py             # Stage-2 + Stage-3: Supervised + self-training
├── train_self.py               # Stage-3: Self-training only
├── val.py                      # Evaluation and inference
│
├── cups/                       # Core CUPS package
│   ├── config.py               # YACS configuration system
│   ├── augmentation.py         # Copy-paste, photometric, resolution jitter
│   ├── pl_model_pseudo.py      # PyTorch Lightning wrapper (Stage-2)
│   ├── pl_model_self.py        # PyTorch Lightning wrapper (Stage-3)
│   ├── model/
│   │   ├── model.py            # Panoptic Cascade Mask R-CNN (ResNet-50)
│   │   ├── model_vitb.py       # DINOv2 ViT-B/14 variant
│   │   ├── backbone_dinov2_vit.py  # DINOv2 Detectron2 backbone wrapper
│   │   └── modeling/           # Custom Detectron2 heads
│   ├── data/
│   │   ├── cityscapes.py       # Cityscapes dataset loaders
│   │   ├── pseudo_label_dataset.py  # Pseudo-label dataset (with spatial alignment fix)
│   │   └── utils.py            # Data utilities
│   └── metrics/
│       └── panoptic_quality.py # PQ metric with Hungarian matching
│
├── pseudo_labels/              # Stage-1: Pseudo-label generation
│   ├── generate_semantic_pseudolabels_cause.py   # CAUSE-TR 27-class semantics
│   ├── generate_overclustered_semantics.py       # K-means overclustering (k=300)
│   ├── overclustering_cause.py                   # Overclustering analysis
│   ├── extract_dinov2_features.py                # DINOv2 feature extraction
│   ├── generate_depth_spidepth.py                # SPIdepth monocular depth
│   ├── generate_depth_guided_instances.py        # Depth-gradient instances
│   ├── generate_panoptic_pseudolabels.py         # Combine semantic + instances
│   ├── convert_to_cups_format.py                 # Convert to CUPS training format
│   ├── classify_stuff_things.py                  # Stuff/things classification
│   ├── remap_cause27_to_trainid.py               # 27-class → 19-class remapping
│   └── evaluation/
│       ├── evaluate_pseudolabels.py              # Full panoptic evaluation
│       ├── evaluate_semantic_pseudolabels.py     # Semantic-only evaluation
│       ├── evaluate_cascade_pseudolabels.py      # CUPS-format evaluation
│       └── diagnose_cups_pseudolabels.py         # Diagnostic validation
│
├── configs/                    # Training configurations
│   ├── train_cityscapes.yaml               # ResNet-50 baseline
│   ├── train_cityscapes_resnet50_v4.yaml   # ResNet-50 (fixed alignment)
│   ├── train_cityscapes_vitb.yaml          # ViT-B/14 baseline
│   ├── train_cityscapes_vitb_v3.yaml       # ViT-B/14 (overclustered)
│   ├── train_cityscapes_vitb_v4.yaml       # ViT-B/14 (fixed alignment)
│   ├── train_cityscapes_vitb_v5.yaml       # ViT-B/14 (self-training + grad accum)
│   └── val_cityscapes.yaml                 # Validation config
│
├── scripts/                    # Launch scripts
│   ├── run_vitb_stage2.sh
│   ├── run_vitb_v3_stage2.sh
│   ├── run_vitb_v4_stage2.sh
│   ├── run_vitb_v5_stage2.sh
│   ├── run_resnet50_v4_stage2.sh
│   └── cups_inference_val.py               # Inference + evaluation
│
├── reports/                    # Technical reports
│   ├── stage_1_report.md                   # Stage-1 pseudo-label pipeline
│   ├── cause_tr_refinement.md              # Overclustering discovery
│   ├── overclustered_spidepth_sweep.md     # Instance method comparison
│   └── stage2_spatial_alignment_fix.md     # Spatial misalignment bug fix
│
└── requirements.txt
```

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

External dependencies (not included):
- [CAUSE-TR](https://github.com/xxx/cause) — semantic pseudo-label generation
- [SPIdepth](https://github.com/xxx/spidepth) — monocular depth estimation
- Cityscapes dataset (register at https://www.cityscapes-dataset.com/)

### Stage 1: Generate Pseudo-Labels

```bash
# 1. Extract DINOv2 features
python pseudo_labels/extract_dinov2_features.py \
    --cityscapes_root /path/to/cityscapes --split train

# 2. Generate CAUSE-TR semantic labels (requires CAUSE model)
python pseudo_labels/generate_semantic_pseudolabels_cause.py \
    --cityscapes_root /path/to/cityscapes --split train

# 3. Overclustering (k=300) to recover zero-IoU classes
python pseudo_labels/generate_overclustered_semantics.py \
    --cityscapes_root /path/to/cityscapes --k 300 --split train

# 4. Generate depth maps
python pseudo_labels/generate_depth_spidepth.py \
    --cityscapes_root /path/to/cityscapes --split train

# 5. Generate depth-guided instances
python pseudo_labels/generate_depth_guided_instances.py \
    --cityscapes_root /path/to/cityscapes --split train

# 6. Convert to CUPS format
python pseudo_labels/convert_to_cups_format.py \
    --cityscapes_root /path/to/cityscapes \
    --output_dir /path/to/cityscapes/cups_pseudo_labels/ \
    --trainid_input
```

### Stage 2: Train Cascade Mask R-CNN

```bash
# ViT-B/14 backbone with gradient accumulation (effective batch=16)
python train.py \
    --experiment_config_file configs/train_cityscapes_vitb_v5.yaml \
    --disable_wandb
```

### Stage 2+3: Train with Self-Training

```bash
# Hybrid training: Phase 1 (4000 steps) + Phase 2 self-training (3 rounds)
python train_hybrid.py \
    --experiment_config_file configs/train_cityscapes_vitb_v5.yaml \
    --disable_wandb
```

### Evaluation

```bash
python val.py \
    --experiment_config_file configs/val_cityscapes.yaml \
    --checkpoint /path/to/checkpoint.ckpt
```

## Key Technical Contributions

1. **Overclustered CAUSE-TR semantics**: K-means overclustering (k=300) on CAUSE's 90-dim Segment_TR features recovers 7 zero-IoU classes (fence, pole, traffic light/sign, rider, train, motorcycle), pushing mIoU from 40.4% to 60.7%.

2. **Spatial alignment fix**: Identified and corrected a spatial misalignment bug in CUPS's `PseudoLabelDataset` where pseudo-labels were not scaled by `ground_truth_scale` before `CenterCrop`, causing image-label spatial displacement of up to 384 pixels. Fix improved Stage-2 PQ from 8-11% to 22.5%.

3. **Monocular-only pipeline**: Unlike CUPS (which requires stereo video sequences for optical flow-based instance segmentation), our pipeline uses only monocular images + self-supervised DINOv2 features + monocular depth.

## Backbone Options

| Config | Backbone | Notes |
|--------|----------|-------|
| `train_cityscapes.yaml` | DINOv2-ResNet-50 | Original CUPS backbone |
| `train_cityscapes_vitb_v5.yaml` | DINOv2 ViT-B/14 (frozen) | Higher capacity, gradient accumulation |

## Citation

Based on CUPS (Hahn et al., CVPR 2025) with modifications for unsupervised pseudo-label training.

## License

See individual file headers for attribution. CUPS components are subject to the original CUPS license.
