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
│   │   ├── modeling/           # Custom Detectron2 heads
│   │   ├── Base-RCNN-FPN.yaml            # Detectron2 base config
│   │   ├── Base-Panoptic-FPN.yaml        # Panoptic FPN base config
│   │   ├── Cascade-Mask-R-CNN.yaml       # Instance-only config
│   │   └── Panoptic-Cascade-Mask-R-CNN.yaml  # Full panoptic config
│   ├── data/
│   │   ├── cityscapes.py       # Cityscapes dataset loaders
│   │   ├── pseudo_label_dataset.py  # Pseudo-label dataset (with spatial alignment fix)
│   │   └── utils.py            # Data utilities
│   └── metrics/
│       └── panoptic_quality.py # PQ metric with Hungarian matching
│
├── pseudo_labels/              # Stage-1: Pseudo-label generation
│   ├── generate_semantic_pseudolabels_cause.py   # CAUSE-TR 27-class semantics
│   ├── generate_overclustered_semantics.py       # K-means overclustering (k=50/300)
│   ├── overclustering_cause.py                   # Overclustering analysis
│   ├── extract_dinov2_features.py                # DINOv2 ViT-B/14 feature extraction
│   ├── extract_dinov3_features.py                # DINOv3 feature extraction (for DINOSAUR)
│   ├── generate_depth_spidepth.py                # SPIdepth monocular depth
│   ├── generate_depth_guided_instances.py        # Depth-gradient instances
│   ├── generate_dinosaur_instances.py            # DINOSAUR slot attention instances
│   ├── train_dinosaur.py                         # DINOSAUR slot attention training
│   ├── generate_panoptic_pseudolabels.py         # Combine semantic + instances
│   ├── convert_to_cups_format.py                 # Convert to CUPS training format
│   ├── classify_stuff_things.py                  # Stuff/things classification
│   ├── remap_cause27_to_trainid.py               # 27-class → 19-class remapping
│   ├── sweep_k50_spidepth.py                     # Parameter sweep for k=50 + depth splitting
│   ├── refine_net.py                             # CSCMRefineNet model (DINOv2+depth → refined semantics)
│   ├── joint_refine_net.py                       # JointRefineNet (semantic + boundary + embedding)
│   ├── train_refine_net.py                       # CSCMRefineNet training
│   ├── train_joint_refine_net.py                 # JointRefineNet training
│   ├── generate_refined_semantics.py             # RefineNet inference
│   ├── losses/
│   │   └── instance_embedding_loss.py            # Discriminative loss
│   └── evaluation/
│       ├── evaluate_pseudolabels.py              # Full panoptic evaluation
│       ├── evaluate_semantic_pseudolabels.py     # Semantic-only evaluation
│       ├── evaluate_cascade_pseudolabels.py      # CUPS-format evaluation
│       ├── evaluate_k50_pseudolabels.py          # k=50 overcluster evaluation
│       └── diagnose_cups_pseudolabels.py         # Diagnostic validation
│
├── configs/                    # Training configurations
│   ├── train_cityscapes.yaml               # ResNet-50 baseline
│   ├── train_cityscapes_resnet50_v4.yaml   # ResNet-50 (fixed alignment)
│   ├── train_cityscapes_resnet50_k50.yaml  # ResNet-50 (k=50 raw overclusters)
│   ├── train_cityscapes_vitb.yaml          # ViT-B/14 baseline
│   ├── train_cityscapes_vitb_v3.yaml       # ViT-B/14 (overclustered)
│   ├── train_cityscapes_vitb_v4.yaml       # ViT-B/14 (fixed alignment)
│   ├── train_cityscapes_vitb_v5.yaml       # ViT-B/14 (self-training + grad accum)
│   ├── train_self_cityscapes.yaml          # Stage-3 self-training config
│   ├── train_self_cityscapes_vitb_local.yaml  # Stage-3 ViT-B local config
│   ├── train_hybrid_local.yaml             # Stage-2+3 hybrid local config
│   └── val_cityscapes.yaml                 # Validation config
│
├── scripts/                    # Launch scripts
│   ├── run_vitb_stage2.sh
│   ├── run_vitb_v3_stage2.sh
│   ├── run_vitb_v4_stage2.sh
│   ├── run_vitb_v5_stage2.sh
│   ├── run_resnet50_v4_stage2.sh
│   ├── run_resnet50_k50_stage2.sh          # k=50 raw overclusters Stage-2
│   └── cups_inference_val.py               # Inference + evaluation
│
├── refs/                       # Reference codebases (source only, checkpoints gitignored)
│   ├── cause/                  # CAUSE (Pattern Recognition 2024)
│   │   ├── models/             # DINOv2 ViT backbone
│   │   └── modules/            # Segment TR decoder + modularity codebook
│   └── spidepth/               # SPIdepth self-supervised monocular depth
│       ├── SQLdepth.py, layers.py
│       └── networks/           # Encoder-decoder architecture (12 files)
│
├── reports/                    # Technical reports
│   ├── stage_1_report.md                   # Stage-1 pseudo-label pipeline
│   ├── cause_tr_refinement.md              # Overclustering discovery
│   ├── overclustered_spidepth_sweep.md     # Instance method comparison
│   └── stage2_spatial_alignment_fix.md     # Spatial misalignment bug fix
│
├── requirements.txt
└── requirements_cups.txt
```

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Download model checkpoints (not tracked by git):
- **CAUSE-TR**: Place in `refs/cause/CAUSE/` (DINOv2 ViT-B/14 + TR decoder weights)
- **SPIdepth**: Place in `refs/spidepth/checkpoints/` (monocular depth weights)
- **DINO ResNet-50**: Place in `cups/model/backbone_checkpoints/dino_RN50_full_pretrain_d2_format.pkl`
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

#### Alternative: DINOSAUR Slot Attention Instances

```bash
# 1. Extract DINOv3 features (required for DINOSAUR)
python pseudo_labels/extract_dinov3_features.py \
    --cityscapes_root /path/to/cityscapes --split train

# 2. Train DINOSAUR slot attention model
python pseudo_labels/train_dinosaur.py \
    --cityscapes_root /path/to/cityscapes --num_slots 30

# 3. Generate DINOSAUR instance masks
python pseudo_labels/generate_dinosaur_instances.py \
    --cityscapes_root /path/to/cityscapes --checkpoint /path/to/dinosaur.pth
```

#### Alternative: Semantic Refinement with CSCMRefineNet

```bash
# Train RefineNet to refine CAUSE semantics using DINOv2 features + depth
python pseudo_labels/train_refine_net.py \
    --cityscapes_root /path/to/cityscapes

# Generate refined semantic pseudo-labels
python pseudo_labels/generate_refined_semantics.py \
    --cityscapes_root /path/to/cityscapes --checkpoint /path/to/refine_net.pth
```

#### k=50 Raw Overcluster Pipeline

```bash
# Generate raw overclusters (preserving all 50 cluster IDs)
python pseudo_labels/generate_overclustered_semantics.py \
    --cityscapes_root /path/to/cityscapes --k 50 --raw --split train

# Run parameter sweep for depth-guided splitting
python pseudo_labels/sweep_k50_spidepth.py

# Convert best config to CUPS format
python pseudo_labels/convert_to_cups_format.py \
    --cityscapes_root /path/to/cityscapes \
    --output_dir /path/to/cityscapes/cups_pseudo_labels_k50/ \
    --num_semantic_classes 50
```

### Stage 2: Train Cascade Mask R-CNN

```bash
# ViT-B/14 backbone with gradient accumulation (effective batch=16)
python train.py \
    --experiment_config_file configs/train_cityscapes_vitb_v5.yaml \
    --disable_wandb

# Override dataset path and training hyperparameters via CLI:
python train.py \
    --experiment_config_file configs/train_cityscapes_resnet50_k50.yaml \
    --data_root /your/cityscapes/path/ \
    --pseudo_root /your/cityscapes/path/cups_pseudo_labels_k50/ \
    --batch_size 2 --lr 0.0001 --steps 8000 \
    --num_gpus 2 --num_workers 4 \
    --log_path ./experiments --run_name my_run \
    --disable_wandb
```

All CLI flags (`--data_root`, `--pseudo_root`, `--batch_size`, `--lr`, `--steps`, `--num_gpus`, `--num_workers`, `--log_path`, `--run_name`, `--val_every`, `--accumulate_grad`) override the YAML config values.

### Stage 2+3: Train with Self-Training

```bash
# Hybrid training: Phase 1 (4000 steps) + Phase 2 self-training (3 rounds)
python train_hybrid.py \
    --experiment_config_file configs/train_cityscapes_vitb_v5.yaml \
    --data_root /your/cityscapes/path/ \
    --batch_size 1 --num_gpus 2 \
    --disable_wandb
```

### Evaluation

```bash
python val.py \
    --experiment_config_file configs/val_cityscapes.yaml \
    --checkpoint /path/to/checkpoint.ckpt
```

### Parameter Sweep (Custom Grid)

```bash
# Default sweep grid (31 configs + CC baseline)
python pseudo_labels/sweep_k50_spidepth.py \
    --cityscapes_root /your/cityscapes/path/

# Custom sweep: specify your own grad_thresholds and min_areas
python pseudo_labels/sweep_k50_spidepth.py \
    --cityscapes_root /your/cityscapes/path/ \
    --grad_thresholds 0.1 0.3 0.5 0.7 \
    --min_areas 200 500 800 1000 \
    --semantic_subdir pseudo_semantic_raw_k50 \
    --depth_subdir depth_spidepth

# Quick test with limited images
python pseudo_labels/sweep_k50_spidepth.py \
    --cityscapes_root /your/cityscapes/path/ \
    --max_images 50 --no_cc_baseline
```

## Key Technical Contributions

1. **Overclustered CAUSE-TR semantics**: K-means overclustering (k=300) on CAUSE's 90-dim Segment_TR features recovers 7 zero-IoU classes (fence, pole, traffic light/sign, rider, train, motorcycle), pushing mIoU from 40.4% to 60.7%.

2. **Spatial alignment fix**: Identified and corrected a spatial misalignment bug in CUPS's `PseudoLabelDataset` where pseudo-labels were not scaled by `ground_truth_scale` before `CenterCrop`, causing image-label spatial displacement of up to 384 pixels. Fix improved Stage-2 PQ from 8-11% to 22.5%.

3. **Monocular-only pipeline**: Unlike CUPS (which requires stereo video sequences for optical flow-based instance segmentation), our pipeline uses only monocular images + self-supervised DINOv2 features + monocular depth.

## Backbone Options

| Config | Backbone | Notes |
|--------|----------|-------|
| `train_cityscapes.yaml` | DINOv2-ResNet-50 | Original CUPS backbone |
| `train_cityscapes_resnet50_k50.yaml` | DINOv2-ResNet-50 | k=50 raw overclusters (50 pseudo-classes) |
| `train_cityscapes_vitb_v5.yaml` | DINOv2 ViT-B/14 (frozen) | Higher capacity, gradient accumulation |
| `train_self_cityscapes.yaml` | ResNet-50 | Stage-3 self-training only |
| `train_hybrid_local.yaml` | ResNet-50 | Stage-2+3 hybrid training |

## Citation

Based on CUPS (Hahn et al., CVPR 2025) with modifications for unsupervised pseudo-label training.

## License

See individual file headers for attribution. CUPS components are subject to the original CUPS license.
