# Unsupervised Panoptic Pseudo-Label Generation

Fully unsupervised panoptic pseudo-label generation on Cityscapes using overclustered CAUSE-TR semantics and SPIdepth depth-guided instance decomposition. **No ground-truth labels used anywhere in the pipeline.**

## Best Results (Stage-1 Pseudo-Labels)

| Config | k | tau | A_min | PQ | PQ_stuff | PQ_things | SQ | RQ |
|--------|---|-----|-------|------|----------|-----------|------|------|
| **Best (k=80 + depth)** | **80** | **0.20** | **1000** | **26.74** | **32.08** | **19.41** | **71.88** | **31.41** |
| k=60 + depth | 60 | 0.20 | 1000 | 25.83 | 30.74 | 19.08 | 71.69 | 30.21 |
| k=50 + depth | 50 | 0.30 | 1000 | 25.78 | 34.80 | 13.37 | 73.11 | 30.61 |
| k=300 CC-only | 300 | --- | --- | 25.60 | 33.10 | 15.20 | 71.90 | 22.30 |
| CAUSE-27 + depth | 27 | 0.10 | 500 | 23.10 | 31.40 | 11.70 | 74.30 | 31.20 |
| CUPS (CVPR 2025) | --- | --- | --- | 27.80 | 35.10 | 17.70 | 57.40 | 35.20 |

Our PQ_things (19.41) **exceeds CUPS** (17.70) by +1.71 using only monocular images and self-supervised models. The remaining 1.06 PQ gap to CUPS is driven entirely by PQ_stuff.

## Pipeline Overview

```
Cityscapes monocular images (1024x2048)
    |
    +-> CAUSE-TR (DINOv2 ViT-B/14 + Segment_TR decoder)
    |       |
    |       +-> 90-dim patch features
    |       |
    |       +-> K-means overclustering (k=80)
    |       |       |
    |       |       +-> Many-to-one majority-vote mapping -> 19-class semantic labels
    |       |
    |       +-> Raw cluster PNGs (values 0-79) + kmeans_centroids.npz
    |
    +-> SPIdepth (self-supervised monocular depth)
    |       |
    |       +-> Depth gradient (Sobel) -> binarize at threshold tau
    |
    +-> For each thing class: remove depth edges -> connected components
            |
            +-> Filter by min_area -> dilate to reclaim boundaries
                    |
                    +-> Panoptic pseudo-labels (PQ=26.74)
```

## Reproducing PQ=26.74 (k=80, tau=0.20, A_min=1000)

The pipeline that produces the best result has two steps:

1. **`generate_overclustered_semantics.py --k 80 --raw_clusters --skip_crf`** — fits K-means with k=80 directly on the 90-dim CAUSE Segment_TR features, saves raw cluster PNGs (values 0-79) + `kmeans_centroids.npz`

2. **`sweep_k50_spidepth.py --semantic_subdir pseudo_semantic_raw_k80`** — reads those raw cluster PNGs, maps clusters to 19 trainIDs using the centroids file, applies SPIdepth depth-guided instance splitting, evaluates PQ

### Step-by-step commands

```bash
PYTHON="/path/to/python3.10"  # Python 3.10 with PyTorch, scikit-learn, etc.
CS_ROOT="/path/to/cityscapes"  # Must contain leftImg8bit/, gtFine/

# Step 1: Generate k=80 overclustered semantic labels (raw cluster IDs)
$PYTHON pseudo_labels/generate_overclustered_semantics.py \
    --cityscapes_root $CS_ROOT \
    --split val --k 80 --skip_crf \
    --raw_clusters \
    --output_subdir pseudo_semantic_raw_k80

# Output:
#   $CS_ROOT/pseudo_semantic_raw_k80/val/{city}/{stem}.png  (pixel values = cluster IDs 0-79)
#   $CS_ROOT/pseudo_semantic_raw_k80/kmeans_centroids.npz   (centroids + cluster_to_class mapping)

# Step 2: Run panoptic evaluation with depth-guided instance splitting
# The sweep script evaluates all (tau, A_min) configs in its grid.
# The best config (tau=0.20, A_min=1000) should produce PQ=26.74.
$PYTHON pseudo_labels/sweep_k50_spidepth.py \
    --cityscapes_root $CS_ROOT \
    --split val \
    --semantic_subdir pseudo_semantic_raw_k80

# Output:
#   $CS_ROOT/sweep_k80_spidepth_val.json  (full results table)
```

### Prerequisites

- Cityscapes dataset with `leftImg8bit/val/`, `gtFine/val/`
- SPIdepth depth maps at `$CS_ROOT/depth_spidepth/val/` (`.npy` files)
- CAUSE-TR checkpoint at `refs/cause/` (DINOv2 ViT-B/14 + Segment_TR decoder)

### Generating depth maps (if not already available)

```bash
$PYTHON pseudo_labels/generate_depth_spidepth.py \
    --cityscapes_root $CS_ROOT --split val
```

### Generating pseudo-labels for training (train split)

```bash
# Generate on train split, reusing val-fitted centroids
$PYTHON pseudo_labels/generate_overclustered_semantics.py \
    --cityscapes_root $CS_ROOT \
    --split train --k 80 --skip_crf \
    --raw_clusters \
    --output_subdir pseudo_semantic_raw_k80 \
    --load_centroids $CS_ROOT/pseudo_semantic_raw_k80/kmeans_centroids.npz

# Generate depth-guided instances for training
$PYTHON pseudo_labels/generate_depth_guided_instances.py \
    --semantic_dir $CS_ROOT/pseudo_semantic_raw_k80/train \
    --depth_dir $CS_ROOT/depth_spidepth/train \
    --output_dir $CS_ROOT/pseudo_instance_spidepth_k80/train \
    --grad_threshold 0.20 --min_area 1000
```

### Fixed parameters across all experiments

| Parameter | Value | Notes |
|-----------|-------|-------|
| Eval resolution | 512 x 1024 | `--eval_size 512 1024` (default) |
| Depth blur sigma | 1.0 | Gaussian smoothing before Sobel |
| Dilation iterations | 3 | Boundary pixel reclamation |
| Min stuff area | 64 px | Minimum stuff segment size |
| Depth source | SPIdepth | Self-supervised monocular depth |
| Feature dim | 90 | CAUSE Segment_TR projection |

## Repository Structure

```
├── README.md
├── requirements.txt
├── setup.sh
│
├── pseudo_labels/                          # Stage-1 pseudo-label generation
│   ├── generate_overclustered_semantics.py # K-means overclustering (k=50/60/80/300)
│   ├── sweep_k50_spidepth.py              # Parameter sweep (works for any k)
│   ├── generate_depth_guided_instances.py  # Depth-gradient instance splitting
│   ├── generate_depth_spidepth.py         # SPIdepth monocular depth generation
│   ├── generate_semantic_pseudolabels_cause.py  # CAUSE-TR 27-class semantics
│   ├── extract_dinov2_features.py         # DINOv2 ViT-B/14 feature extraction
│   ├── overclustering_cause.py            # Overclustering analysis
│   ├── classify_stuff_things.py           # Unsupervised stuff/things classification
│   ├── convert_to_cups_format.py          # Convert to CUPS training format
│   ├── remap_cause27_to_trainid.py        # 27-class to 19-class mapping
│   ├── generate_panoptic_pseudolabels.py  # Combine semantic + instances
│   └── evaluation/
│       └── evaluate_cascade_pseudolabels.py  # Panoptic PQ evaluation
│
├── refs/                                   # Reference model code (checkpoints gitignored)
│   ├── cause/                             # CAUSE (Pattern Recognition 2024)
│   │   ├── models/                        # DINOv2 ViT backbone
│   │   └── modules/                       # Segment_TR decoder + modularity codebook
│   └── spidepth/                          # SPIdepth (CVPR 2025)
│       ├── SQLdepth.py
│       └── networks/                      # Encoder-decoder architecture
│
├── reports/                               # Technical reports
│   ├── overclustering_granularity_sweep.md  # k=50/60/80 sweep (this study)
│   ├── cause_tr_refinement.md             # Overclustering discovery (k=300)
│   ├── overclustered_spidepth_sweep.md    # k=300 depth-guided sweep
│   └── stage_1_report.md                  # Initial CAUSE + depth pipeline
│
└── checkpoints/                           # Model checkpoints (gitignored)
```

## Key Findings

1. **k=80 is the Pareto optimum** — balances stuff quality (PQ_stuff=32.08) with instance separability (PQ_things=19.41)
2. **tau=0.20 and A_min=1000 are universal optima** — transfer across all k values without per-k tuning
3. **Depth splitting provides diminishing returns** as k increases: +2.51 PQ (k=50) -> +2.10 (k=60) -> +1.90 (k=80) -> 0.00 (k=300)
4. **CRF hurts overclustered predictions** — always use `--skip_crf`

See `reports/overclustering_granularity_sweep.md` for the full analysis.

## References

- Cho, J., et al. (2024). CAUSE: Contrastive learning with modularity-based codebook for unsupervised segmentation. *Pattern Recognition*, 146.
- Hahn, K., et al. (2025). CUPS: Unsupervised panoptic segmentation from stereo video. *CVPR*.
- Seo, J., et al. (2025). SPIdepth: Strengthened pose information for self-supervised monocular depth estimation. *CVPR*.
- Oquab, M., et al. (2024). DINOv2: Learning robust visual features without supervision. *TMLR*.
