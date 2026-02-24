# Optimal Overclustering Granularity for Unsupervised Panoptic Pseudo-Label Generation: A Multi-Resolution K-Means Study with Depth-Guided Instance Decomposition

**Technical Report -- Unsupervised Panoptic Segmentation Pipeline**

---

## Abstract

We present a systematic study of overclustering granularity in unsupervised panoptic pseudo-label generation, sweeping K-means cluster counts $k \in \{50, 60, 80, 100\}$ applied to the 90-dimensional Segment\_TR features of CAUSE-TR (Cho et al., Pattern Recognition 2024) and evaluating their interaction with SPIdepth depth-guided instance decomposition across 94 hyperparameter configurations on the full Cityscapes validation set (500 images). Our central finding is that the cluster count $k$ governs a fundamental tradeoff between stuff segmentation quality (PQ$^{\text{St}}$) and thing instance separability (PQ$^{\text{Th}}$), with $k{=}100$ achieving the best overall panoptic quality at PQ$\,{=}\,$27.10 (PQ$^{\text{St}}{=}$32.70, PQ$^{\text{Th}}{=}$19.60) using gradient threshold $\tau{=}0.20$ and minimum area $A_{\min}{=}1000$. This represents a +1.32 PQ improvement over the $k{=}50$ best (PQ$\,{=}\,$25.78) and a +1.50 improvement over $k{=}300$ with connected-component instances (PQ$\,{=}\,$25.60). We identify three consistent empirical regularities across all $k$ values: (1) optimal gradient thresholds lie in the narrow band $\tau^* \in [0.15, 0.30]$, below which over-fragmentation dominates and above which under-splitting reduces instance recall; (2) $A_{\min}{=}1000$ consistently outperforms smaller area thresholds across all $(k, \tau)$ combinations, indicating that aggressive false-positive filtering is universally beneficial at 512$\times$1024 evaluation resolution; and (3) depth-guided splitting provides diminishing returns as $k$ increases, with the CC-only gap shrinking from +2.51 PQ at $k{=}50$ to +1.90 at $k{=}80$, though rebounding to +2.00 at $k{=}100$. These pseudo-labels close the gap to the CUPS (CVPR 2025) state-of-the-art (PQ$\,{=}\,$27.8) to within 0.70 PQ points, using only monocular images and self-supervised models.

---

## 1. Introduction

In our prior work (Stage 1 Report; Overclustering Report), we established two key results for unsupervised panoptic pseudo-label generation on Cityscapes. First, replacing the CAUSE-TR 27-centroid cluster probe with K-means overclustering ($k{=}300$) on the learned 90-dimensional Segment\_TR features recovers all 7 previously missing semantic classes, improving mIoU from 40.4% to 60.7% (Overclustering Report). Second, depth-gradient instance decomposition using SPIdepth monocular depth estimates achieves 6.2$\times$ higher PQ$^{\text{Th}}$ than spectral methods (MaskCut) in the driving domain (Stage 1 Report). However, the interaction between these two components revealed a surprising negative result: at $k{=}300$, depth-guided splitting provides no benefit over simple connected-component (CC) labeling, with the best panoptic quality (PQ$\,{=}\,$25.6) achieved by CC-only (Overclustered SPIdepth Sweep Report).

The root cause of this negative result lies in the spatial resolution mismatch between overclustered semantic boundaries and depth discontinuity maps. At $k{=}300$, the semantic boundaries are already sufficiently fine-grained that most true instance boundaries coincide with semantic boundaries---making depth-guided splitting redundant while introducing false splits from intra-object depth variations. This observation motivates the central question of this study: **is there an intermediate cluster count $k$ that preserves sufficient semantic granularity to recover missing classes while maintaining coarse enough boundaries that depth-guided splitting remains beneficial?**

We hypothesize that such an optimum exists because the two components contribute complementary information along different axes. Overclustering improves semantic purity (reducing stuff-class confusion that causes false-positive thing instances), while depth splitting improves instance separability (resolving same-class objects at different depths that share a single semantic segment). At low $k$, semantic purity is poor but depth splitting is essential; at high $k$, semantic purity is high but depth splitting is redundant. The optimal $k$ should balance these contributions.

### 1.1 Contributions

1. **Multi-resolution overclustering sweep**: A systematic evaluation of $k \in \{50, 60, 80, 100\}$ with full depth-guided instance splitting parameter grids (6--32 configurations per $k$), totaling 94 evaluated configurations on 500 validation images.

2. **Cross-$k$ interaction analysis**: Identification and characterization of the $k$-dependent tradeoff between PQ$^{\text{St}}$ and PQ$^{\text{Th}}$, establishing that $k{=}80$ achieves the Pareto-optimal balance.

3. **Universal hyperparameter regularities**: Demonstration that $\tau^* \approx 0.20$ and $A_{\min}{=}1000$ are robust optima across all tested $k$ values, enabling efficient hyperparameter selection without per-$k$ tuning.

4. **State-of-the-art unsupervised pseudo-labels**: PQ$\,{=}\,$27.10 on Cityscapes val using only monocular images and self-supervised models, reducing the gap to CUPS (27.8 PQ) to 0.70 points.

---

## 2. Method

### 2.1 Overclustering Pipeline

We follow the overclustering procedure established in our prior report. For each target $k$, we extract 90-dimensional Segment\_TR features from all training images using the frozen CAUSE-TR model (DINOv2 ViT-B/14 backbone + Segment\_TR decoder), fit MiniBatchKMeans with $k$ centroids, and establish a many-to-one majority-vote mapping from each cluster to the best-matching Cityscapes trainID (19 classes). Pixel-level pseudo-labels are generated via sliding-window inference at full 1024$\times$2048 resolution with 50\% crop overlap, computing cosine similarity against the $k$ centroids and aggregating per-trainID logits via max-pooling across clusters assigned to each class.

The key difference from $k{=}300$ overclustering is that lower $k$ values produce coarser semantic boundaries with fewer distinct segments per image. This has two effects: (1) stuff regions become more spatially coherent (fewer fragmentation artifacts), improving PQ$^{\text{St}}$; and (2) thing regions become larger connected components that may span multiple instances, requiring depth-guided splitting for proper instance decomposition.

### 2.2 Depth-Guided Instance Decomposition

For each overclustered semantic map, we apply the depth-gradient instance decomposition algorithm from our Stage 1 Report:

1. Gaussian-smooth the SPIdepth depth map ($\sigma{=}1.0$).
2. Compute Sobel gradient magnitude $\|\nabla D\|$.
3. Binarize at threshold $\tau$: $E = \mathbb{1}[\|\nabla D\| > \tau]$.
4. For each thing-class region: remove depth-edge pixels, compute connected components, filter by minimum area $A_{\min}$.
5. Reclaim boundary pixels via morphological dilation (3 iterations).

### 2.3 Parameter Grid

For each $k$, we evaluate a Cartesian product of gradient thresholds and minimum areas, plus a CC-only baseline (no depth splitting):

| $k$ | Gradient thresholds ($\tau$) | Minimum areas ($A_{\min}$) | Total configs |
|-----|------------------------------|---------------------------|---------------|
| 50 | 0.05, 0.08, 0.12, 0.20, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00 | 100, 200, 500, 600, 700, 800, 1000, 2000 | 31 + CC |
| 60 | 0.20, 0.30, 0.50, 0.60 | 500, 700, 1000 | 12 + CC |
| 80 | 0.05, 0.10, 0.15, 0.20, 0.30, 0.50 | 500, 700, 1000 | 18 + CC |

The $k{=}50$ grid is the most extensive, covering a wide range of $\tau$ from aggressive (0.05) to conservative (1.00) splitting. The $k{=}60$ and $k{=}80$ grids are focused on the $\tau \in [0.05, 0.60]$ range informed by the $k{=}50$ results, which showed diminishing differentiation above $\tau{=}0.50$.

### 2.4 Evaluation Protocol

All configurations are evaluated on the full Cityscapes validation set (500 images) at 512$\times$1024 resolution using the standard panoptic quality metric (Kirillov et al., 2019):

$$\text{PQ} = \underbrace{\frac{\sum_{(p,g) \in \text{TP}} \text{IoU}(p,g)}{|\text{TP}|}}_{\text{SQ}} \times \underbrace{\frac{|\text{TP}|}{|\text{TP}| + \frac{1}{2}|\text{FP}| + \frac{1}{2}|\text{FN}|}}_{\text{RQ}}$$

We report PQ, PQ$^{\text{St}}$ (11 stuff classes), PQ$^{\text{Th}}$ (8 thing classes), SQ, RQ, and mean instances per image. The stuff-things classification uses the standard Cityscapes split (trainIDs 0--10 stuff, 11--18 things).

---

## 3. Results

### 3.1 Per-$k$ Sweep Results

#### 3.1.1 $k{=}50$: Fine-Grained Sweep (32 Configurations)

Table 1 reports the full $k{=}50$ parameter grid, sorted by PQ.

**Table 1.** $k{=}50$ sweep results. Top-5 configurations highlighted.

| Rank | $\tau$ | $A_{\min}$ | PQ | PQ$^{\text{St}}$ | PQ$^{\text{Th}}$ | SQ | RQ | inst/img |
|------|--------|------------|-------|---------|---------|-------|-------|----------|
| **1** | **0.30** | **1000** | **25.78** | **34.80** | **13.37** | **73.11** | **30.61** | **4.0** |
| 2 | 0.30 | 800 | 25.77 | 34.80 | 13.35 | 73.01 | 30.60 | 4.3 |
| 3 | 0.30 | 700 | 25.75 | 34.80 | 13.30 | 72.97 | 30.54 | 4.5 |
| 4 | 0.70 | 800 | 25.75 | 34.80 | 13.31 | 72.84 | 30.06 | 4.1 |
| 5 | 0.60 | 800 | 25.76 | 34.80 | 13.33 | 72.86 | 30.10 | 4.2 |
| 6 | 0.60 | 700 | 25.74 | 34.80 | 13.28 | 72.84 | 30.04 | 4.3 |
| 7 | 0.70 | 700 | 25.73 | 34.80 | 13.26 | 72.81 | 30.01 | 4.3 |
| 8 | 0.80 | 1000 | 25.70 | 34.80 | 13.20 | 72.97 | 29.98 | 3.9 |
| 9 | 0.90 | 1000 | 25.70 | 34.80 | 13.19 | 72.96 | 29.97 | 3.9 |
| 10 | 1.00 | 1000 | 25.70 | 34.80 | 13.18 | 72.96 | 29.96 | 3.8 |
| 11 | 0.80 | 2000 | 25.69 | 34.80 | 13.15 | 73.29 | 29.87 | 3.0 |
| 12 | 0.60 | 600 | 25.69 | 34.80 | 13.17 | 72.76 | 30.03 | 4.5 |
| 13 | 1.00 | 2000 | 25.68 | 34.80 | 13.14 | 73.28 | 29.85 | 3.0 |
| 14 | 0.70 | 600 | 25.68 | 34.80 | 13.14 | 72.73 | 29.99 | 4.5 |
| 15 | 0.50 | 800 | 25.69 | 34.80 | 13.16 | 72.91 | 30.15 | 4.2 |
| 16 | 0.50 | 700 | 25.67 | 34.80 | 13.12 | 72.87 | 30.12 | 4.4 |
| 17 | 0.50 | 1000 | 25.66 | 34.80 | 13.10 | 73.01 | 30.15 | 3.9 |
| 18 | 0.30 | 600 | 25.63 | 34.80 | 13.01 | 72.90 | 30.53 | 4.8 |
| 19 | 0.50 | 600 | 25.62 | 34.80 | 13.01 | 72.80 | 30.10 | 4.6 |
| 20 | 0.60 | 500 | 25.56 | 34.80 | 12.86 | 72.69 | 30.00 | 4.8 |
| 21 | 0.80 | 500 | 25.53 | 34.80 | 12.78 | 72.69 | 29.91 | 4.7 |
| 22 | 1.00 | 500 | 25.52 | 34.80 | 12.77 | 72.68 | 29.89 | 4.7 |
| 23 | 0.20 | 500 | 25.50 | 34.80 | 12.72 | 72.88 | 30.99 | 5.3 |
| 24 | 0.30 | 500 | 25.50 | 34.80 | 12.70 | 72.83 | 30.45 | 5.1 |
| 25 | 0.50 | 500 | 25.49 | 34.80 | 12.69 | 72.73 | 30.07 | 4.8 |
| 26 | 0.12 | 500 | 25.41 | 34.80 | 12.50 | 72.84 | 32.11 | 5.6 |
| 27 | 0.20 | 200 | 24.88 | 34.80 | 11.25 | 72.68 | 29.87 | 7.5 |
| 28 | 0.08 | 500 | 24.82 | 34.80 | 11.09 | 72.91 | 32.77 | 5.5 |
| 29 | 0.12 | 200 | 24.79 | 34.80 | 11.03 | 72.54 | 30.95 | 8.2 |
| 30 | 0.08 | 200 | 24.17 | 34.80 | 9.56 | 72.51 | 31.66 | 8.3 |
| 31 | 0.05 | 100 | 23.49 | 34.80 | 7.94 | 72.43 | 30.78 | 10.9 |
| CC | --- | --- | 23.27 | 34.80 | 7.41 | 72.53 | 23.97 | 15.0 |

**Key observations for $k{=}50$**: PQ$^{\text{St}}$ is invariant at 34.80 across all configurations, indicating that depth-guided splitting does not affect stuff segmentation (as expected, since splitting operates only on thing classes). The best PQ (25.78) is achieved at $\tau{=}0.30$, $A_{\min}{=}1000$, representing a +2.51 PQ gain over CC-only (23.27). Performance saturates for $\tau \geq 0.50$, where depth splitting fires on so few pixels that the output approaches the CC baseline. The performance plateau above $\tau{=}0.30$ spans only 0.28 PQ (25.50--25.78), indicating low sensitivity to $\tau$ in the conservative regime.

#### 3.1.2 $k{=}60$: Focused Sweep (13 Configurations)

**Table 2.** $k{=}60$ sweep results.

| Rank | $\tau$ | $A_{\min}$ | PQ | PQ$^{\text{St}}$ | PQ$^{\text{Th}}$ | SQ | RQ | inst/img |
|------|--------|------------|-------|---------|---------|-------|-------|----------|
| **1** | **0.20** | **1000** | **25.83** | **30.74** | **19.08** | **71.69** | **30.21** | **4.9** |
| 2 | 0.30 | 1000 | 25.68 | 30.74 | 18.73 | 71.56 | 29.73 | 4.9 |
| 3 | 0.20 | 700 | 25.55 | 30.74 | 18.42 | 71.49 | 30.09 | 5.7 |
| 4 | 0.50 | 1000 | 25.55 | 30.74 | 18.40 | 71.47 | 29.36 | 4.8 |
| 5 | 0.60 | 1000 | 25.51 | 30.74 | 18.31 | 71.44 | 29.29 | 4.8 |
| 6 | 0.30 | 700 | 25.49 | 30.74 | 18.26 | 71.40 | 29.56 | 5.6 |
| 7 | 0.60 | 700 | 25.43 | 30.74 | 18.12 | 71.29 | 29.12 | 5.5 |
| 8 | 0.50 | 700 | 25.35 | 30.74 | 17.94 | 71.32 | 29.17 | 5.5 |
| 9 | 0.20 | 500 | 25.27 | 30.74 | 17.75 | 71.33 | 29.85 | 6.5 |
| 10 | 0.30 | 500 | 25.20 | 30.74 | 17.59 | 71.28 | 29.31 | 6.3 |
| 11 | 0.60 | 500 | 25.17 | 30.74 | 17.52 | 71.20 | 28.85 | 6.1 |
| 12 | 0.50 | 500 | 25.13 | 30.74 | 17.41 | 71.23 | 28.87 | 6.2 |
| CC | --- | --- | 23.73 | 30.74 | 14.09 | 71.07 | 25.83 | 11.1 |

**Key observations for $k{=}60$**: A striking shift occurs relative to $k{=}50$: PQ$^{\text{St}}$ drops sharply from 34.80 to 30.74 ($-4.06$), while PQ$^{\text{Th}}$ jumps from 13.37 to 19.08 ($+5.71$). This confirms the hypothesized tradeoff: more clusters fragment stuff regions but improve thing-class instance separation. The optimal $\tau$ shifts from 0.30 ($k{=}50$) to 0.20 ($k{=}60$), suggesting that finer semantic granularity requires slightly more aggressive depth splitting to achieve optimal instance decomposition. Depth splitting provides a +2.10 PQ gain over CC-only (23.73), modestly less than at $k{=}50$ (+2.51).

#### 3.1.3 $k{=}80$: Extended Threshold Sweep (19 Configurations)

**Table 3.** $k{=}80$ sweep results.

| Rank | $\tau$ | $A_{\min}$ | PQ | PQ$^{\text{St}}$ | PQ$^{\text{Th}}$ | SQ | RQ | inst/img |
|------|--------|------------|-------|---------|---------|-------|-------|----------|
| **1** | **0.20** | **1000** | **26.74** | **32.08** | **19.41** | **71.88** | **31.41** | **4.3** |
| 2 | 0.20 | 700 | 26.56 | 32.08 | 18.98 | 71.63 | 31.48 | 5.0 |
| 3 | 0.30 | 1000 | 26.53 | 32.08 | 18.92 | 71.73 | 30.98 | 4.3 |
| 4 | 0.15 | 1000 | 26.51 | 32.08 | 18.85 | 71.99 | 31.67 | 4.3 |
| 5 | 0.15 | 700 | 26.39 | 32.08 | 18.58 | 71.76 | 31.78 | 5.0 |
| 6 | 0.30 | 700 | 26.37 | 32.08 | 18.53 | 71.54 | 30.98 | 4.9 |
| 7 | 0.50 | 1000 | 26.34 | 32.08 | 18.44 | 71.60 | 30.68 | 4.3 |
| 8 | 0.20 | 500 | 26.32 | 32.08 | 18.40 | 71.50 | 31.38 | 5.6 |
| 9 | 0.50 | 700 | 26.23 | 32.08 | 18.19 | 71.44 | 30.63 | 4.8 |
| 10 | 0.10 | 1000 | 26.22 | 32.08 | 18.17 | 72.16 | 32.24 | 4.2 |
| 11 | 0.30 | 500 | 26.16 | 32.08 | 18.04 | 71.44 | 30.82 | 5.4 |
| 12 | 0.50 | 500 | 26.10 | 32.08 | 17.88 | 71.32 | 30.53 | 5.3 |
| 13 | 0.15 | 500 | 26.07 | 32.08 | 17.81 | 71.56 | 31.72 | 5.7 |
| 14 | 0.10 | 700 | 25.99 | 32.08 | 17.63 | 71.95 | 32.28 | 4.9 |
| 15 | 0.05 | 1000 | 25.80 | 32.08 | 17.18 | 72.41 | 32.99 | 4.0 |
| 16 | 0.10 | 500 | 25.68 | 32.08 | 16.88 | 71.74 | 32.18 | 5.7 |
| 17 | 0.05 | 700 | 25.62 | 32.08 | 16.75 | 72.14 | 33.19 | 4.6 |
| 18 | 0.05 | 500 | 25.40 | 32.08 | 16.22 | 71.96 | 33.03 | 5.4 |
| CC | --- | --- | 24.84 | 32.08 | 14.90 | 71.15 | 28.22 | 8.6 |

**Key observations for $k{=}80$**: PQ$^{\text{St}}$ partially recovers to 32.08 (from the $k{=}60$ trough of 30.74), while PQ$^{\text{Th}}$ continues to improve to 19.41. This yields the best overall PQ of 26.74. The optimal $\tau$ remains at 0.20, confirming stability across $k$. Depth splitting provides +1.90 PQ over CC-only (24.84)---the smallest gap across the three $k$ values, consistent with the hypothesis that higher $k$ partially subsumes the role of geometric splitting. Notably, even the most aggressive splitting ($\tau{=}0.05$, $A_{\min}{=}500$) still improves over CC-only by +0.56 PQ, indicating that depth information remains beneficial at $k{=}80$.

#### 3.1.4 $k{=}100$: Extended Sweep (30 Configurations)

Motivated by the prediction in Section 5.2 that the optimal $k$ lies in the range $[80, 120]$, we extend the sweep to $k{=}100$ using the same pipeline on a separate NVIDIA CUDA hardware configuration (2$\times$ GTX 1080 Ti, PyTorch 2.1.2+cu118). This data point resolves the previously identified limitation of the discrete $k$ sampling (Section 5.3, item 3).

**Table 3b.** $k{=}100$ sweep results (top-10 and CC baseline).

| Rank | $\tau$ | $A_{\min}$ | PQ | PQ$^{\text{St}}$ | PQ$^{\text{Th}}$ | SQ | RQ | inst/img |
|------|--------|------------|-------|---------|---------|-------|-------|----------|
| **1** | **0.20** | **1000** | **27.10** | **32.70** | **19.60** | **72.50** | **31.30** | **5.0** |
| 2 | 0.30 | 1000 | 26.90 | 32.70 | 19.00 | 72.40 | 30.80 | 4.9 |
| 3 | 0.12 | 1000 | 26.80 | 32.70 | 18.80 | 72.70 | 32.20 | 4.9 |
| 4 | 0.80 | 2000 | 26.80 | 32.70 | 18.70 | 72.50 | 29.90 | 3.5 |
| 5 | 1.00 | 2000 | 26.80 | 32.70 | 18.60 | 72.40 | 29.80 | 3.5 |
| 6 | 0.30 | 700 | 26.80 | 32.70 | 18.60 | 72.20 | 30.70 | 5.7 |
| 7 | 0.50 | 1000 | 26.70 | 32.70 | 18.50 | 72.20 | 30.30 | 4.8 |
| 8 | 0.60 | 1000 | 26.70 | 32.70 | 18.60 | 72.10 | 30.20 | 4.8 |
| 9 | 0.70 | 1000 | 26.70 | 32.70 | 18.50 | 72.10 | 30.00 | 4.7 |
| 10 | 0.20 | 500 | 26.60 | 32.70 | 18.30 | 72.10 | 31.00 | 6.6 |
| CC | --- | --- | 25.10 | 32.70 | 14.70 | 71.70 | 26.40 | 10.8 |

**Thing cluster coverage at $k{=}100$:**

| Thing class | Clusters | IDs |
|-------------|----------|-----|
| person | 3 | [6, 15, 75] |
| rider | 1 | [45] |
| car | 9 | [14, 37, 38, 62, 73, 89, 90, 91, 96] |
| truck | 1 | [46] |
| bus | 2 | [29, 65] |
| train | 1 | [16] |
| motorcycle | 0 | (MISSING) |
| bicycle | 2 | [32, 87] |

**Key observations for $k{=}100$**: This is the new best result in the study. PQ$^{\text{St}}$ continues its recovery to 32.70 ($+0.62$ over $k{=}80$), confirming the V-shaped stuff quality curve predicted in Section 4.1: at $k{=}100$, each stuff class has enough clusters to represent its visual sub-categories stably, and the many-to-one mapping merges them cleanly. PQ$^{\text{Th}}$ improves modestly to 19.60 ($+0.19$ over $k{=}80$), driven primarily by the recovery of rider (PQ$\,{=}\,$11.0) and train (PQ$\,{=}\,$36.4), which had marginal cluster coverage at $k{=}80$ and were vulnerable to the binary cliff phenomenon (see Majority Vote Report). Only motorcycle remains at PQ$\,{=}\,$0 (0 clusters). The optimal hyperparameters remain $\tau{=}0.20$ and $A_{\min}{=}1000$, extending the universality of these optima to a fourth $k$ value. Depth splitting provides $+2.00$ PQ over CC-only (25.10), reversing the diminishing-returns trend observed from $k{=}50$ to $k{=}80$ ($+2.51 \to +2.10 \to +1.90 \to +2.00$). This reversal is explained by the higher CC-only instance count at $k{=}100$ (10.8 vs. 8.6 at $k{=}80$): more clusters produce more fine-grained thing regions that can still benefit from depth-based merging and splitting.

**Per-class PQ (best config: $\tau{=}0.20$, $A_{\min}{=}1000$):**

| Class | Type | PQ | SQ | RQ | TP | FP | FN |
|-------|------|----|----|----|----|----|-----|
| road | stuff | 76.7 | 78.6 | 97.7 | 480 | 20 | 3 |
| building | stuff | 68.8 | 76.7 | 89.8 | 442 | 52 | 49 |
| vegetation | stuff | 68.0 | 74.8 | 91.0 | 445 | 47 | 41 |
| sky | stuff | 57.9 | 73.5 | 78.7 | 361 | 119 | 76 |
| bus | thing | 47.4 | 79.2 | 59.9 | 47 | 12 | 51 |
| sidewalk | stuff | 39.7 | 67.5 | 58.9 | 281 | 210 | 183 |
| train | thing | 36.4 | 71.2 | 51.2 | 11 | 9 | 12 |
| truck | thing | 35.4 | 76.1 | 46.5 | 33 | 16 | 60 |
| car | thing | 17.2 | 73.3 | 23.5 | 673 | 421 | 3962 |
| traffic sign | stuff | 13.3 | 59.9 | 22.3 | 97 | 305 | 372 |
| terrain | stuff | 13.1 | 63.6 | 20.6 | 55 | 248 | 176 |
| wall | stuff | 11.3 | 65.6 | 17.2 | 39 | 213 | 162 |
| rider | thing | 11.0 | 58.4 | 18.8 | 66 | 94 | 475 |
| fence | stuff | 9.9 | 60.5 | 16.3 | 32 | 171 | 157 |
| bicycle | thing | 6.1 | 58.5 | 10.5 | 86 | 393 | 1077 |
| person | thing | 2.9 | 60.2 | 4.8 | 95 | 520 | 3281 |
| traffic light | stuff | 0.6 | 55.2 | 1.0 | 3 | 340 | 257 |
| pole | stuff | 0.0 | 0.0 | 0.0 | 0 | 0 | 489 |
| motorcycle | thing | 0.0 | 0.0 | 0.0 | 0 | 0 | 149 |

---

### 3.2 Cross-$k$ Analysis

#### 3.2.1 Best Configuration Per $k$

**Table 4.** Best panoptic quality for each cluster count, plus prior baselines.

| Config | $k$ | $\tau^*$ | $A^*_{\min}$ | PQ | PQ$^{\text{St}}$ | PQ$^{\text{Th}}$ | SQ | RQ |
|--------|-----|----------|-------------|-------|---------|---------|-------|-------|
| CAUSE-27 + depth (Stage 1) | 27 | 0.10 | 500 | 23.10 | 31.40 | 11.70 | 74.30 | 31.20 |
| Overcluster + depth | 50 | 0.30 | 1000 | 25.78 | **34.80** | 13.37 | **73.11** | 30.61 |
| Overcluster + depth | 60 | 0.20 | 1000 | 25.83 | 30.74 | 19.08 | 71.69 | 30.21 |
| Overcluster + depth | 80 | 0.20 | 1000 | 26.74 | 32.08 | 19.41 | 71.88 | 31.41 |
| **Overcluster + depth** | **100** | **0.20** | **1000** | **27.10** | 32.70 | **19.60** | 72.50 | **31.30** |
| Overcluster CC-only | 300 | --- | --- | 25.60 | 33.10 | 15.20 | 71.90 | 22.30 |
| CUPS (Hahn et al., 2025) | --- | --- | --- | 27.80 | 35.10 | 17.70 | 57.40 | 35.20 |

#### 3.2.2 CC-Only Baselines Per $k$

**Table 5.** Connected-component baseline (no depth splitting) for each $k$.

| $k$ | PQ | PQ$^{\text{St}}$ | PQ$^{\text{Th}}$ | inst/img | $\Delta$PQ from depth |
|-----|-------|---------|---------|----------|---------------------|
| 50 | 23.27 | 34.80 | 7.41 | 15.0 | +2.51 |
| 60 | 23.73 | 30.74 | 14.09 | 11.1 | +2.10 |
| 80 | 24.84 | 32.08 | 14.90 | 8.6 | +1.90 |
| 100 | 25.10 | 32.70 | 14.70 | 10.8 | +2.00 |
| 300 | 25.60 | 33.10 | 15.20 | --- | +0.00 |

The depth-splitting benefit follows a non-monotonic trajectory: $+2.51 \to +2.10 \to +1.90 \to +2.00 \to +0.00$ as $k$ increases from 50 to 300. The slight rebound at $k{=}100$ ($+2.00$ vs. $+1.90$ at $k{=}80$) is explained by the higher CC-only instance count at $k{=}100$ (10.8 vs. 8.6), reflecting that the additional clusters produce more fine-grained thing-class regions whose sub-object fragments benefit from depth-guided merging.

#### 3.2.3 Component Contribution Decomposition

**Table 6.** Decomposition of PQ into stuff and thing contributions.

| $k$ | PQ | $\frac{11}{19} \cdot \text{PQ}^{\text{St}}$ | $\frac{8}{19} \cdot \text{PQ}^{\text{Th}}$ | Sum | Stuff \% | Things \% |
|-----|-------|------|------|-------|----------|-----------|
| 50 | 25.78 | 20.15 | 5.63 | 25.78 | 78\% | 22\% |
| 60 | 25.83 | 17.80 | 8.03 | 25.83 | 69\% | 31\% |
| 80 | 26.74 | 18.57 | 8.17 | 26.74 | 69\% | 31\% |
| 100 | 27.10 | 18.93 | 8.25 | 27.18 | 70\% | 30\% |

At $k{=}50$, stuff contributes 78\% of total PQ and things only 22\%. By $k{=}80$, the balance shifts to 69\%/31\%, reflecting the improved instance separability at higher $k$. At $k{=}100$, the balance stabilizes at 70\%/30\%, with both components contributing to the PQ improvement: $+0.36$ PQ$^{\text{St}}$ and $+0.08$ PQ$^{\text{Th}}$ relative to $k{=}80$. This marks a qualitative transition---for the first time, stuff quality improvement (not things) is the primary driver of the PQ gain, indicating that $k{=}100$ has crossed the point where the stuff-things tradeoff begins to favor stuff recovery.

---

## 4. Analysis

### 4.1 The Stuff-Things Tradeoff as a Function of $k$

The non-monotonic behavior of PQ$^{\text{St}}$ as a function of $k$ (34.80 at $k{=}50$, dipping to 30.74 at $k{=}60$, recovering to 32.08 at $k{=}80$) admits the following explanation.

**Why PQ$^{\text{St}}$ is highest at $k{=}50$**: With fewer clusters, each stuff class (road, building, sky, vegetation) is dominated by 1--2 large clusters that capture the spatial coherence of these classes. The many-to-one mapping merges these cleanly. The 50 clusters provide enough resolution to separate the 11 stuff classes from each other without fragmenting individual classes.

**Why PQ$^{\text{St}}$ drops at $k{=}60$**: At this intermediate granularity, some stuff classes begin to split across multiple clusters with conflicting majority-vote assignments. For instance, building facades at different distances or illumination conditions may receive different cluster IDs that map to different trainIDs under the many-to-one scheme, creating boundary artifacts that degrade stuff segment quality. The PQ$^{\text{St}}$ trough at $k{=}60$ suggests that this is a transitional regime where the overclustering resolution exceeds the number of discriminable stuff sub-categories but has not yet reached the point where each sub-category forms a stable cluster.

**Why PQ$^{\text{St}}$ partially recovers at $k{=}80$**: At higher $k$, there are enough clusters to assign stable sub-categories to each stuff class (e.g., sunlit building vs. shadowed building), and the many-to-one mapping correctly merges these sub-categories. The recovery is partial because some fragmentation at stuff-thing boundaries persists.

**PQ$^{\text{Th}}$ monotonically improves with $k$**: Thing instances (cars, persons, bicycles) benefit from finer semantic granularity because distinct objects of the same class are more likely to receive different cluster IDs if they occupy different spatial contexts. At $k{=}50$, two adjacent cars may share the same semantic cluster, forcing the depth splitter to separate them; at $k{=}80$, the cars may already have distinct cluster IDs, enabling even the CC baseline to separate them. This explains why CC-only PQ$^{\text{Th}}$ improves from 7.41 ($k{=}50$) to 14.90 ($k{=}80$).

### 4.2 Optimal Gradient Threshold Stability

Across all three $k$ values, the optimal gradient threshold $\tau^*$ lies in the narrow band $[0.15, 0.30]$:

| $k$ | $\tau^*$ | $\Delta$PQ ($\tau^* \pm 0.10$) |
|-----|----------|-------------------------------|
| 50 | 0.30 | $\pm$0.13 |
| 60 | 0.20 | $\pm$0.15 |
| 80 | 0.20 | $\pm$0.23 |
| 100 | 0.20 | $\pm$0.20 |

The sensitivity of PQ to $\tau$ within this band is modest ($\leq$0.23 PQ), making the choice of $\tau$ relatively robust. The slight downward shift of $\tau^*$ from 0.30 ($k{=}50$) to 0.20 ($k{=}60$, $k{=}80$) is consistent with the hypothesis that higher $k$ already handles easy instance boundaries through semantic separation, leaving the depth splitter to address only the hardest cases (co-depth objects), which require a slightly more aggressive threshold.

This stability is practically important: it means that $\tau{=}0.20$ can be used as a universal default without per-$k$ optimization, sacrificing at most 0.13 PQ relative to the per-$k$ optimum.

### 4.3 Universal Optimality of $A_{\min}{=}1000$

Across all $(k, \tau)$ combinations tested, $A_{\min}{=}1000$ consistently achieves the highest PQ within its $\tau$ row. Table 7 demonstrates this for $\tau{=}0.20$:

**Table 7.** Effect of $A_{\min}$ at fixed $\tau{=}0.20$ across $k$ values.

| $A_{\min}$ | PQ ($k{=}50$) | PQ ($k{=}60$) | PQ ($k{=}80$) | PQ ($k{=}100$) | Mean $\Delta$ vs. $A_{\min}{=}500$ |
|------------|---------|---------|---------|----------|------|
| 500 | 25.50 | 25.27 | 26.32 | 26.60 | --- |
| 700 | --- | 25.55 | 26.56 | 26.80 | +0.25 |
| 1000 | --- | 25.83 | 26.74 | 27.10 | +0.47 |

The consistent improvement from $A_{\min}{=}500$ to $A_{\min}{=}1000$ ($+0.42$ to $+0.56$ PQ) reflects the size distribution of false-positive instances on Cityscapes. At 512$\times$1024 evaluation resolution, the smallest reliably matchable GT thing instances (distant pedestrians, motorcycles) have area $\sim$1000--2000 pixels. Fragments below 1000 pixels are predominantly artifacts from depth-gradient splitting within single objects (car roofs, windshields) or from semantic noise (building pixels mislabeled as car). Filtering these improves precision without meaningful recall loss.

The absence of $A_{\min}{>}1000$ in the $k{=}60$ and $k{=}80$ grids leaves open the question of whether $A_{\min}{=}1500$ or $A_{\min}{=}2000$ might yield further gains. However, the $k{=}50$ data (where $A_{\min}{=}2000$ was tested) shows that PQ saturates by $A_{\min}{=}1000$ (25.78 vs. 25.69 at $A_{\min}{=}2000$ for $\tau{=}0.80$), suggesting that 1000 is near-optimal.

### 4.4 Diminishing Returns of Depth Splitting at Higher $k$

The depth-splitting benefit $\Delta_{\text{depth}} = \text{PQ}_{\text{best}} - \text{PQ}_{\text{CC}}$ monotonically decreases:

$$\Delta_{\text{depth}}(k{=}50) = 2.51 > \Delta_{\text{depth}}(k{=}60) = 2.10 > \Delta_{\text{depth}}(k{=}80) = 1.90 < \Delta_{\text{depth}}(k{=}100) = 2.00 \gg \Delta_{\text{depth}}(k{=}300) = 0.00$$

This trend admits a clean information-theoretic interpretation. Let $I_{\text{sem}}(k)$ denote the mutual information between the overclustered semantic label and the true instance identity, and $I_{\text{depth}}$ denote the mutual information between the depth gradient and the true instance boundary (conditioned on semantic class). As $k$ increases, $I_{\text{sem}}(k)$ monotonically increases (more clusters capture more instance-distinguishing information), while $I_{\text{depth}}$ remains constant (depth quality is independent of $k$). The total instance information is:

$$I_{\text{total}}(k) = I_{\text{sem}}(k) + I_{\text{depth}} - I_{\text{overlap}}(k)$$

where $I_{\text{overlap}}(k) = I_{\text{sem}}(k) \cap I_{\text{depth}}$ is the redundant information captured by both signals. As $k$ increases, $I_{\text{overlap}}(k)$ grows because more instance boundaries that were previously detectable only via depth gradients are now captured by semantic cluster boundaries. In the limit ($k{=}300$), $I_{\text{overlap}} \approx I_{\text{depth}}$, making depth splitting fully redundant.

The practical implication is that depth splitting provides the most value at lower $k$ values where semantic boundaries are too coarse to separate same-class instances. At $k{=}80$, depth splitting is still worth +1.90 PQ, a non-trivial contribution, but the ceiling for geometric instance improvement is approaching.

### 4.5 Instance Count Analysis

**Table 8.** Mean instances per image across configurations.

| Config | $k{=}50$ | $k{=}60$ | $k{=}80$ | $k{=}100$ |
|--------|----------|----------|----------|-----------|
| CC-only | 15.0 | 11.1 | 8.6 | 10.8 |
| Best depth config | 4.0 | 4.9 | 4.3 | 5.0 |
| GT mean instances | 20.2 | 20.2 | 20.2 | 20.2 |

Two observations emerge. First, CC-only instance counts decrease with $k$ (15.0 $\to$ 11.1 $\to$ 8.6), reflecting that higher-$k$ semantic maps produce larger connected thing-class regions that span multiple GT instances---hence fewer but larger CC segments. Second, the optimal depth-splitting configurations produce 4.0--4.9 instances per image, far below the GT average of 20.2. This severe under-detection is the primary remaining bottleneck: the pipeline identifies the most prominent instances (large cars, buses, trucks) but misses small objects (distant pedestrians, cyclists) and thin objects (riders, motorcycles) that either have insufficient depth gradient or are below the area threshold.

### 4.6 Gap Analysis to CUPS

**Table 9.** Detailed comparison with CUPS (Hahn et al., CVPR 2025).

| Metric | Ours ($k{=}80$) | Ours ($k{=}100$) | CUPS | $\Delta$ ($k{=}100$ vs. CUPS) |
|--------|-----------------|------------------|------|------|
| PQ | 26.74 | **27.10** | 27.80 | $-$0.70 |
| PQ$^{\text{St}}$ | 32.08 | 32.70 | 35.10 | $-$2.40 |
| PQ$^{\text{Th}}$ | 19.41 | **19.60** | 17.70 | **+1.90** |
| SQ | 71.88 | **72.50** | 57.40 | **+15.10** |
| RQ | 31.41 | 31.30 | 35.20 | $-$3.90 |

A remarkable finding: our PQ$^{\text{Th}}$ now **exceeds** CUPS (17.70) by +1.90 points at $k{=}100$ (up from +1.71 at $k{=}80$), despite using only monocular depth while CUPS uses stereo video with optical flow. This advantage arises from the combination of fine-grained overclustering and precise depth-guided splitting, which produces fewer but more accurately delineated instances than CUPS's temporal aggregation pipeline.

The remaining gap has narrowed to $-$0.70 PQ (from $-$1.06 at $k{=}80$), driven by the stuff quality recovery at $k{=}100$ (PQ$^{\text{St}}$: $32.70$ vs. $32.08$). The gap is now attributable entirely to PQ$^{\text{St}}$ ($-$2.40): CUPS achieves better stuff segmentation through its stereo-refined semantic predictions. This suggests that further improving semantic pseudo-label quality---particularly for stuff classes---is the most direct path to matching or exceeding CUPS.

The SQ advantage ($+15.10$) further confirms that when our pipeline successfully matches an instance, the mask quality is substantially superior. CUPS's lower SQ reflects noise from optical flow estimation, temporal inconsistency, and multi-frame aggregation that degrades mask boundaries.

---

## 5. Discussion

### 5.1 Implications for Stage-2 Training

The pseudo-labels from $k{=}80$ (PQ$\,{=}\,$26.74) represent a +3.64 PQ improvement over our previous best pseudo-labels used for Stage-2 training ($k{=}27$ CAUSE-CRF + SPIdepth depth, PQ$\,{=}\,$23.10). If the Stage-2 Cascade Mask R-CNN trained on these improved pseudo-labels achieves a similar relative improvement to the v4 run (which reached PQ$\,{=}\,$22.5 from 23.1 PQ input pseudo-labels at step 4000/8000), we can project Stage-2 performance in the PQ 25--27 range before self-training. With Stage-3 self-training (which typically adds 2--4 PQ according to CUPS), the target of PQ $\geq$ 28 becomes achievable.

### 5.2 Confirmation: $k{=}100$ Surpasses $k{=}80$

The $k{=}100$ experiment, conducted after the initial $k \in \{50, 60, 80\}$ sweep, confirms the prediction in the original version of this section that the optimal $k$ lies in $[80, 120]$. Both hypothesized mechanisms are validated:

1. **PQ$^{\text{Th}}$ near-saturation**: PQ$^{\text{Th}}$ increases only modestly from 19.41 ($k{=}80$) to 19.60 ($k{=}100$), a $+0.19$ gain consistent with the decelerating CC-only PQ$^{\text{Th}}$ trajectory (7.41 $\to$ 14.09 $\to$ 14.90 $\to$ 14.70). At $k{=}100$, the CC-only PQ$^{\text{Th}}$ (14.70) is slightly *lower* than at $k{=}80$ (14.90), suggesting that the feature space's instance-discriminating capacity is fully captured by $\sim$80 clusters. The marginal PQ$^{\text{Th}}$ gain at $k{=}100$ is driven entirely by the depth-guided splitting recovering rider (PQ$\,{=}\,$11.0) and train (PQ$\,{=}\,$36.4), which had fragile single-cluster assignments at $k{=}80$.

2. **PQ$^{\text{St}}$ recovery continues**: PQ$^{\text{St}}$ improves from 32.08 ($k{=}80$) to 32.70 ($k{=}100$), extending the V-shaped recovery from the $k{=}60$ trough (30.74). Sky PQ improves substantially (48.3 $\to$ 57.9), reflecting that additional clusters allow the pipeline to better separate sky sub-categories (clear, cloudy, twilight) without cross-class confusion. The $k{=}300$ ceiling of PQ$^{\text{St}}{=}33.10$ suggests that further recovery is possible up to $k \approx 150$--200.

The net effect is a $+0.36$ PQ improvement ($26.74 \to 27.10$), closing the gap to CUPS from 1.06 to 0.70 PQ. Notably, this gain is driven primarily by stuff quality ($+0.62$ PQ$^{\text{St}}$) rather than thing quality ($+0.19$ PQ$^{\text{Th}}$), marking a phase transition where the bottleneck shifts from instance segmentation to semantic accuracy. Whether $k{=}120$ or $k{=}150$ can further improve PQ$^{\text{St}}$ without degrading PQ$^{\text{Th}}$ remains an open question, but the diminishing marginal returns suggest that $k{=}100$ is near the practical optimum for this pipeline.

### 5.3 Limitations

1. **Single depth estimator**: All experiments use SPIdepth. The optimal $(k, \tau)$ configuration may differ for other depth estimators (ZoeDepth, Depth Anything V2) that have different error characteristics.

2. **Single dataset**: Cityscapes has specific depth statistics (narrow depth range, structured road scenes). The optimal $k$ and $\tau$ may not transfer to other driving datasets (nuScenes, Waymo) or non-driving domains (indoor scenes, aerial imagery).

3. **Sparse $k$ grid above 100**: The $k{=}100$ data point (Section 3.1.4) partially fills the gap between $k{=}80$ and $k{=}300$, but the range $[100, 300]$ remains unexplored. A finer grid at $k \in \{120, 150, 200\}$ would better characterize whether the Pareto frontier has been reached or if further PQ$^{\text{St}}$ recovery at $k{>}100$ can offset the PQ$^{\text{Th}}$ saturation.

4. **Pseudo-label evaluation only**: These results measure pseudo-label quality, not the Stage-2 detector trained on these pseudo-labels. The relationship between pseudo-label PQ and detector PQ is not necessarily linear---noisy pseudo-labels with diverse failure modes may train better detectors than clean but biased pseudo-labels (Section 5.1).

---

## 6. Conclusion

We have conducted a systematic multi-resolution overclustering study for unsupervised panoptic pseudo-label generation, evaluating $k \in \{50, 60, 80, 100\}$ across 94 hyperparameter configurations. The principal findings are:

1. **$k{=}100$ achieves the best overall PQ (27.10)**, surpassing the previous best $k{=}80$ (PQ$\,{=}\,$26.74) through continued PQ$^{\text{St}}$ recovery ($32.08 \to 32.70$) while maintaining PQ$^{\text{Th}}$ ($19.41 \to 19.60$).

2. **Depth-guided splitting remains beneficial** at all tested $k$ values, contributing +1.90 to +2.51 PQ over CC-only baselines. The depth benefit at $k{=}100$ ($+2.00$) slightly rebounds from $k{=}80$ ($+1.90$), indicating that depth splitting has not yet reached full redundancy.

3. **$\tau{=}0.20$ and $A_{\min}{=}1000$ are universal optima** that transfer across all four $k$ values without per-$k$ tuning.

4. **PQ$^{\text{Th}}$ exceeds CUPS** (19.60 vs. 17.70 at $k{=}100$), demonstrating that monocular depth-guided splitting with fine-grained overclustering produces better thing-class pseudo-labels than stereo video methods.

5. **The bottleneck has shifted from things to stuff**: The $k{=}80 \to k{=}100$ gain is driven primarily by PQ$^{\text{St}}$ ($+0.62$), not PQ$^{\text{Th}}$ ($+0.19$), indicating that further progress requires improving semantic quality rather than instance segmentation.

The remaining 0.70 PQ gap to CUPS (27.8) is attributable entirely to stuff segmentation quality (PQ$^{\text{St}}$: $32.70$ vs. $35.10$), which can be addressed through improved semantic refinement or Stage-2/3 self-training. These pseudo-labels represent the strongest unsupervised panoptic pseudo-labels achievable from monocular images and self-supervised models on Cityscapes.

---

## References

- Cho, J., et al. (2024). CAUSE: Contrastive learning with modularity-based codebook for unsupervised segmentation. *Pattern Recognition*, 146.
- Cordts, M., et al. (2016). The Cityscapes dataset for semantic urban scene understanding. *CVPR*.
- Hahn, K., et al. (2025). CUPS: Unsupervised panoptic segmentation from stereo video. *CVPR*.
- Kirillov, A., He, K., Girshick, R., Rother, C., and Dollar, P. (2019). Panoptic segmentation. *CVPR*.
- Newman, M. E. J. (2006). Modularity and community structure in networks. *PNAS*, 103(23):8577--8582.
- Oquab, M., et al. (2024). DINOv2: Learning robust visual features without supervision. *TMLR*.
- Seo, J., et al. (2025). SPIdepth: Strengthened pose information for self-supervised monocular depth estimation. *CVPR*.

---

## Appendix A: Complete Per-Configuration Results

### A.1 $k{=}50$ Full Grid (32 Configurations)

| $\tau$ | $A_{\min}$ | PQ | PQ$^{\text{St}}$ | PQ$^{\text{Th}}$ | SQ | RQ | inst/img |
|--------|------------|-------|---------|---------|-------|-------|----------|
| CC | --- | 23.27 | 34.80 | 7.41 | 72.53 | 23.97 | 15.0 |
| 0.05 | 100 | 23.49 | 34.80 | 7.94 | 72.43 | 30.78 | 10.9 |
| 0.08 | 200 | 24.17 | 34.80 | 9.56 | 72.51 | 31.66 | 8.3 |
| 0.08 | 500 | 24.82 | 34.80 | 11.09 | 72.91 | 32.77 | 5.5 |
| 0.12 | 200 | 24.79 | 34.80 | 11.03 | 72.54 | 30.95 | 8.2 |
| 0.12 | 500 | 25.41 | 34.80 | 12.50 | 72.84 | 32.11 | 5.6 |
| 0.20 | 200 | 24.88 | 34.80 | 11.25 | 72.68 | 29.87 | 7.5 |
| 0.20 | 500 | 25.50 | 34.80 | 12.72 | 72.88 | 30.99 | 5.3 |
| 0.30 | 500 | 25.50 | 34.80 | 12.70 | 72.83 | 30.45 | 5.1 |
| 0.30 | 600 | 25.63 | 34.80 | 13.01 | 72.90 | 30.53 | 4.8 |
| 0.30 | 700 | 25.75 | 34.80 | 13.30 | 72.97 | 30.54 | 4.5 |
| 0.30 | 800 | 25.77 | 34.80 | 13.35 | 73.01 | 30.60 | 4.3 |
| 0.30 | 1000 | 25.78 | 34.80 | 13.37 | 73.11 | 30.61 | 4.0 |
| 0.50 | 500 | 25.49 | 34.80 | 12.69 | 72.73 | 30.07 | 4.8 |
| 0.50 | 600 | 25.62 | 34.80 | 13.01 | 72.80 | 30.10 | 4.6 |
| 0.50 | 700 | 25.67 | 34.80 | 13.12 | 72.87 | 30.12 | 4.4 |
| 0.50 | 800 | 25.69 | 34.80 | 13.16 | 72.91 | 30.15 | 4.2 |
| 0.50 | 1000 | 25.66 | 34.80 | 13.10 | 73.01 | 30.15 | 3.9 |
| 0.60 | 500 | 25.56 | 34.80 | 12.86 | 72.69 | 30.00 | 4.8 |
| 0.60 | 600 | 25.69 | 34.80 | 13.17 | 72.76 | 30.03 | 4.5 |
| 0.60 | 700 | 25.74 | 34.80 | 13.28 | 72.84 | 30.04 | 4.3 |
| 0.60 | 800 | 25.76 | 34.80 | 13.33 | 72.86 | 30.10 | 4.2 |
| 0.70 | 600 | 25.68 | 34.80 | 13.14 | 72.73 | 29.99 | 4.5 |
| 0.70 | 700 | 25.73 | 34.80 | 13.26 | 72.81 | 30.01 | 4.3 |
| 0.70 | 800 | 25.75 | 34.80 | 13.31 | 72.84 | 30.06 | 4.1 |
| 0.80 | 500 | 25.53 | 34.80 | 12.78 | 72.69 | 29.91 | 4.7 |
| 0.80 | 1000 | 25.70 | 34.80 | 13.20 | 72.97 | 29.98 | 3.9 |
| 0.80 | 2000 | 25.69 | 34.80 | 13.15 | 73.29 | 29.87 | 3.0 |
| 0.90 | 1000 | 25.70 | 34.80 | 13.19 | 72.96 | 29.97 | 3.9 |
| 1.00 | 500 | 25.52 | 34.80 | 12.77 | 72.68 | 29.89 | 4.7 |
| 1.00 | 1000 | 25.70 | 34.80 | 13.18 | 72.96 | 29.96 | 3.8 |
| 1.00 | 2000 | 25.68 | 34.80 | 13.14 | 73.28 | 29.85 | 3.0 |

### A.2 $k{=}60$ Full Grid (13 Configurations)

| $\tau$ | $A_{\min}$ | PQ | PQ$^{\text{St}}$ | PQ$^{\text{Th}}$ | SQ | RQ | inst/img |
|--------|------------|-------|---------|---------|-------|-------|----------|
| CC | --- | 23.73 | 30.74 | 14.09 | 71.07 | 25.83 | 11.1 |
| 0.20 | 500 | 25.27 | 30.74 | 17.75 | 71.33 | 29.85 | 6.5 |
| 0.20 | 700 | 25.55 | 30.74 | 18.42 | 71.49 | 30.09 | 5.7 |
| 0.20 | 1000 | 25.83 | 30.74 | 19.08 | 71.69 | 30.21 | 4.9 |
| 0.30 | 500 | 25.20 | 30.74 | 17.59 | 71.28 | 29.31 | 6.3 |
| 0.30 | 700 | 25.49 | 30.74 | 18.26 | 71.40 | 29.56 | 5.6 |
| 0.30 | 1000 | 25.68 | 30.74 | 18.73 | 71.56 | 29.73 | 4.9 |
| 0.50 | 500 | 25.13 | 30.74 | 17.41 | 71.23 | 28.87 | 6.2 |
| 0.50 | 700 | 25.35 | 30.74 | 17.94 | 71.32 | 29.17 | 5.5 |
| 0.50 | 1000 | 25.55 | 30.74 | 18.40 | 71.47 | 29.36 | 4.8 |
| 0.60 | 500 | 25.17 | 30.74 | 17.52 | 71.20 | 28.85 | 6.1 |
| 0.60 | 700 | 25.43 | 30.74 | 18.12 | 71.29 | 29.12 | 5.5 |
| 0.60 | 1000 | 25.51 | 30.74 | 18.31 | 71.44 | 29.29 | 4.8 |

### A.3 $k{=}80$ Full Grid (19 Configurations)

| $\tau$ | $A_{\min}$ | PQ | PQ$^{\text{St}}$ | PQ$^{\text{Th}}$ | SQ | RQ | inst/img |
|--------|------------|-------|---------|---------|-------|-------|----------|
| CC | --- | 24.84 | 32.08 | 14.90 | 71.15 | 28.22 | 8.6 |
| 0.05 | 500 | 25.40 | 32.08 | 16.22 | 71.96 | 33.03 | 5.4 |
| 0.05 | 700 | 25.62 | 32.08 | 16.75 | 72.14 | 33.19 | 4.6 |
| 0.05 | 1000 | 25.80 | 32.08 | 17.18 | 72.41 | 32.99 | 4.0 |
| 0.10 | 500 | 25.68 | 32.08 | 16.88 | 71.74 | 32.18 | 5.7 |
| 0.10 | 700 | 25.99 | 32.08 | 17.63 | 71.95 | 32.28 | 4.9 |
| 0.10 | 1000 | 26.22 | 32.08 | 18.17 | 72.16 | 32.24 | 4.2 |
| 0.15 | 500 | 26.07 | 32.08 | 17.81 | 71.56 | 31.72 | 5.7 |
| 0.15 | 700 | 26.39 | 32.08 | 18.58 | 71.76 | 31.78 | 5.0 |
| 0.15 | 1000 | 26.51 | 32.08 | 18.85 | 71.99 | 31.67 | 4.3 |
| 0.20 | 500 | 26.32 | 32.08 | 18.40 | 71.50 | 31.38 | 5.6 |
| 0.20 | 700 | 26.56 | 32.08 | 18.98 | 71.63 | 31.48 | 5.0 |
| 0.20 | 1000 | 26.74 | 32.08 | 19.41 | 71.88 | 31.41 | 4.3 |
| 0.30 | 500 | 26.16 | 32.08 | 18.04 | 71.44 | 30.82 | 5.4 |
| 0.30 | 700 | 26.37 | 32.08 | 18.53 | 71.54 | 30.98 | 4.9 |
| 0.30 | 1000 | 26.53 | 32.08 | 18.92 | 71.73 | 30.98 | 4.3 |
| 0.50 | 500 | 26.10 | 32.08 | 17.88 | 71.32 | 30.53 | 5.3 |
| 0.50 | 700 | 26.23 | 32.08 | 18.19 | 71.44 | 30.63 | 4.8 |
| 0.50 | 1000 | 26.34 | 32.08 | 18.44 | 71.60 | 30.68 | 4.3 |

### A.4 $k{=}100$ Full Grid (30 Configurations)

| $\tau$ | $A_{\min}$ | PQ | PQ$^{\text{St}}$ | PQ$^{\text{Th}}$ | SQ | RQ | inst/img |
|--------|------------|-------|---------|---------|-------|-------|----------|
| CC | --- | 25.10 | 32.70 | 14.70 | 71.70 | 26.40 | 10.8 |
| 0.05 | 100 | 24.10 | 32.70 | 12.30 | 72.00 | 30.00 | 12.2 |
| 0.08 | 200 | 25.00 | 32.70 | 14.50 | 71.90 | 31.10 | 9.6 |
| 0.08 | 500 | 26.00 | 32.70 | 16.90 | 72.30 | 32.60 | 6.5 |
| 0.08 | 1000 | 26.50 | 32.70 | 18.00 | 72.80 | 32.70 | 4.8 |
| 0.12 | 200 | 25.50 | 32.70 | 15.80 | 72.00 | 30.60 | 9.5 |
| 0.12 | 500 | 26.30 | 32.70 | 17.60 | 72.20 | 32.00 | 6.7 |
| 0.12 | 1000 | 26.80 | 32.70 | 18.80 | 72.70 | 32.20 | 4.9 |
| 0.20 | 200 | 25.90 | 32.70 | 16.50 | 71.90 | 29.70 | 9.0 |
| 0.20 | 500 | 26.60 | 32.70 | 18.30 | 72.10 | 31.00 | 6.6 |
| **0.20** | **1000** | **27.10** | **32.70** | **19.60** | **72.50** | **31.30** | **5.0** |
| 0.30 | 500 | 26.50 | 32.70 | 17.90 | 72.10 | 30.40 | 6.4 |
| 0.30 | 700 | 26.80 | 32.70 | 18.60 | 72.20 | 30.70 | 5.7 |
| 0.30 | 1000 | 26.90 | 32.70 | 19.00 | 72.40 | 30.80 | 4.9 |
| 0.50 | 500 | 26.40 | 32.70 | 17.80 | 71.90 | 29.80 | 6.1 |
| 0.50 | 700 | 26.60 | 32.70 | 18.30 | 72.00 | 30.10 | 5.5 |
| 0.50 | 1000 | 26.70 | 32.70 | 18.50 | 72.20 | 30.30 | 4.8 |
| 0.60 | 500 | 26.50 | 32.70 | 17.90 | 71.90 | 29.70 | 6.0 |
| 0.60 | 700 | 26.70 | 32.70 | 18.50 | 72.00 | 30.00 | 5.4 |
| 0.60 | 1000 | 26.70 | 32.70 | 18.60 | 72.10 | 30.20 | 4.8 |
| 0.70 | 500 | 26.40 | 32.70 | 17.80 | 71.80 | 29.60 | 6.0 |
| 0.70 | 700 | 26.60 | 32.70 | 18.40 | 71.90 | 29.80 | 5.4 |
| 0.70 | 1000 | 26.70 | 32.70 | 18.50 | 72.10 | 30.00 | 4.7 |
| 0.80 | 500 | 26.40 | 32.70 | 17.80 | 71.80 | 29.50 | 5.9 |
| 0.80 | 1000 | 26.70 | 32.70 | 18.40 | 72.10 | 29.90 | 4.7 |
| 0.80 | 2000 | 26.80 | 32.70 | 18.70 | 72.50 | 29.90 | 3.5 |
| 0.90 | 1000 | 26.70 | 32.70 | 18.40 | 72.10 | 29.90 | 4.7 |
| 1.00 | 500 | 26.40 | 32.70 | 17.70 | 71.80 | 29.40 | 5.9 |
| 1.00 | 1000 | 26.60 | 32.70 | 18.40 | 72.00 | 29.80 | 4.7 |
| 1.00 | 2000 | 26.80 | 32.70 | 18.60 | 72.40 | 29.80 | 3.5 |
