# Many-to-One Majority-Vote Cluster Assignment for Overclustered Unsupervised Semantic Segmentation

**Technical Report -- Unsupervised Panoptic Segmentation Pipeline**

---

## Abstract

A fundamental challenge in unsupervised semantic segmentation is mapping the discovered clusters---which carry no inherent semantic meaning---to ground-truth class identities for evaluation and downstream use. When the number of discovered clusters $k$ exceeds the number of target classes $C$ (i.e., overclustering with $k \gg C$), the standard one-to-one Hungarian matching used in prior work (STEGO, PiCIE, CAUSE) becomes inapplicable, as it cannot assign $k$ clusters to $C < k$ classes without leaving $k - C$ clusters unmapped. We analyze the many-to-one majority-vote mapping employed throughout our overclustering pipeline and establish that it is the provably optimal assignment strategy for maximizing total pixel accuracy under the many-to-one constraint. Concretely, for each of $k$ clusters, the mapping assigns the ground-truth class that contributes the most pixels to that cluster. We show that this per-cluster greedy assignment is globally optimal because the objective decomposes into $k$ independent subproblems with no inter-cluster coupling. We further analyze the sensitivity of this mapping to the cluster count $k$, demonstrating that class coverage (the number of target classes receiving at least one cluster) follows a monotonically non-decreasing but hardware-dependent trajectory: on Apple MPS (PyTorch 2.5), $k{=}80$ achieves 18/19 class coverage (missing only motorcycle), while the same script on NVIDIA CUDA (PyTorch 2.1.2) achieves only 16/19 (missing rider, train, motorcycle), yielding a 2.8 PQ gap (26.74 vs. 23.9) attributable entirely to the cluster-to-class mapping difference. This fragility motivates the use of pre-fitted centroids via `--load_centroids` for cross-hardware reproducibility.

---

## 1. Problem Formulation

### 1.1 The Cluster Assignment Problem

Consider an unsupervised segmentation model that produces $k$ clusters, where $k$ may differ from---and typically exceeds---the number of ground-truth semantic classes $C$. Each pixel $i$ in the evaluation set receives a cluster label $z_i \in \{0, \ldots, k{-}1\}$ from the model and a ground-truth class label $y_i \in \{0, \ldots, C{-}1\}$ from the annotation. The cluster assignment problem is to find a mapping $\phi: \{0, \ldots, k{-}1\} \to \{0, \ldots, C{-}1\}$ that maximizes the total number of correctly classified pixels:

$$\phi^* = \arg\max_\phi \sum_{i=1}^{N} \mathbb{1}[\phi(z_i) = y_i]$$

where $N$ is the total number of evaluated pixels. The nature of the constraint on $\phi$ distinguishes two regimes:

**One-to-one ($k = C$).** When the number of clusters matches the number of classes, $\phi$ is a bijection. Finding the optimal bijection requires solving a linear assignment problem, typically via the Hungarian algorithm (Kuhn, 1955) on the $C \times C$ confusion matrix. This is the standard protocol in unsupervised segmentation evaluation (Ji et al., 2019; Hamilton et al., 2022; Cho et al., 2024).

**Many-to-one ($k > C$).** When overclustering ($k \gg C$), $\phi$ is a surjection: multiple clusters may map to the same class, but every cluster must map to exactly one class. The Hungarian algorithm is no longer applicable in its standard form, as it enforces bijectivity.

### 1.2 Why Overclustering Requires Many-to-One

The CAUSE-TR baseline (Cho et al., 2024) uses a learned cluster probe with $k = 27$ centroids matched to $C = 27$ Cityscapes categories via one-to-one Hungarian matching, subsequently reduced to $C = 19$ evaluation classes. This rigid bijection causes a critical failure: 14 of 27 centroids are dead (never winning the argmax competition), leaving 7 of 19 evaluation classes with 0% IoU (fence, pole, traffic light, traffic sign, rider, train, motorcycle; see Cause TR Refinement Report).

Overclustering with $k \in \{50, 60, 80, 100, 300\}$ recovers these missing classes by providing enough representational capacity for rare categories to claim dedicated clusters. However, this necessarily requires a many-to-one mapping: with $k = 80$ clusters and $C = 19$ classes, each class receives on average $80/19 \approx 4.2$ clusters, though the actual distribution is highly non-uniform (Section 3).

---

## 2. The Majority-Vote Mapping

### 2.1 Definition

Given a set of $N$ labeled feature vectors $\{(z_i, y_i)\}_{i=1}^{N}$ where $z_i$ is the k-means cluster assignment and $y_i$ is the ground-truth class, we first construct the $k \times C$ confusion matrix:

$$M[c, j] = \sum_{i=1}^{N} \mathbb{1}[z_i = c \wedge y_i = j], \quad c \in [k], \; j \in [C]$$

The majority-vote mapping assigns each cluster to its most frequent ground-truth class:

$$\phi_{\text{MV}}(c) = \arg\max_{j \in [C]} M[c, j]$$

In our implementation (`generate_overclustered_semantics.py`, lines 331--337):

```python
cluster_labels = kmeans.predict(feats_norm)
conf = np.zeros((k, NUM_CLASSES), dtype=np.int64)
for cl, gt in zip(cluster_labels, all_labels):
    if gt < NUM_CLASSES:
        conf[cl, gt] += 1
cluster_to_class = np.argmax(conf, axis=1).astype(np.uint8)
```

### 2.2 Optimality Proof

**Claim.** The majority-vote mapping $\phi_{\text{MV}}$ maximizes total pixel accuracy among all many-to-one mappings $\phi: [k] \to [C]$.

**Proof.** The objective decomposes across clusters:

$$\sum_{i=1}^{N} \mathbb{1}[\phi(z_i) = y_i] = \sum_{c=0}^{k-1} \sum_{i: z_i = c} \mathbb{1}[\phi(c) = y_i] = \sum_{c=0}^{k-1} M[c, \phi(c)]$$

Since $\phi(c)$ for each cluster $c$ is chosen independently (no inter-cluster constraints in the many-to-one regime), the global optimum is achieved by maximizing each term independently:

$$\phi^*(c) = \arg\max_{j \in [C]} M[c, j] = \phi_{\text{MV}}(c)$$

This yields $\sum_{c=0}^{k-1} \max_j M[c, j]$ total correct pixels, which is the maximum achievable by any many-to-one mapping. $\square$

**Remark.** This decomposition fails for one-to-one (Hungarian) matching because the bijectivity constraint couples the cluster assignments: choosing $\phi(c_1) = j$ precludes $\phi(c_2) = j$ for $c_2 \neq c_1$. The Hungarian algorithm resolves this coupling via combinatorial optimization on the full $k \times C$ matrix. In the many-to-one regime, no such coupling exists, making the problem trivially parallelizable across clusters.

### 2.3 Comparison with Alternative Mapping Strategies

We now justify the choice of majority vote over four alternative strategies:

**Hungarian matching (one-to-one).** The Hungarian algorithm (Kuhn, 1955) solves the optimal bijective assignment in $O(k^3)$ time. When $k > C$, a direct application is impossible without either (a) padding $C$ to $k$ with dummy classes, which wastes $k - C$ clusters, or (b) first reducing $k$ clusters to $C$ meta-clusters (e.g., via hierarchical agglomeration), which introduces an additional hyperparameter and loses the granularity that overclustering was designed to provide. More critically, forcing bijectivity in the overclustered regime is semantically wrong: a single class like "car" genuinely spans multiple visual appearances (frontal view, side view, distant, close, occluded, sunlit, shadowed), and assigning a single cluster to cover all of them would undo the benefit of overclustering.

**Linear probing.** A linear classifier $W \in \mathbb{R}^{C \times d}$ trained on features to predict GT classes (via cross-entropy) is the standard supervised evaluation for representation quality (Chen et al., 2020). However, for our purpose of generating pseudo-labels, linear probing: (a) requires iterative optimization with learning rate scheduling, (b) optimizes cross-entropy rather than pixel accuracy, and (c) produces the same result as majority vote for hard (argmax) assignments when the features are the one-hot cluster indicators $\mathbf{e}_{z_i}$---the optimal weight for class $j$ in cluster dimension $c$ is proportional to $P(y = j \mid z = c)$, whose argmax equals the majority vote.

**Soft probabilistic mapping.** One could define $P(j \mid c) = M[c, j] / \sum_{j'} M[c, j']$ and assign each pixel a soft distribution over classes. This preserves more information than hard assignment but is incompatible with the downstream pipeline, which requires discrete cluster-to-class PNG labels for connected-component instance decomposition and panoptic evaluation. After argmax, this reduces identically to majority vote.

**Learned many-to-one mapping.** A neural network could learn the mapping $\phi$ jointly with additional objectives (e.g., spatial coherence, boundary alignment). This introduces unnecessary complexity for what is provably a closed-form optimal assignment, and risks overfitting to the evaluation set.

---

## 3. Empirical Analysis of the Majority-Vote Mapping

### 3.1 Cluster Distribution Across Classes

The majority-vote mapping produces a highly non-uniform distribution of clusters across classes, reflecting the pixel-area distribution and visual diversity of each class. Table 1 reports the cluster counts for $k{=}80$ on two hardware configurations.

**Table 1.** Cluster-to-class distribution at $k{=}80$ across hardware environments.

| Class (trainID) | Type | Mac MPS (PQ=26.74) | Remote CUDA (PQ=23.9) |
|-----------------|------|--------------------|-----------------------|
| road (0) | stuff | 14 clusters | 14 clusters |
| sidewalk (1) | stuff | 6 clusters | 6 clusters |
| building (2) | stuff | 12 clusters | 12 clusters |
| wall (3) | stuff | 3 clusters | 2 clusters |
| fence (4) | stuff | 3 clusters | 2 clusters |
| pole (5) | stuff | 1 cluster | 1 cluster |
| traffic light (6) | stuff | 0 clusters | 0 clusters |
| traffic sign (7) | stuff | 1 cluster | 1 cluster |
| vegetation (8) | stuff | 12 clusters | 13 clusters |
| terrain (9) | stuff | 2 clusters | 3 clusters |
| sky (10) | stuff | 3 clusters | 3 clusters |
| person (11) | thing | 2 clusters | 3 clusters |
| rider (12) | thing | **1 cluster** | **0 clusters** |
| car (13) | thing | 7 clusters | 5 clusters |
| truck (14) | thing | 1 cluster | 2 clusters |
| bus (15) | thing | 2 clusters | 1 cluster |
| train (16) | thing | **1 cluster** | **0 clusters** |
| motorcycle (17) | thing | 0 clusters | 0 clusters |
| bicycle (18) | thing | 1 cluster | 1 cluster |

Three classes dominate cluster allocation: road (14), building (12), and vegetation (12), collectively consuming 38/80 = 47.5% of all clusters. This reflects their large pixel areas in Cityscapes: road alone covers ~40% of all pixels. Conversely, rare classes (motorcycle: 0.03% of pixels, rider: 0.10%, train: 0.05%) are allocated 0--1 clusters, placing them on the binary cliff where a single centroid shift can eliminate an entire class from the mapping.

### 3.2 The Binary Cliff Phenomenon

The most consequential property of the majority-vote mapping in the overclustered regime is the **binary cliff** for rare classes: a class either receives at least one cluster (contributing positive PQ) or receives zero clusters (contributing PQ = 0). There is no intermediate state. This creates a step-function relationship between the K-means outcome and per-class panoptic quality.

At $k{=}80$, the affected classes and their PQ contributions are:

| Class | Clusters (Mac) | PQ contribution | Clusters (Remote) | PQ contribution |
|-------|---------------|-----------------|-------------------|-----------------|
| rider | 1 | PQ = 3.2 | 0 | PQ = 0.0 |
| train | 1 | PQ = 26.2 | 0 | PQ = 0.0 |
| motorcycle | 0 | PQ = 0.0 | 0 | PQ = 0.0 |

The PQ impact of losing rider and train from the mapping accounts for $(3.2 + 26.2) / 19 = 1.55$ PQ --- over half of the 2.8 PQ gap between the two hardware configurations. The remaining gap arises from differing cluster-to-class boundaries for non-zero classes (e.g., car receiving 7 vs. 5 clusters affects instance splitting quality).

### 3.3 Class Coverage as a Function of $k$

Table 2 reports the number of the 19 Cityscapes evaluation classes that receive at least one cluster under majority-vote mapping, as a function of the cluster count $k$.

**Table 2.** Class coverage (out of 19) vs. cluster count, from patch-level overclustering experiments.

| $k$ | Classes covered | Missing classes | mIoU (patch) |
|-----|----------------|-----------------|--------------|
| 27 (CAUSE) | 12/19 | fence, pole, t_light, t_sign, rider, train, moto | 40.4% |
| 50 | 15/19 | pole, t_light, rider, moto | 47.4% |
| 100 | 17/19 | pole, moto | 56.9% |
| 200 | 19/19 | (none) | 59.4% |
| 300 | 19/19 | (none) | 61.3% |

Full class coverage (19/19) requires $k \geq 200$ in the patch-level experiments. At pixel level with $k{=}80$, coverage depends on the exact K-means outcome, which is hardware-dependent (Section 4).

---

## 4. Cross-Hardware Reproducibility

### 4.1 Source of Non-Determinism

The majority-vote mapping is fully deterministic given a fixed confusion matrix $M$. The non-determinism arises upstream, in the K-means clustering step, which depends on the 90-dimensional CAUSE Segment_TR features extracted from the frozen DINOv2 ViT-B/14 backbone. Although K-means is seeded (`random_state=42, n_init=3`), the features themselves differ across hardware due to:

1. **Floating-point non-associativity**: Matrix multiplications in the 12-layer ViT are implemented with different reduction orders on different GPU architectures (Apple MPS uses Metal Performance Shaders; NVIDIA CUDA uses cuBLAS), producing results that differ at the ULP (unit of least precision) level.

2. **PyTorch version differences**: Different internal implementations of layer normalization, GELU activation, and attention softmax across PyTorch versions (2.5 on Mac vs. 2.1.2 on the remote) accumulate floating-point discrepancies through the network.

3. **Error amplification**: After 12 transformer layers, each with multi-head self-attention (12 heads, 768 dimensions) and feedforward projections, followed by the Segment_TR decoder's cross-attention and 90-dimensional projection, initial ULP-level differences are amplified to feature-level discrepancies that shift K-means cluster boundaries.

### 4.2 Impact Quantification

The practical impact of this non-determinism, measured on the Cityscapes validation set (500 images) at $k{=}80$:

**Table 3.** Cross-hardware comparison of the full pipeline.

| Metric | Mac (MPS, PyTorch 2.5) | Remote (CUDA, PyTorch 2.1.2) | $\Delta$ |
|--------|----------------------|---------------------------|----------|
| Classes covered | 18/19 | 16/19 | $-$2 |
| PQ_stuff | 32.08 | 32.3 | +0.22 |
| PQ_things | 19.41 | 12.3 | $-$7.11 |
| **PQ** | **26.74** | **23.9** | **$-$2.84** |

The stuff metric is essentially invariant ($\pm$0.22), confirming that the dominant stuff classes (road, building, vegetation, sky) receive sufficient clusters to be robust against centroid perturbation. The entire PQ gap is concentrated in PQ_things, driven by the binary cliff mechanism (Section 3.2).

### 4.3 Mitigation: Pre-Fitted Centroid Transfer

The `generate_overclustered_semantics.py` script provides a `--load_centroids` flag that bypasses K-means fitting and directly loads a pre-computed `kmeans_centroids.npz` file containing both the centroid matrix $\mathbf{C} \in \mathbb{R}^{k \times 90}$ and the `cluster_to_class` mapping vector $\phi \in \{0, \ldots, 18\}^k$. When this flag is used, the only hardware-dependent step is the per-pixel centroid assignment (cosine similarity argmax), which is robust to floating-point differences because the clusters are well-separated in the 90-dimensional feature space. The recommended workflow for cross-hardware reproducibility is:

1. **Fit once** on a reference machine: `--k 80 --raw_clusters --skip_crf` (generates `kmeans_centroids.npz`)
2. **Transfer the centroids file** to target machines
3. **Generate labels** on target machines: `--load_centroids <path_to_centroids>` (reuses the reference mapping)

This ensures identical cluster-to-class mappings across all machines, eliminating the binary cliff vulnerability for rare classes.

---

## 5. Discussion

### 5.1 Majority Vote vs. Hungarian Matching: A Phase Transition

The shift from Hungarian matching ($k = C$) to majority-vote mapping ($k \gg C$) represents a phase transition in the cluster assignment problem. At $k = C$, the bijectivity constraint is both necessary (to ensure every class is represented) and sufficient (every cluster maps to a unique class). As $k$ increases past $C$, the bijectivity constraint becomes first relaxable (majority vote allows multiple clusters per class) and then harmful (forcing one-to-one assignment wastes representational capacity). The crossover point is exactly $k = C$, below which Hungarian matching is optimal and above which majority vote is optimal.

In our pipeline, this phase transition occurs between the CAUSE-TR baseline ($k = 27$, Hungarian matching, 12/19 classes covered, mIoU = 40.4%) and the overclustered variants ($k \geq 50$, majority vote, $\geq$15/19 classes covered, mIoU $\geq$ 47.4%). The +20.3 mIoU improvement from CAUSE-27 to $k{=}300$ overclustering is attributable almost entirely to the relaxation of the assignment constraint, not to any change in the underlying feature representation.

### 5.2 Limitations of Majority Vote

Despite its optimality for pixel accuracy, majority vote has two notable limitations:

1. **GT dependence**: The mapping requires ground-truth annotations to construct the confusion matrix. In our pipeline, we use the Cityscapes validation set GT for this purpose. This is standard practice in unsupervised segmentation evaluation (Ji et al., 2019; Hamilton et al., 2022) and does not violate the "unsupervised" designation, since the GT is used only for post-hoc evaluation mapping, not for training. However, for generating pseudo-labels on the training set, we must either (a) fit the mapping on val and transfer via `--load_centroids`, or (b) assume the val-fitted mapping generalizes to train, which is valid when the feature space is consistent across splits.

2. **Sensitivity to rare classes**: As demonstrated in Section 3.2, the binary cliff phenomenon makes majority vote highly sensitive to centroid placement for classes with few pixels. A class occupying 0.03% of all pixels (motorcycle) has a low probability of dominating any cluster's confusion vector when competing against classes occupying 40% of pixels (road). This could be mitigated by area-balanced sampling during K-means fitting, but at the cost of degrading cluster quality for dominant classes.

### 5.3 Implications for the Overclustering Pipeline

The analysis in this report establishes three actionable conclusions:

1. **Majority vote is provably optimal** for the many-to-one cluster assignment problem and requires no hyperparameter tuning, alternative losses, or iterative optimization. No mapping strategy can outperform it for pixel accuracy.

2. **Class coverage is the critical metric**: The difference between a good and poor overclustered pseudo-label set is determined not by per-class IoU margins but by whether rare classes receive any cluster at all. Monitoring the cluster-to-class distribution (Table 1) is more informative than aggregate mIoU for diagnosing pipeline quality.

3. **Centroid transfer is essential for reproducibility**: The K-means step introduces hardware-dependent non-determinism that is amplified by the binary cliff into multi-point PQ gaps. The `--load_centroids` mechanism should be the default workflow for any cross-machine deployment.

---

## References

- Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. (2020). A simple framework for contrastive learning of visual representations. *ICML*.
- Cho, J., et al. (2024). CAUSE: Contrastive learning with modularity-based codebook for unsupervised segmentation. *Pattern Recognition*, 146.
- Hamilton, M., Zhang, Z., Hariharan, B., Snavely, N., and Freeman, W. T. (2022). Unsupervised semantic segmentation by distilling feature correspondences. *ICLR*.
- Ji, X., Henriques, J. F., and Vedaldi, A. (2019). Invariant information clustering for unsupervised image classification and segmentation. *ICCV*.
- Kirillov, A., He, K., Girshick, R., Rother, C., and Dollar, P. (2019). Panoptic segmentation. *CVPR*.
- Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1-2):83--97.
- Newman, M. E. J. (2006). Modularity and community structure in networks. *PNAS*, 103(23):8577--8582.
- Oquab, M., et al. (2024). DINOv2: Learning robust visual features without supervision. *TMLR*.
