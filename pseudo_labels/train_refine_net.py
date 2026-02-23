#!/usr/bin/env python3
"""Train CSCMRefineNet: Coupled-State Cross-Modal Mamba Refinement Network.

Trains the semantic pseudo-label refiner using self-supervised losses:
  1. Cross-view consistency (weak/strong augmentation agreement)
  2. Depth-boundary alignment (semantic edges ↔ depth discontinuities)
  3. Feature-prototype consistency (compact DINOv2 clusters per class)
  4. Entropy minimization (encourage confident predictions)

Usage:
    python mbps_pytorch/train_refine_net.py \
        --cityscapes_root /path/to/cityscapes \
        --output_dir checkpoints/refine_net \
        --num_epochs 50 --batch_size 4 --device auto

References:
    - UniMatch V2 (TPAMI 2025): Cross-view consistency
    - DepthG (CVPR 2024): Depth-boundary alignment
    - CAUSE (Pattern Recognition 2024): Feature-prototype concept
    - Goodfellow "Deep Learning" Ch. 19: Entropy minimization
"""

import argparse
import json
import logging
import math
import os
import time
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys; sys.path.insert(0, os.path.dirname(__file__))
from refine_net import CSCMRefineNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Patch grid dimensions for DINOv2 ViT-B/14 at 448×896 input
PATCH_H, PATCH_W = 32, 64
NUM_CLASSES = 27


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PseudoLabelDataset(Dataset):
    """Load pre-computed CAUSE semantics, DINOv2 features, and SPIdepth depth."""

    def __init__(
        self,
        cityscapes_root: str,
        split: str = "train",
        semantic_subdir: str = "pseudo_semantic_cause_crf",
        feature_subdir: str = "dinov2_features",
        depth_subdir: str = "depth_spidepth",
        logits_subdir: str = None,
    ):
        self.root = cityscapes_root
        self.split = split
        self.semantic_subdir = semantic_subdir
        self.feature_subdir = feature_subdir
        self.depth_subdir = depth_subdir
        self.logits_subdir = logits_subdir

        # Find all images and extract stems
        img_dir = os.path.join(cityscapes_root, "leftImg8bit", split)
        self.entries = []
        for city in sorted(os.listdir(img_dir)):
            city_path = os.path.join(img_dir, city)
            if not os.path.isdir(city_path):
                continue
            for fname in sorted(os.listdir(city_path)):
                if not fname.endswith("_leftImg8bit.png"):
                    continue
                stem = fname.replace("_leftImg8bit.png", "")
                self.entries.append({"stem": stem, "city": city})

        log.info(f"PseudoLabelDataset: {len(self.entries)} images ({split})")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        stem, city = entry["stem"], entry["city"]

        # Load DINOv2 features: (2048, 768) float16 → (768, 32, 64) float32
        feat_path = os.path.join(
            self.root, self.feature_subdir, self.split, city,
            f"{stem}_leftImg8bit.npy",
        )
        features = np.load(feat_path).astype(np.float32)  # (2048, 768)
        features = features.reshape(PATCH_H, PATCH_W, -1)  # (32, 64, 768)
        features = features.transpose(2, 0, 1)  # (768, 32, 64)

        # Load depth: (512, 1024) float32 → downsample to (1, 32, 64)
        depth_path = os.path.join(
            self.root, self.depth_subdir, self.split, city, f"{stem}.npy",
        )
        depth_full = np.load(depth_path)  # (512, 1024)
        depth_patch = torch.from_numpy(depth_full).unsqueeze(0).unsqueeze(0)
        depth_patch = F.interpolate(
            depth_patch, size=(PATCH_H, PATCH_W),
            mode="bilinear", align_corners=False,
        ).squeeze(0)  # (1, 32, 64)
        depth_np = depth_patch.numpy()

        # Compute Sobel gradients on depth at patch resolution
        depth_grads = self._sobel_gradients(depth_np[0])  # (2, 32, 64)

        # Load semantics: (1024, 2048) uint8 → one-hot (27, 32, 64)
        if self.logits_subdir is not None:
            logits_path = os.path.join(
                self.root, self.logits_subdir, self.split, city,
                f"{stem}_logits.pt",
            )
            if os.path.exists(logits_path):
                cause_logits = torch.load(logits_path, weights_only=True).float()
                # Expected: (27, logits_h, logits_w) → resize to (27, 32, 64)
                if cause_logits.shape[1:] != (PATCH_H, PATCH_W):
                    cause_logits = F.interpolate(
                        cause_logits.unsqueeze(0),
                        size=(PATCH_H, PATCH_W),
                        mode="bilinear", align_corners=False,
                    ).squeeze(0)
                # Stored values are softmax probabilities — convert to
                # log-space so they behave as proper logits for the loss
                # functions (which apply F.softmax internally).
                cause_logits = torch.log(cause_logits.clamp(min=1e-7))
                cause_logits = cause_logits.numpy()
            else:
                cause_logits = self._load_onehot_semantics(city, stem)
        else:
            cause_logits = self._load_onehot_semantics(city, stem)

        return {
            "cause_logits": torch.from_numpy(cause_logits).float(),
            "dinov2_features": torch.from_numpy(features).float(),
            "depth": torch.from_numpy(depth_np).float(),
            "depth_grads": torch.from_numpy(depth_grads).float(),
            "stem": stem,
            "city": city,
        }

    def _load_onehot_semantics(self, city, stem):
        """Load argmax PNG and convert to smoothed one-hot at patch resolution."""
        sem_path = os.path.join(
            self.root, self.semantic_subdir, self.split, city, f"{stem}.png",
        )
        sem_full = np.array(Image.open(sem_path))  # (1024, 2048) uint8

        # Downsample to patch resolution via nearest neighbor
        sem_pil = Image.fromarray(sem_full)
        sem_patch = np.array(
            sem_pil.resize((PATCH_W, PATCH_H), Image.NEAREST)
        )  # (32, 64)

        # Convert to smoothed one-hot (label smoothing = 0.1)
        onehot = np.zeros((NUM_CLASSES, PATCH_H, PATCH_W), dtype=np.float32)
        smooth = 0.1
        onehot[:] = smooth / NUM_CLASSES
        for c in range(NUM_CLASSES):
            mask = sem_patch == c
            onehot[c][mask] = 1.0 - smooth + smooth / NUM_CLASSES

        # Convert to log-space (consistent with .pt logits path)
        return np.log(np.clip(onehot, 1e-7, None))

    @staticmethod
    def _sobel_gradients(depth_2d):
        """Compute Sobel gradients. depth_2d: (H, W) → (2, H, W)."""
        d = torch.from_numpy(depth_2d).unsqueeze(0).unsqueeze(0).float()
        # Sobel kernels
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32).reshape(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=torch.float32).reshape(1, 1, 3, 3)
        grad_x = F.conv2d(d, kx, padding=1).squeeze()  # (H, W)
        grad_y = F.conv2d(d, ky, padding=1).squeeze()  # (H, W)
        return torch.stack([grad_x, grad_y], dim=0).numpy()  # (2, H, W)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def apply_augmentation(batch, mode="weak"):
    """Apply augmentation to a batch dict (in-place friendly).

    Since DINOv2 features are pre-extracted, augmentations are spatial only:
      - weak: random horizontal flip
      - strong: horizontal flip + Gaussian noise on features + spatial dropout
    """
    aug_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

    if mode in ("weak", "strong"):
        # Random horizontal flip (per sample)
        B = aug_batch["cause_logits"].shape[0]
        for i in range(B):
            if torch.rand(1).item() > 0.5:
                aug_batch["cause_logits"][i] = torch.flip(
                    aug_batch["cause_logits"][i], [-1])
                aug_batch["dinov2_features"][i] = torch.flip(
                    aug_batch["dinov2_features"][i], [-1])
                aug_batch["depth"][i] = torch.flip(
                    aug_batch["depth"][i], [-1])
                aug_batch["depth_grads"][i] = torch.flip(
                    aug_batch["depth_grads"][i], [-1])
                # Flip sign of horizontal gradient
                aug_batch["depth_grads"][i, 0] *= -1

    if mode == "strong":
        # Gaussian noise on features
        noise = torch.randn_like(aug_batch["dinov2_features"]) * 0.05
        aug_batch["dinov2_features"] = aug_batch["dinov2_features"] + noise

        # Spatial dropout: zero out random 10% of spatial positions
        B, C, H, W = aug_batch["dinov2_features"].shape
        mask = (torch.rand(B, 1, H, W, device=aug_batch["dinov2_features"].device)
                > 0.1).float()
        aug_batch["dinov2_features"] = aug_batch["dinov2_features"] * mask

    return aug_batch


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def cross_view_consistency_loss(logits_weak, logits_strong, threshold=0.9):
    """Enforce prediction agreement between weak and strong augmented views.

    Weak predictions (detached) provide pseudo-targets for strong predictions.
    Only high-confidence pixels supervise.

    Reference: UniMatch V2 (TPAMI 2025)
    """
    probs_weak = F.softmax(logits_weak.detach(), dim=1)
    max_probs, pseudo_targets = probs_weak.max(dim=1)  # (B, H, W)
    mask = max_probs > threshold

    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits_strong.device)

    loss = F.cross_entropy(logits_strong, pseudo_targets, reduction="none")
    loss = (loss * mask.float()).sum() / mask.sum()
    return loss


def depth_boundary_alignment_loss(logits, depth, sigma=0.05):
    """Encourage label consistency between depth-similar neighbors.

    Pixels with similar depth should have similar predictions.
    Pixels across depth discontinuities are allowed to disagree.

    Reference: DepthG (CVPR 2024)
    """
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)

    # Horizontal neighbors
    depth_diff_h = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
    weight_h = torch.exp(-depth_diff_h ** 2 / (2 * sigma ** 2))
    prob_diff_h = (probs[:, :, :, 1:] - probs[:, :, :, :-1]) ** 2
    loss_h = (weight_h * prob_diff_h.sum(dim=1, keepdim=True)).mean()

    # Vertical neighbors
    depth_diff_v = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])
    weight_v = torch.exp(-depth_diff_v ** 2 / (2 * sigma ** 2))
    prob_diff_v = (probs[:, :, 1:, :] - probs[:, :, :-1, :]) ** 2
    loss_v = (weight_v * prob_diff_v.sum(dim=1, keepdim=True)).mean()

    return loss_h + loss_v


def feature_prototype_loss(logits, dinov2_features, temperature=0.1):
    """Encourage compact DINOv2 feature clusters per predicted class.

    Compute per-class prototypes (weighted mean DINOv2 feature), then
    maximize cosine similarity of each pixel to its predicted prototype.
    """
    probs = F.softmax(logits / temperature, dim=1)  # (B, C, H, W)
    B, C, H, W = probs.shape
    D = dinov2_features.shape[1]

    probs_flat = probs.reshape(B, C, H * W)      # (B, C, N)
    feats_flat = dinov2_features.reshape(B, D, H * W)  # (B, D, N)

    # Weighted prototypes per class: (B, D, C)
    prototypes = torch.bmm(feats_flat, probs_flat.permute(0, 2, 1))
    weights = probs_flat.sum(dim=2).unsqueeze(1)  # (B, 1, C)
    prototypes = prototypes / (weights + 1e-6)

    # Cosine similarity
    prototypes_norm = F.normalize(prototypes, dim=1)  # (B, D, C)
    feats_norm = F.normalize(feats_flat, dim=1)       # (B, D, N)
    sim = torch.bmm(prototypes_norm.permute(0, 2, 1), feats_norm)  # (B, C, N)

    # Loss: high-probability pixels should be close to their prototype
    loss = -(probs_flat * sim).sum() / (B * H * W)
    return loss


def entropy_loss(logits):
    """Minimize prediction entropy to encourage confident assignments.

    Reference: Goodfellow "Deep Learning" Ch. 19
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1).mean()
    return entropy


class RefineNetLoss(nn.Module):
    """Combined loss for CSCMRefineNet.

    Uses cross-entropy from CAUSE hard labels (not KL divergence) with
    cosine warmdown: supervision starts strong, drops to near-zero by
    end of training so self-supervised losses dominate.

    Self-supervised losses:
      - depth_boundary_alignment: spatial smoothness weighted by depth
      - feature_prototype: DINOv2 cluster compactness
      - entropy: prediction confidence
    """

    def __init__(
        self,
        lambda_distill: float = 1.0,
        lambda_distill_min: float = 0.0,
        lambda_align: float = 5.0,
        lambda_proto: float = 0.5,
        lambda_ent: float = 0.3,
        label_smoothing: float = 0.2,
    ):
        super().__init__()
        self.lambda_distill = lambda_distill
        self.lambda_distill_min = lambda_distill_min
        self.lambda_align = lambda_align
        self.lambda_proto = lambda_proto
        self.lambda_ent = lambda_ent
        self.label_smoothing = label_smoothing
        self._distill_scale = 1.0

    def set_epoch(self, epoch: int, total_epochs: int):
        """Cosine warmdown: fast decay in middle epochs."""
        progress = (epoch - 1) / max(total_epochs - 1, 1)
        hi, lo = self.lambda_distill, self.lambda_distill_min
        # Cosine decay: stays high initially, drops fast, then flattens near min
        self._distill_scale = lo + 0.5 * (hi - lo) * (1 + math.cos(math.pi * progress))

    def forward(self, logits, cause_logits, dinov2_features, depth):
        """
        Args:
            logits: (B, 27, H, W) model output logits
            cause_logits: (B, 27, H, W) original CAUSE log-probabilities
            dinov2_features: (B, 768, H, W) original features
            depth: (B, 1, H, W) depth map
        Returns:
            total_loss, loss_dict
        """
        losses = {}
        eff_distill = self._distill_scale

        # Cross-entropy from CAUSE hard labels with label smoothing
        # (simpler, more stable than KL; label smoothing gives model freedom)
        if eff_distill > 0:
            cause_labels = cause_logits.argmax(dim=1).detach()  # (B, H, W)
            B, C, H, W = logits.shape
            l_distill = F.cross_entropy(
                logits, cause_labels, label_smoothing=self.label_smoothing,
            )
            losses["distill"] = l_distill
        else:
            l_distill = 0.0

        if self.lambda_align > 0:
            l_align = depth_boundary_alignment_loss(logits, depth)
            losses["align"] = l_align
        else:
            l_align = 0.0

        if self.lambda_proto > 0:
            l_proto = feature_prototype_loss(logits, dinov2_features)
            losses["proto"] = l_proto
        else:
            l_proto = 0.0

        if self.lambda_ent > 0:
            l_ent = entropy_loss(logits)
            losses["entropy"] = l_ent
        else:
            l_ent = 0.0

        total = (eff_distill * l_distill
                 + self.lambda_align * l_align
                 + self.lambda_proto * l_proto
                 + self.lambda_ent * l_ent)
        losses["total"] = total
        losses["eff_distill_w"] = eff_distill

        return total, losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

_CS_ID_TO_TRAIN = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18,
}
_STUFF_IDS = set(range(0, 11))   # trainIDs 0-10
_THING_IDS = set(range(11, 19))  # trainIDs 11-18
_CS_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

# CAUSE 27-class → 19 trainID (validated mapping from evaluate_cascade_pseudolabels.py)
_CAUSE27_TO_TRAINID = np.full(256, 255, dtype=np.uint8)
for _c27, _t19 in {
    0: 0, 1: 1, 2: 255, 3: 255, 4: 2, 5: 3, 6: 4,
    7: 255, 8: 255, 9: 255, 10: 5, 11: 5, 12: 6, 13: 7,
    14: 8, 15: 9, 16: 10, 17: 11, 18: 12, 19: 13, 20: 14,
    21: 15, 22: 13, 23: 14, 24: 16, 25: 17, 26: 18,
}.items():
    _CAUSE27_TO_TRAINID[_c27] = _t19


def evaluate_panoptic(model, val_loader, device, cityscapes_root,
                      eval_hw=(512, 1024), cc_min_area=50):
    """Evaluate refined semantics with full panoptic metrics.

    Computes PQ, PQ_stuff, PQ_things, SQ, RQ and mIoU on the val set.
    Thing instances are derived via connected components of the semantic map.
    """
    from scipy import ndimage
    from collections import defaultdict

    gt_dir = os.path.join(cityscapes_root, "gtFine", "val")
    H, W = eval_hw
    num_cls = 19

    # Accumulators
    confusion = np.zeros((num_cls, num_cls), dtype=np.int64)
    tp = np.zeros(num_cls)
    fp = np.zeros(num_cls)
    fn = np.zeros(num_cls)
    iou_sum = np.zeros(num_cls)
    changed_pixels = 0
    total_pixels = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", ncols=100, leave=False):
            logits = model(
                batch["dinov2_features"].to(device),
                batch["depth"].to(device),
                batch["depth_grads"].to(device),
            )
            pred_27 = logits.argmax(dim=1).cpu().numpy()  # (B, 32, 64)
            orig_27 = batch["cause_logits"].argmax(dim=1).cpu().numpy()

            for i in range(pred_27.shape[0]):
                city, stem = batch["city"][i], batch["stem"][i]

                # Track prediction changes
                changed_pixels += (pred_27[i] != orig_27[i]).sum()
                total_pixels += pred_27[i].size

                # Map 27-class → 19 trainID and upsample to eval resolution
                pred_tid_patch = _CAUSE27_TO_TRAINID[pred_27[i]]
                pred_sem = np.array(
                    Image.fromarray(pred_tid_patch).resize((W, H), Image.NEAREST)
                )

                # Load GT semantic
                gt_path = os.path.join(
                    gt_dir, city, f"{stem}_gtFine_labelIds.png")
                if not os.path.exists(gt_path):
                    continue
                gt_raw = np.array(Image.open(gt_path))
                gt_sem = np.full_like(gt_raw, 255, dtype=np.uint8)
                for raw_id, tid in _CS_ID_TO_TRAIN.items():
                    gt_sem[gt_raw == raw_id] = tid
                if gt_sem.shape != (H, W):
                    gt_sem = np.array(
                        Image.fromarray(gt_sem).resize((W, H), Image.NEAREST))

                # mIoU confusion matrix
                valid = (gt_sem < num_cls) & (pred_sem < num_cls)
                if valid.sum() > 0:
                    np.add.at(confusion, (gt_sem[valid], pred_sem[valid]), 1)

                # --- Panoptic evaluation ---
                # Build predicted panoptic map
                pred_pan = np.zeros((H, W), dtype=np.int32)
                pred_segments = {}
                nxt = 1

                # Stuff segments
                for cls in _STUFF_IDS:
                    mask = pred_sem == cls
                    if mask.sum() < 64:
                        continue
                    pred_pan[mask] = nxt
                    pred_segments[nxt] = cls
                    nxt += 1

                # Thing segments via connected components
                for cls in _THING_IDS:
                    cls_mask = pred_sem == cls
                    if cls_mask.sum() < cc_min_area:
                        continue
                    labeled, n_cc = ndimage.label(cls_mask)
                    for comp in range(1, n_cc + 1):
                        cmask = labeled == comp
                        if cmask.sum() < cc_min_area:
                            continue
                        pred_pan[cmask] = nxt
                        pred_segments[nxt] = cls
                        nxt += 1

                # Build GT panoptic map
                gt_pan = np.zeros((H, W), dtype=np.int32)
                gt_segments = {}
                gt_nxt = 1

                for cls in _STUFF_IDS:
                    mask = gt_sem == cls
                    if mask.sum() < 64:
                        continue
                    gt_pan[mask] = gt_nxt
                    gt_segments[gt_nxt] = cls
                    gt_nxt += 1

                gt_inst_path = os.path.join(
                    gt_dir, city, f"{stem}_gtFine_instanceIds.png")
                if os.path.exists(gt_inst_path):
                    gt_inst = np.array(Image.open(gt_inst_path), dtype=np.int32)
                    if gt_inst.shape != (H, W):
                        gt_inst = np.array(
                            Image.fromarray(gt_inst).resize((W, H), Image.NEAREST))
                    for uid in np.unique(gt_inst):
                        if uid < 1000:
                            continue
                        raw_cls = uid // 1000
                        if raw_cls not in _CS_ID_TO_TRAIN:
                            continue
                        tid = _CS_ID_TO_TRAIN[raw_cls]
                        if tid not in _THING_IDS:
                            continue
                        mask = gt_inst == uid
                        if mask.sum() < 10:
                            continue
                        gt_pan[mask] = gt_nxt
                        gt_segments[gt_nxt] = tid
                        gt_nxt += 1

                # Match segments per category
                gt_by_cat = defaultdict(list)
                for sid, cat in gt_segments.items():
                    gt_by_cat[cat].append(sid)
                pred_by_cat = defaultdict(list)
                for sid, cat in pred_segments.items():
                    pred_by_cat[cat].append(sid)

                matched_pred = set()
                for cat in range(num_cls):
                    for gt_id in gt_by_cat.get(cat, []):
                        gt_mask = gt_pan == gt_id
                        best_iou, best_pid = 0.0, None
                        for pid in pred_by_cat.get(cat, []):
                            if pid in matched_pred:
                                continue
                            inter = np.sum(gt_mask & (pred_pan == pid))
                            union = np.sum(gt_mask | (pred_pan == pid))
                            if union == 0:
                                continue
                            iou_val = inter / union
                            if iou_val > best_iou:
                                best_iou, best_pid = iou_val, pid
                        if best_iou > 0.5 and best_pid is not None:
                            tp[cat] += 1
                            iou_sum[cat] += best_iou
                            matched_pred.add(best_pid)
                        else:
                            fn[cat] += 1

                    for pid in pred_by_cat.get(cat, []):
                        if pid not in matched_pred:
                            fp[cat] += 1

    # Compute metrics
    intersection = np.diag(confusion)
    union = confusion.sum(1) + confusion.sum(0) - intersection
    iou = np.where(union > 0, intersection / union, 0.0)
    miou = iou[union > 0].mean() * 100

    all_pq, stuff_pq, thing_pq = [], [], []
    per_class = {}
    for c in range(num_cls):
        t, f_p, f_n, s = tp[c], fp[c], fn[c], iou_sum[c]
        if t + f_p + f_n > 0:
            sq = s / (t + 1e-8)
            rq = t / (t + 0.5 * f_p + 0.5 * f_n)
            pq = sq * rq
        else:
            sq = rq = pq = 0.0
        per_class[_CS_CLASS_NAMES[c]] = round(pq * 100, 2)
        if t + f_p + f_n > 0:
            all_pq.append(pq)
            (stuff_pq if c in _STUFF_IDS else thing_pq).append(pq)

    pq_all = float(np.mean(all_pq)) * 100 if all_pq else 0.0
    pq_stuff = float(np.mean(stuff_pq)) * 100 if stuff_pq else 0.0
    pq_things = float(np.mean(thing_pq)) * 100 if thing_pq else 0.0

    change_pct = changed_pixels / max(total_pixels, 1) * 100

    model.train()
    return {
        "PQ": round(pq_all, 2),
        "PQ_stuff": round(pq_stuff, 2),
        "PQ_things": round(pq_things, 2),
        "mIoU": round(miou, 2),
        "changed_pct": round(change_pct, 2),
        "per_class_pq": per_class,
        "per_class_iou": {_CS_CLASS_NAMES[i]: round(iou[i] * 100, 2) for i in range(num_cls)},
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log.info(f"Device: {device}")

    # Model
    model = CSCMRefineNet(
        num_classes=NUM_CLASSES,
        feature_dim=768,
        bridge_dim=args.bridge_dim,
        num_blocks=args.num_blocks,
        block_type=args.block_type,
        layer_type=args.layer_type,
        scan_mode=args.scan_mode,
        coupling_strength=args.coupling_strength,
        d_state=args.d_state,
        chunk_size=args.chunk_size,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"CSCMRefineNet: {total_params:,} parameters")

    # Datasets
    train_dataset = PseudoLabelDataset(
        args.cityscapes_root, split="train",
        semantic_subdir=args.semantic_subdir,
        logits_subdir=args.logits_subdir,
    )
    val_dataset = PseudoLabelDataset(
        args.cityscapes_root, split="val",
        semantic_subdir=args.semantic_subdir,
        logits_subdir=args.logits_subdir,
    )

    pin_mem = device.type == "cuda"  # MPS doesn't support pin_memory
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_mem,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_mem,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01)

    # Loss
    loss_fn = RefineNetLoss(
        lambda_distill=args.lambda_distill,
        lambda_distill_min=args.lambda_distill_min,
        lambda_align=args.lambda_align,
        lambda_proto=args.lambda_proto,
        lambda_ent=args.lambda_ent,
        label_smoothing=args.label_smoothing,
    )

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config = vars(args)
    config["total_params"] = total_params
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Mixed precision setup
    use_amp = args.amp
    if use_amp:
        if device.type == "mps":
            amp_dtype = torch.bfloat16
            amp_device = "mps"
        elif device.type == "cuda":
            amp_dtype = torch.bfloat16
            amp_device = "cuda"
        else:
            use_amp = False
            amp_dtype = torch.float32
            amp_device = "cpu"
    else:
        amp_dtype = torch.float32
        amp_device = device.type
    if use_amp:
        log.info(f"Mixed precision: {amp_dtype} on {amp_device}")

    # Training
    best_pq = 0.0
    log.info(f"Training for {args.num_epochs} epochs, "
             f"{len(train_loader)} batches/epoch")

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        loss_fn.set_epoch(epoch, args.num_epochs)
        epoch_losses = {"total": 0, "distill": 0, "align": 0, "proto": 0, "entropy": 0}
        num_batches = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}",
                    ncols=120, leave=True)
        for batch_idx, batch in enumerate(pbar):
            # Move tensors to device
            cause_logits = batch["cause_logits"].to(device)
            dinov2_features = batch["dinov2_features"].to(device)
            depth = batch["depth"].to(device)
            depth_grads = batch["depth_grads"].to(device)

            # Single forward pass (no dual augmentation)
            with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
                logits = model(dinov2_features, depth, depth_grads)
                total_loss, loss_dict = loss_fn(logits, cause_logits, dinov2_features, depth)

            # Backward (outside autocast — gradients computed in float32)
            optimizer.zero_grad()
            total_loss.backward()

            # Sanitize NaN/Inf gradients (GatedDeltaNet backward on MPS
            # produces NaN in the SSD exp-cumsum chain)
            nan_grad_count = 0
            for p in model.parameters():
                if p.grad is not None and not p.grad.isfinite().all():
                    nan_grad_count += p.grad.isfinite().logical_not().sum().item()
                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            if nan_grad_count > 0 and (batch_idx + 1) % 50 == 0:
                log.warning(f"  Replaced {nan_grad_count} NaN/Inf gradient elements")

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Accumulate
            for k, v in loss_dict.items():
                if k not in epoch_losses:
                    continue
                if isinstance(v, torch.Tensor):
                    epoch_losses[k] += v.item()
                else:
                    epoch_losses[k] += float(v)
            num_batches += 1

            # Update progress bar
            pbar.set_postfix(
                loss=f"{loss_dict['total'].item():.4f}",
                dist=f"{epoch_losses['distill']/num_batches:.4f}",
                align=f"{epoch_losses['align']/num_batches:.4f}",
                proto=f"{epoch_losses['proto']/num_batches:.4f}",
                ent=f"{epoch_losses['entropy']/num_batches:.4f}",
            )

        pbar.close()
        scheduler.step()

        # Epoch summary
        dt = time.time() - t0
        avg_losses = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}
        log.info(
            f"Epoch {epoch}/{args.num_epochs} ({dt:.0f}s) | "
            f"loss={avg_losses['total']:.4f} "
            f"distill={avg_losses['distill']:.4f} (w={loss_fn._distill_scale:.3f}) "
            f"align={avg_losses['align']:.4f} "
            f"proto={avg_losses['proto']:.4f} "
            f"ent={avg_losses['entropy']:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        # Evaluate every eval_interval epochs
        if epoch % args.eval_interval == 0 or epoch == args.num_epochs:
            log.info(f"Evaluating at epoch {epoch}...")
            metrics = evaluate_panoptic(
                model, val_loader, device, args.cityscapes_root)
            log.info(
                f"  PQ={metrics['PQ']:.2f} | "
                f"PQ_stuff={metrics['PQ_stuff']:.2f} | "
                f"PQ_things={metrics['PQ_things']:.2f} | "
                f"mIoU={metrics['mIoU']:.2f} | "
                f"changed={metrics['changed_pct']:.1f}%"
            )

            # Log coupling strengths
            for i, block in enumerate(model.blocks):
                log.info(
                    f"  Block {i}: alpha={block.alpha.item():.4f}, "
                    f"beta={block.beta.item():.4f}"
                )

            # Save checkpoint
            ckpt_path = os.path.join(
                args.output_dir, f"checkpoint_epoch_{epoch:04d}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics": metrics,
                "config": config,
            }, ckpt_path)

            # Save best (track PQ as primary metric)
            if metrics["PQ"] > best_pq:
                best_pq = metrics["PQ"]
                best_path = os.path.join(args.output_dir, "best.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "config": config,
                }, best_path)
                log.info(f"  New best PQ: {best_pq:.2f}% (saved to best.pth)")

            # Save metrics history
            metrics["epoch"] = epoch
            metrics_path = os.path.join(args.output_dir, "metrics_history.jsonl")
            with open(metrics_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")

    log.info(f"Training complete. Best PQ: {best_pq:.2f}%")
    log.info(f"Checkpoints saved to: {args.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CSCMRefineNet for semantic pseudo-label refinement")

    # Data
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--semantic_subdir", type=str,
                        default="pseudo_semantic_cause_crf",
                        help="Subdirectory for semantic pseudo-labels")
    parser.add_argument("--logits_subdir", type=str, default=None,
                        help="Subdirectory for soft logits (optional)")

    # Output
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/refine_net")

    # Model architecture
    parser.add_argument("--bridge_dim", type=int, default=192)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--block_type", type=str, default="conv",
                        choices=["conv", "mamba"])
    parser.add_argument("--layer_type", type=str, default="gated_delta_net",
                        choices=["mamba2", "gated_delta_net"])
    parser.add_argument("--scan_mode", type=str, default="bidirectional",
                        choices=["raster", "bidirectional", "cross_scan"])
    parser.add_argument("--coupling_strength", type=float, default=0.1)
    parser.add_argument("--d_state", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        default=True,
                        help="Enable gradient checkpointing to reduce memory")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                        action="store_false")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Enable bfloat16 mixed precision (off by default — fp32 is faster on MPS)")
    parser.add_argument("--no_amp", dest="amp", action="store_false")

    # Training
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=1)

    # Loss weights
    parser.add_argument("--lambda_distill", type=float, default=1.0,
                        help="Initial supervision weight (cosine decay to --lambda_distill_min)")
    parser.add_argument("--lambda_distill_min", type=float, default=0.0,
                        help="Final supervision weight after cosine warmdown")
    parser.add_argument("--lambda_align", type=float, default=5.0)
    parser.add_argument("--lambda_proto", type=float, default=0.5)
    parser.add_argument("--lambda_ent", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.2)

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--gpu", type=int, default=0)

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
