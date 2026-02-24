#!/usr/bin/env python3
"""Convert DINO ResNet-50 checkpoint to Detectron2 format.

Usage:
    python scripts/convert_dino_to_d2.py

Expects:
    refs/cups/cups/model/backbone_checkpoints/dino_resnet50_pretrain.pth

Produces:
    refs/cups/cups/model/backbone_checkpoints/dino_RN50_pretrain_d2_format.pkl
"""

import os
import pickle

import torch

CKPT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "refs", "cups", "cups", "model", "backbone_checkpoints",
)

RAW_PATH = os.path.join(CKPT_DIR, "dino_resnet50_pretrain.pth")
OUT_PATH = os.path.join(CKPT_DIR, "dino_RN50_pretrain_d2_format.pkl")

if os.path.exists(OUT_PATH) and os.path.getsize(OUT_PATH) > 0:
    print(f"Already exists: {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e6:.1f} MB)")
    exit(0)

if not os.path.exists(RAW_PATH):
    print(f"ERROR: Raw checkpoint not found: {RAW_PATH}")
    print("Download it first:")
    print(f"  curl -L -o {RAW_PATH} https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth")
    exit(1)

size_mb = os.path.getsize(RAW_PATH) / 1e6
print(f"Loading {RAW_PATH} ({size_mb:.1f} MB)...")

sd = torch.load(RAW_PATH, map_location="cpu", weights_only=False)

nd = {}
for k, v in sd.items():
    n = k
    n = n.replace("layer1", "res2")
    n = n.replace("layer2", "res3")
    n = n.replace("layer3", "res4")
    n = n.replace("layer4", "res5")
    n = "backbone.bottom_up." + n
    n = n.replace("downsample.0", "shortcut")
    n = n.replace("downsample.1", "shortcut.norm")
    n = n.replace(".bn1", ".conv1.norm")
    n = n.replace(".bn2", ".conv2.norm")
    n = n.replace(".bn3", ".conv3.norm")
    nd[n] = v.numpy()

with open(OUT_PATH, "wb") as f:
    pickle.dump(
        {"model": nd, "__author__": "DINO", "matching_heuristics": True}, f
    )

print(f"Saved: {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e6:.1f} MB)")
