#!/usr/bin/env python3
"""Download SPIdepth Cityscapes checkpoint from Hugging Face."""

import os
import shutil

from huggingface_hub import hf_hub_download

REPO_ID = "MykolaL/SPIdepth"
OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "refs", "spidepth", "checkpoints", "cityscapes",
)

os.makedirs(OUT_DIR, exist_ok=True)

for fname in ["encoder.pth", "depth.pth"]:
    print(f"Downloading {fname}...")
    cached = hf_hub_download(REPO_ID, filename=f"cityscapes/{fname}")
    dest = os.path.join(OUT_DIR, fname)
    shutil.copy(cached, dest)
    print(f"  Saved to {dest}")

print("Done.")
