#!/usr/bin/env python3
"""Restore the exact k=80 centroids that produced PQ=26.74.

Usage:
    python pseudo_labels/restore_centroids_k80.py \
        --cityscapes_root datasets/cityscapes

This reads kmeans_centroids_k80.b64 (base64-encoded npz) and writes
kmeans_centroids.npz into the pseudo_semantic_raw_k80 directory.
"""

import argparse
import base64
import io
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes_root", required=True)
    args = parser.parse_args()

    # Read base64 file
    b64_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "kmeans_centroids_k80.b64",
    )
    with open(b64_path) as f:
        encoded = f.read().strip()

    # Decode
    buf = io.BytesIO(base64.b64decode(encoded))
    d = np.load(buf)

    # Save
    out_dir = os.path.join(args.cityscapes_root, "pseudo_semantic_raw_k80")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "kmeans_centroids.npz")
    np.savez(out_path, centroids=d["centroids"], cluster_to_class=d["cluster_to_class"])

    print(f"Saved to {out_path}")
    print(f"  centroids: {d['centroids'].shape}")
    print(f"  cluster_to_class: {d['cluster_to_class']}")

    # Show thing class coverage
    c2c = d["cluster_to_class"]
    names = [
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain", "sky",
        "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
    ]
    print("\n  Thing class clusters:")
    for tid in range(11, 19):
        clusters = [i for i, c in enumerate(c2c) if c == tid]
        status = f"{len(clusters)} clusters -> {clusters}" if clusters else "0 clusters (MISSING)"
        print(f"    {names[tid]:15s}: {status}")


if __name__ == "__main__":
    main()
