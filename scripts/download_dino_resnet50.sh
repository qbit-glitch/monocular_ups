#!/bin/bash
# Download DINO ResNet-50 pretrained weights and convert to Detectron2 format.
# This is required for CUPS stage-2 training.
#
# Usage: bash scripts/download_dino_resnet50.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_DIR="${REPO_ROOT}/refs/cups/cups/model/backbone_checkpoints"
D2_CKPT="${CKPT_DIR}/dino_RN50_pretrain_d2_format.pkl"

if [ -f "${D2_CKPT}" ] && [ -s "${D2_CKPT}" ]; then
    echo "DINO ResNet-50 checkpoint already exists: ${D2_CKPT} ($(du -h "${D2_CKPT}" | cut -f1))"
    exit 0
fi

mkdir -p "${CKPT_DIR}"

# Download raw DINO checkpoint
RAW_CKPT="${CKPT_DIR}/dino_resnet50_pretrain.pth"
if [ ! -f "${RAW_CKPT}" ]; then
    echo "Downloading DINO ResNet-50 pretrained weights..."
    wget -O "${RAW_CKPT}" \
        "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
    echo "Downloaded: ${RAW_CKPT} ($(du -h "${RAW_CKPT}" | cut -f1))"
fi

# Convert to Detectron2 format
echo "Converting to Detectron2 format..."
python -c "
import pickle
import torch

# Load DINO checkpoint
state_dict = torch.load('${RAW_CKPT}', map_location='cpu')

# Convert keys: 'layer1.0.conv1.weight' -> 'backbone.bottom_up.res2.0.conv1.weight'
# Following Detectron2 ResNet key naming convention
new_state_dict = {}
for k, v in state_dict.items():
    # Map DINO torchvision keys to Detectron2 keys
    new_key = k
    # Map layer names
    new_key = new_key.replace('layer1', 'res2')
    new_key = new_key.replace('layer2', 'res3')
    new_key = new_key.replace('layer3', 'res4')
    new_key = new_key.replace('layer4', 'res5')
    # Add backbone prefix
    new_key = 'backbone.bottom_up.' + new_key
    # Convert downsample to shortcut
    new_key = new_key.replace('downsample.0', 'shortcut')
    new_key = new_key.replace('downsample.1', 'shortcut.norm')
    # Convert BN to norm
    new_key = new_key.replace('.bn1', '.conv1.norm')
    new_key = new_key.replace('.bn2', '.conv2.norm')
    new_key = new_key.replace('.bn3', '.conv3.norm')
    new_state_dict[new_key] = v.numpy()

# Save as Detectron2 pickle
d2_model = {'model': new_state_dict, '__author__': 'DINO', 'matching_heuristics': True}
with open('${D2_CKPT}', 'wb') as f:
    pickle.dump(d2_model, f)
print(f'Saved Detectron2 format checkpoint: ${D2_CKPT}')
"

# Clean up raw checkpoint
rm -f "${RAW_CKPT}"
echo "Done. DINO ResNet-50 D2 checkpoint: ${D2_CKPT} ($(du -h "${D2_CKPT}" | cut -f1))"
