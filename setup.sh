#!/bin/bash
# ============================================================
# End-to-End Environment Setup for Unsupervised Panoptic Segmentation
# ============================================================
#
# Usage:
#   bash setup.sh                          # Full setup (new conda env + all deps)
#   bash setup.sh --env-name my_env        # Custom env name
#   bash setup.sh --python 3.10            # Custom Python version
#   bash setup.sh --cuda 11.8              # Custom CUDA version (for GPU machines)
#   bash setup.sh --skip-conda             # Skip conda env creation (use current env)
#   bash setup.sh --skip-detectron2        # Skip Detectron2 installation
#
# Tested on:
#   - macOS (Apple Silicon M4 Pro) with MPS backend
#   - Linux (Ubuntu 20.04) with CUDA 11.8 + GTX 1080 Ti
# ============================================================

set -euo pipefail

# ─── Defaults ───
ENV_NAME="ups"
PYTHON_VERSION="3.10"
CUDA_VERSION=""
SKIP_CONDA=false
SKIP_DETECTRON2=false

# ─── Parse arguments ───
while [[ $# -gt 0 ]]; do
    case $1 in
        --env-name)     ENV_NAME="$2"; shift 2 ;;
        --python)       PYTHON_VERSION="$2"; shift 2 ;;
        --cuda)         CUDA_VERSION="$2"; shift 2 ;;
        --skip-conda)   SKIP_CONDA=true; shift ;;
        --skip-detectron2) SKIP_DETECTRON2=true; shift ;;
        -h|--help)
            echo "Usage: bash setup.sh [--env-name NAME] [--python VER] [--cuda VER] [--skip-conda] [--skip-detectron2]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "============================================================"
echo "  Unsupervised Panoptic Segmentation — Environment Setup"
echo "============================================================"
echo "  Repo:           ${REPO_DIR}"
echo "  Env name:       ${ENV_NAME}"
echo "  Python:         ${PYTHON_VERSION}"
echo "  CUDA:           ${CUDA_VERSION:-auto-detect}"
echo "  Skip conda:     ${SKIP_CONDA}"
echo "  Skip detectron2: ${SKIP_DETECTRON2}"
echo "============================================================"
echo ""

# ─── Step 1: Create conda environment ───
if [ "$SKIP_CONDA" = false ]; then
    echo "[1/6] Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    if conda info --envs | grep -q "^${ENV_NAME} "; then
        echo "  Environment '${ENV_NAME}' already exists. Activating..."
    else
        conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    fi
    # Activate
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
    echo "  Python: $(python --version)"
    echo "  Pip:    $(pip --version)"
else
    echo "[1/6] Skipping conda env creation (using current environment)."
fi
echo ""

# ─── Step 2: Install PyTorch ───
echo "[2/6] Installing PyTorch..."
if [ -n "$CUDA_VERSION" ]; then
    # CUDA version specified
    case "$CUDA_VERSION" in
        11.8) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
        12.1) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
        12.4) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
        *)    echo "  Unsupported CUDA version: ${CUDA_VERSION}. Supported: 11.8, 12.1, 12.4"; exit 1 ;;
    esac
    pip install torch torchvision --index-url "${TORCH_INDEX}"
elif [[ "$(uname)" == "Darwin" ]]; then
    # macOS — MPS backend
    echo "  Detected macOS — installing PyTorch with MPS support..."
    pip install torch torchvision
else
    # Linux without explicit CUDA — try auto-detect
    if command -v nvidia-smi &>/dev/null; then
        DETECTED_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 || echo "")
        echo "  Detected NVIDIA GPU (driver: ${DETECTED_CUDA}). Installing CUDA PyTorch..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "  No GPU detected. Installing CPU PyTorch..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
fi
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
echo ""

# ─── Step 3: Install Detectron2 ───
if [ "$SKIP_DETECTRON2" = false ]; then
    echo "[3/6] Installing Detectron2 from source..."
    pip install 'git+https://github.com/facebookresearch/detectron2.git'
    python -c "import detectron2; print(f'  Detectron2 {detectron2.__version__}')"
else
    echo "[3/6] Skipping Detectron2 installation."
fi
echo ""

# ─── Step 4: Install Python requirements ───
echo "[4/6] Installing Python requirements..."
pip install -r "${REPO_DIR}/requirements.txt"
echo ""

# ─── Step 5: Verify all imports ───
echo "[5/6] Verifying imports..."
ERRORS=0

verify_import() {
    if python -c "import $1" 2>/dev/null; then
        echo "  [OK] $1"
    else
        echo "  [FAIL] $1"
        ERRORS=$((ERRORS + 1))
    fi
}

verify_import torch
verify_import torchvision
verify_import pytorch_lightning
verify_import kornia
verify_import timm
verify_import scipy
verify_import sklearn
verify_import cv2
verify_import PIL
verify_import yacs
verify_import torchmetrics
verify_import tqdm
verify_import numpy
verify_import pydensecrf
verify_import fvcore
verify_import colored
verify_import transformers

if [ "$SKIP_DETECTRON2" = false ]; then
    verify_import detectron2
fi

# Verify local packages
cd "${REPO_DIR}"
if python -c "import cups" 2>/dev/null; then
    echo "  [OK] cups (local)"
else
    echo "  [FAIL] cups (local) — make sure PYTHONPATH includes ${REPO_DIR}"
    ERRORS=$((ERRORS + 1))
fi

if python -c "import sys; sys.path.insert(0, 'pseudo_labels'); from refine_net import CSCMRefineNet" 2>/dev/null; then
    echo "  [OK] refine_net (local)"
else
    echo "  [FAIL] refine_net (local)"
    ERRORS=$((ERRORS + 1))
fi

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "  All imports verified successfully!"
else
    echo "  WARNING: ${ERRORS} import(s) failed. Check the output above."
fi
echo ""

# ─── Step 6: Print usage instructions ───
echo "[6/6] Setup complete!"
echo ""
echo "============================================================"
echo "  Quick Start"
echo "============================================================"
echo ""
echo "  # Activate environment:"
if [ "$SKIP_CONDA" = false ]; then
    echo "  conda activate ${ENV_NAME}"
fi
echo ""
echo "  # Set PYTHONPATH (required for all commands):"
echo "  export PYTHONPATH=\"${REPO_DIR}:\${PYTHONPATH:-}\""
echo ""
echo "  # Stage 1: Generate pseudo-labels"
echo "  python pseudo_labels/generate_semantic_pseudolabels_cause.py \\"
echo "      --cityscapes_root /path/to/cityscapes \\"
echo "      --checkpoint_dir /path/to/cause/checkpoints"
echo ""
echo "  # Stage 2: Train"
echo "  python train.py \\"
echo "      --experiment_config_file configs/train_cityscapes_resnet50_k50.yaml \\"
echo "      --data_root /path/to/cityscapes/ \\"
echo "      --pseudo_root /path/to/cityscapes/cups_pseudo_labels_k50/ \\"
echo "      --batch_size 2 --num_gpus 2 \\"
echo "      --disable_wandb"
echo ""
echo "  # Parameter sweep"
echo "  python pseudo_labels/sweep_k50_spidepth.py \\"
echo "      --cityscapes_root /path/to/cityscapes \\"
echo "      --grad_thresholds 0.2 0.3 0.5 \\"
echo "      --min_areas 500 700 1000"
echo ""
echo "============================================================"
