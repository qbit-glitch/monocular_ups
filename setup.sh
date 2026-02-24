#!/bin/bash
# ============================================================
# Environment Setup for Unsupervised Panoptic Pseudo-Label Generation
# ============================================================
#
# Usage:
#   bash setup.sh                          # Full setup (new conda env + all deps)
#   bash setup.sh --env-name my_env        # Custom env name
#   bash setup.sh --python 3.10            # Custom Python version
#   bash setup.sh --cuda 11.8              # Custom CUDA version (for GPU machines)
#   bash setup.sh --skip-conda             # Skip conda env creation (use current env)
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

# ─── Parse arguments ───
while [[ $# -gt 0 ]]; do
    case $1 in
        --env-name)     ENV_NAME="$2"; shift 2 ;;
        --python)       PYTHON_VERSION="$2"; shift 2 ;;
        --cuda)         CUDA_VERSION="$2"; shift 2 ;;
        --skip-conda)   SKIP_CONDA=true; shift ;;
        -h|--help)
            echo "Usage: bash setup.sh [--env-name NAME] [--python VER] [--cuda VER] [--skip-conda]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "============================================================"
echo "  Unsupervised Panoptic Pseudo-Labels — Environment Setup"
echo "============================================================"
echo "  Repo:           ${REPO_DIR}"
echo "  Env name:       ${ENV_NAME}"
echo "  Python:         ${PYTHON_VERSION}"
echo "  CUDA:           ${CUDA_VERSION:-auto-detect}"
echo "  Skip conda:     ${SKIP_CONDA}"
echo "============================================================"
echo ""

# ─── Step 1: Create conda environment ───
if [ "$SKIP_CONDA" = false ]; then
    echo "[1/4] Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    if conda info --envs | grep -q "^${ENV_NAME} "; then
        echo "  Environment '${ENV_NAME}' already exists. Activating..."
    else
        conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    fi
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
    echo "  Python: $(python --version)"
else
    echo "[1/4] Skipping conda env creation (using current environment)."
fi
echo ""

# ─── Step 2: Install PyTorch ───
echo "[2/4] Installing PyTorch..."
if [ -n "$CUDA_VERSION" ]; then
    case "$CUDA_VERSION" in
        11.8) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
        12.1) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
        12.4) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
        *)    echo "  Unsupported CUDA version: ${CUDA_VERSION}. Supported: 11.8, 12.1, 12.4"; exit 1 ;;
    esac
    pip install torch torchvision --index-url "${TORCH_INDEX}"
elif [[ "$(uname)" == "Darwin" ]]; then
    echo "  Detected macOS — installing PyTorch with MPS support..."
    pip install torch torchvision
else
    if command -v nvidia-smi &>/dev/null; then
        echo "  Detected NVIDIA GPU. Installing CUDA PyTorch..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "  No GPU detected. Installing CPU PyTorch..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
fi
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
echo ""

# ─── Step 3: Install Python requirements ───
echo "[3/4] Installing Python requirements..."
pip install -r "${REPO_DIR}/requirements.txt"
echo ""

# ─── Step 4: Verify imports ───
echo "[4/4] Verifying imports..."
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
verify_import scipy
verify_import sklearn
verify_import cv2
verify_import PIL
verify_import tqdm
verify_import numpy
verify_import timm
verify_import transformers

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "  All imports verified successfully!"
else
    echo "  WARNING: ${ERRORS} import(s) failed. Check the output above."
fi
echo ""

echo "============================================================"
echo "  Setup complete! Quick Start:"
echo "============================================================"
echo ""
if [ "$SKIP_CONDA" = false ]; then
    echo "  conda activate ${ENV_NAME}"
fi
echo ""
echo "  # Generate k=80 overclustered semantic labels:"
echo "  python pseudo_labels/generate_overclustered_semantics.py \\"
echo "      --cityscapes_root /path/to/cityscapes \\"
echo "      --k 80 --raw_clusters --skip_crf"
echo ""
echo "  # Run parameter sweep with depth-guided instances:"
echo "  python pseudo_labels/sweep_k50_spidepth.py \\"
echo "      --cityscapes_root /path/to/cityscapes \\"
echo "      --semantic_subdir pseudo_semantic_raw_k80"
echo ""
echo "============================================================"
