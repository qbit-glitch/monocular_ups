#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Stage-2: CUPS training with DINO ResNet-50 + Cascade Mask R-CNN
# Pseudo-labels: k=100 overclustered CAUSE + SPIdepth depth-guided
# ═══════════════════════════════════════════════════════════════════
#
# Prerequisites:
#   1. Cityscapes images at datasets/cityscapes/leftImg8bit_sequence/train/
#      (symlink leftImg8bit → leftImg8bit_sequence if needed)
#   2. Pseudo-labels at datasets/cityscapes/cups_pseudo_labels_k100/
#   3. GT validation at datasets/cityscapes/gtFine/
#   4. DINO ResNet-50 checkpoint at refs/cups/cups/model/backbone_checkpoints/
#      Download: https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth
#      Convert: python refs/cups/cups/model/convert_dino_to_d2.py (if not already converted)
#
# Usage:
#   bash scripts/run_stage2_resnet50_k100.sh
#
# Monitor:
#   tail -f experiments/cups_resnet50_k100_stage2.log

set -euo pipefail

# ─── Paths (relative to repo root) ───
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/refs/cups:${PYTHONPATH:-}"

CONFIG="refs/cups/configs/train_cityscapes_resnet50_k100.yaml"
LOG_DIR="experiments"
LOG_FILE="${LOG_DIR}/cups_resnet50_k100_stage2.log"

mkdir -p "${LOG_DIR}"

echo "=== CUPS Stage-2: DINO ResNet-50 + k=100 Pseudo-Labels ===" | tee "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Repo root: ${REPO_ROOT}" | tee -a "${LOG_FILE}"
echo "Config: ${CONFIG}" | tee -a "${LOG_FILE}"

# ─── Verify GPU ───
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, CUDA: {torch.cuda.is_available()}')" 2>&1 | tee -a "${LOG_FILE}"

# ─── Verify pseudo-labels exist ───
PSEUDO_DIR="datasets/cityscapes/cups_pseudo_labels_k100"
N_SEM=$(ls "${PSEUDO_DIR}"/*_semantic.png 2>/dev/null | wc -l)
N_INST=$(ls "${PSEUDO_DIR}"/*_instance.png 2>/dev/null | wc -l)
N_PT=$(ls "${PSEUDO_DIR}"/*.pt 2>/dev/null | wc -l)
echo "Pseudo-labels: ${N_SEM} semantic, ${N_INST} instance, ${N_PT} distribution files" | tee -a "${LOG_FILE}"

# ─── Verify DINO checkpoint ───
DINO_CKPT="refs/cups/cups/model/backbone_checkpoints/dino_RN50_pretrain_d2_format.pkl"
if [ ! -f "${DINO_CKPT}" ]; then
    echo "ERROR: DINO checkpoint not found at ${DINO_CKPT}" | tee -a "${LOG_FILE}"
    echo "Download and convert the DINO ResNet-50 checkpoint first." | tee -a "${LOG_FILE}"
    exit 1
fi
echo "DINO checkpoint: ${DINO_CKPT} ($(du -h "${DINO_CKPT}" | cut -f1))" | tee -a "${LOG_FILE}"

# ─── Launch training ───
echo "Launching training..." | tee -a "${LOG_FILE}"

python refs/cups/train.py \
    --experiment_config_file "${CONFIG}" \
    --disable_wandb \
    2>&1 | tee -a "${LOG_FILE}"

echo "=== Training complete: $(date) ===" | tee -a "${LOG_FILE}"
