#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Stage-2: Mask2Former Swin-L (COCO pre-trained) fine-tuning
# Pseudo-labels: k=100 overclustered CAUSE + SPIdepth depth-guided
# ═══════════════════════════════════════════════════════════════════
#
# Prerequisites:
#   1. Cityscapes images at datasets/cityscapes/leftImg8bit_sequence/train/
#      (symlink leftImg8bit → leftImg8bit_sequence if needed)
#   2. Pseudo-labels at datasets/cityscapes/cups_pseudo_labels_k100/
#   3. GT validation at datasets/cityscapes/gtFine/
#   4. HuggingFace transformers installed (pip install transformers)
#   5. ~4GB disk for Swin-L checkpoint (auto-downloaded from HuggingFace)
#
# Usage:
#   bash scripts/run_stage2_mask2former_k100.sh
#
# Monitor:
#   tail -f experiments/cups_mask2former_swinl_k100_stage2.log

set -euo pipefail

# ─── Paths (relative to repo root) ───
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/refs/cups:${PYTHONPATH:-}"

CONFIG="refs/cups/configs/train_cityscapes_mask2former_swinl_k100.yaml"
LOG_DIR="experiments"
LOG_FILE="${LOG_DIR}/cups_mask2former_swinl_k100_stage2.log"

mkdir -p "${LOG_DIR}"

echo "=== CUPS Stage-2: Mask2Former Swin-L + k=100 Pseudo-Labels ===" | tee "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Repo root: ${REPO_ROOT}" | tee -a "${LOG_FILE}"
echo "Config: ${CONFIG}" | tee -a "${LOG_FILE}"

# ─── Verify GPU ───
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, CUDA: {torch.cuda.is_available()}')" 2>&1 | tee -a "${LOG_FILE}"

# ─── Verify transformers ───
python -c "from transformers import Mask2FormerForUniversalSegmentation; print('transformers OK')" 2>&1 | tee -a "${LOG_FILE}"

# ─── Verify pseudo-labels exist ───
PSEUDO_DIR="datasets/cityscapes/cups_pseudo_labels_k100"
N_SEM=$(ls "${PSEUDO_DIR}"/*_semantic.png 2>/dev/null | wc -l)
N_INST=$(ls "${PSEUDO_DIR}"/*_instance.png 2>/dev/null | wc -l)
N_PT=$(ls "${PSEUDO_DIR}"/*.pt 2>/dev/null | wc -l)
echo "Pseudo-labels: ${N_SEM} semantic, ${N_INST} instance, ${N_PT} distribution files" | tee -a "${LOG_FILE}"

# ─── Launch training ───
echo "Launching Mask2Former training..." | tee -a "${LOG_FILE}"

python refs/cups/train.py \
    --experiment_config_file "${CONFIG}" \
    --disable_wandb \
    2>&1 | tee -a "${LOG_FILE}"

echo "=== Training complete: $(date) ===" | tee -a "${LOG_FILE}"
