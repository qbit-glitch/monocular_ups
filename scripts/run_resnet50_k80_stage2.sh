#!/bin/bash
# Launch CUPS stage-2 training with DINOv2-ResNet50 backbone + k=80 overclustered pseudo-labels.
# k=80 raw overclusters with depth-guided instance splitting (gt=0.20, ma=1000).
# Best Stage-1 pseudo-labels: PQ=26.74 (PQ_st=32.08, PQ_th=19.41).
#
# Usage:
#   bash scripts/run_resnet50_k80_stage2.sh                    # Fresh start
#   bash scripts/run_resnet50_k80_stage2.sh /path/to/ckpt      # Resume from checkpoint
#
# Remote:
#   ssh santosh@172.17.254.146 "nohup bash /media/santosh/Kuldeep/unsupervised-panoptic-segmentation/scripts/run_resnet50_k80_stage2.sh > /dev/null 2>&1 &"
#
# Monitor:
#   ssh santosh@172.17.254.146 "tail -f /media/santosh/Kuldeep/unsupervised-panoptic-segmentation/experiments/cups_resnet50_k80_stage2.log"

set -euo pipefail

# ─── Environment ───
export LD_LIBRARY_PATH="/home/santosh/anaconda3/envs/cups/lib:${LD_LIBRARY_PATH:-}"

CONDA_BIN="/home/santosh/anaconda3/envs/cups/bin"
PYTHON="${CONDA_BIN}/python"
WORK_DIR="/media/santosh/Kuldeep/unsupervised-panoptic-segmentation"
LOG_FILE="${WORK_DIR}/experiments/cups_resnet50_k80_stage2.log"

export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"

cd "${WORK_DIR}"

mkdir -p "${WORK_DIR}/experiments"

CKPT_PATH="${1:-}"  # Optional: pass checkpoint path as first argument to resume

echo "=== CUPS Stage-2 Training: ResNet50 + k=80 Depth-Guided Pseudo-Labels ===" | tee -a "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Python: ${PYTHON}" | tee -a "${LOG_FILE}"
echo "Backbone: DINOv2-ResNet50 (original CUPS backbone)" | tee -a "${LOG_FILE}"
echo "Pseudo-labels: cups_pseudo_labels_k80 (k=80 overclusters, depth-guided gt=0.20 ma=1000)" | tee -a "${LOG_FILE}"

# Verify GPU availability
${PYTHON} -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, CUDA: {torch.cuda.is_available()}')" 2>&1 | tee -a "${LOG_FILE}"

# Verify pseudo-labels exist
PSEUDO_DIR="${WORK_DIR}/datasets/cityscapes/cups_pseudo_labels_k80"
if [ ! -d "${PSEUDO_DIR}" ]; then
    # Try config-specified path
    PSEUDO_DIR="/media/santosh/Kuldeep/panoptic_segmentation/datasets/cityscapes/cups_pseudo_labels_k80"
fi
NUM_SEM=$(ls ${PSEUDO_DIR}/*_semantic.png 2>/dev/null | wc -l)
NUM_INST=$(ls ${PSEUDO_DIR}/*_instance.png 2>/dev/null | wc -l)
echo "Pseudo-labels: ${NUM_SEM} semantic, ${NUM_INST} instance" | tee -a "${LOG_FILE}"

if [ "${NUM_SEM}" -lt 2975 ]; then
    echo "WARNING: Expected 2975 semantic labels, found ${NUM_SEM}." | tee -a "${LOG_FILE}"
    echo "Generate them first:" | tee -a "${LOG_FILE}"
    echo "  python pseudo_labels/generate_overclustered_semantics.py \\" | tee -a "${LOG_FILE}"
    echo "      --cityscapes_root /path/to/cityscapes --k 80 --raw_clusters --split train" | tee -a "${LOG_FILE}"
    echo "  python pseudo_labels/convert_to_cups_format.py \\" | tee -a "${LOG_FILE}"
    echo "      --cityscapes_root /path/to/cityscapes \\" | tee -a "${LOG_FILE}"
    echo "      --semantic_subdir pseudo_semantic_raw_k80 \\" | tee -a "${LOG_FILE}"
    echo "      --output_subdir cups_pseudo_labels_k80 \\" | tee -a "${LOG_FILE}"
    echo "      --split train --num_classes 80 --depth_instances \\" | tee -a "${LOG_FILE}"
    echo "      --centroids_path .../pseudo_semantic_raw_k80/kmeans_centroids.npz \\" | tee -a "${LOG_FILE}"
    echo "      --depth_subdir depth_spidepth --grad_threshold 0.20 --min_instance_area 1000" | tee -a "${LOG_FILE}"
    exit 1
fi

echo "Launching training (2x GPU DDP)..." | tee -a "${LOG_FILE}"

# Launch training with k80 config
if [ -n "${CKPT_PATH}" ]; then
    echo "Resuming from checkpoint: ${CKPT_PATH}" | tee -a "${LOG_FILE}"
    ${PYTHON} train.py \
        --experiment_config_file "${WORK_DIR}/configs/train_cityscapes_resnet50_k80.yaml" \
        --disable_wandb \
        --ckpt_path "${CKPT_PATH}" \
        2>&1 | tee -a "${LOG_FILE}"
else
    ${PYTHON} train.py \
        --experiment_config_file "${WORK_DIR}/configs/train_cityscapes_resnet50_k80.yaml" \
        --disable_wandb \
        2>&1 | tee -a "${LOG_FILE}"
fi

echo "=== Training complete: $(date) ===" | tee -a "${LOG_FILE}"
