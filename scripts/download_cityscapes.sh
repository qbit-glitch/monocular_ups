#!/bin/bash
# ============================================================
# Download and extract Cityscapes dataset
# ============================================================
#
# Prerequisites:
#   Register at https://www.cityscapes-dataset.com/register/
#   Then set your credentials:
#     export CITYSCAPES_USERNAME="your_email"
#     export CITYSCAPES_PASSWORD="your_password"
#
# Usage:
#   bash scripts/download_cityscapes.sh                          # Download to datasets/ in repo
#   bash scripts/download_cityscapes.sh --output /path/to/dir    # Custom output directory
#   bash scripts/download_cityscapes.sh --skip-extract           # Download only, don't extract
#   bash scripts/download_cityscapes.sh --leftimg-only           # Only download leftImg8bit + gtFine
#
# What gets downloaded:
#   leftImg8bit_trainvaltest.zip  (~11 GB) — RGB images
#   gtFine_trainvaltest.zip       (~241 MB) — Fine annotations (semantic + instance)
# ============================================================

set -euo pipefail

# ─── Defaults ───
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${REPO_DIR}/datasets/cityscapes"
SKIP_EXTRACT=false
LEFTIMG_ONLY=false

# ─── Parse arguments ───
while [[ $# -gt 0 ]]; do
    case $1 in
        --output)        OUTPUT_DIR="$2"; shift 2 ;;
        --skip-extract)  SKIP_EXTRACT=true; shift ;;
        --leftimg-only)  LEFTIMG_ONLY=true; shift ;;
        -h|--help)
            echo "Usage: bash scripts/download_cityscapes.sh [--output DIR] [--skip-extract] [--leftimg-only]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ─── Check credentials ───
if [ -z "${CITYSCAPES_USERNAME:-}" ] || [ -z "${CITYSCAPES_PASSWORD:-}" ]; then
    echo "ERROR: Cityscapes credentials not set."
    echo ""
    echo "  1. Register at https://www.cityscapes-dataset.com/register/"
    echo "  2. Export your credentials:"
    echo "     export CITYSCAPES_USERNAME=\"your_email@example.com\""
    echo "     export CITYSCAPES_PASSWORD=\"your_password\""
    echo "  3. Re-run this script."
    exit 1
fi

echo "============================================================"
echo "  Cityscapes Dataset Download"
echo "============================================================"
echo "  Output:   ${OUTPUT_DIR}"
echo "  Username: ${CITYSCAPES_USERNAME}"
echo "============================================================"
echo ""

mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"

# ─── Helper: download with cookie-based auth ───
download_cityscapes() {
    local PACKAGE_ID="$1"
    local FILENAME="$2"

    if [ -f "${FILENAME}" ]; then
        echo "  ${FILENAME} already exists, skipping download."
        return 0
    fi

    echo "  Downloading ${FILENAME}..."
    # Get session cookie
    wget --keep-session-cookies --save-cookies=cookies.txt --post-data \
        "username=${CITYSCAPES_USERNAME}&password=${CITYSCAPES_PASSWORD}&submit=Login" \
        "https://www.cityscapes-dataset.com/login/" -O /dev/null -q 2>/dev/null || true

    # Download with cookie
    wget --load-cookies cookies.txt --content-disposition \
        "https://www.cityscapes-dataset.com/file-handling/?packageID=${PACKAGE_ID}" \
        -O "${FILENAME}" 2>&1 | tail -1

    rm -f cookies.txt

    if [ ! -f "${FILENAME}" ] || [ ! -s "${FILENAME}" ]; then
        echo "  ERROR: Download failed for ${FILENAME}. Check credentials."
        rm -f "${FILENAME}"
        return 1
    fi
    local SIZE=$(du -sh "${FILENAME}" | cut -f1)
    echo "  Downloaded ${FILENAME} (${SIZE})"
}

# ─── Download files ───
echo "[1/3] Downloading Cityscapes packages..."

# leftImg8bit (RGB images) — packageID=3
download_cityscapes 3 "leftImg8bit_trainvaltest.zip"

# gtFine (annotations) — packageID=1
download_cityscapes 1 "gtFine_trainvaltest.zip"

echo ""

# ─── Extract ───
if [ "$SKIP_EXTRACT" = false ]; then
    echo "[2/3] Extracting archives..."

    if [ -d "leftImg8bit" ] && [ "$(find leftImg8bit -name '*.png' | head -1)" != "" ]; then
        echo "  leftImg8bit/ already extracted, skipping."
    else
        echo "  Extracting leftImg8bit_trainvaltest.zip..."
        unzip -q -o leftImg8bit_trainvaltest.zip
    fi

    if [ -d "gtFine" ] && [ "$(find gtFine -name '*_labelIds.png' | head -1)" != "" ]; then
        echo "  gtFine/ already extracted, skipping."
    else
        echo "  Extracting gtFine_trainvaltest.zip..."
        unzip -q -o gtFine_trainvaltest.zip
    fi
    echo ""
else
    echo "[2/3] Skipping extraction."
    echo ""
fi

# ─── Verify ───
echo "[3/3] Verifying dataset..."

count_files() {
    find "$1" -name "$2" 2>/dev/null | wc -l | tr -d ' '
}

TRAIN_IMGS=$(count_files "leftImg8bit/train" "*.png")
VAL_IMGS=$(count_files "leftImg8bit/val" "*.png")
TEST_IMGS=$(count_files "leftImg8bit/test" "*.png")
TRAIN_GT=$(count_files "gtFine/train" "*_labelIds.png")
VAL_GT=$(count_files "gtFine/val" "*_labelIds.png")

echo "  leftImg8bit/train:  ${TRAIN_IMGS} images (expected: 2975)"
echo "  leftImg8bit/val:    ${VAL_IMGS} images (expected: 500)"
echo "  leftImg8bit/test:   ${TEST_IMGS} images (expected: 1525)"
echo "  gtFine/train:       ${TRAIN_GT} labels (expected: 2975)"
echo "  gtFine/val:         ${VAL_GT} labels (expected: 500)"

ERRORS=0
if [ "$TRAIN_IMGS" -lt 2975 ] 2>/dev/null; then ERRORS=$((ERRORS+1)); fi
if [ "$VAL_IMGS" -lt 500 ] 2>/dev/null; then ERRORS=$((ERRORS+1)); fi
if [ "$TRAIN_GT" -lt 2975 ] 2>/dev/null; then ERRORS=$((ERRORS+1)); fi
if [ "$VAL_GT" -lt 500 ] 2>/dev/null; then ERRORS=$((ERRORS+1)); fi

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "  Dataset verified successfully!"
else
    echo "  WARNING: Some expected files are missing. Check the counts above."
fi

echo ""
echo "============================================================"
echo "  Dataset ready at: ${OUTPUT_DIR}"
echo ""
echo "  Use with training:"
echo "    python train.py \\"
echo "      --experiment_config_file configs/train_cityscapes_resnet50_k50.yaml \\"
echo "      --data_root ${OUTPUT_DIR}/ \\"
echo "      --disable_wandb"
echo "============================================================"
