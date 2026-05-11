#!/usr/bin/env bash
# train_fix.sh
# ============
# Train a FIX classifier from hand-labeled ICA blocks.
#
# Steps:
#   1. Write hand_labels_noise.txt into each labeled block's .ica dir
#      (converts OHBA label.txt last line to FIX format: [1, 2, ...])
#   2. Run fix -f (feature extraction) on each labeled block
#   3. Run fix -t to train the classifier
#
# Output model: <PIPELINE_DIR>/staresina_fix.pyfix_model
#
# Usage:
#   bash train_fix.sh
#   bash train_fix.sh --dry-run
#   bash train_fix.sh --loo    # leave-one-out accuracy testing

set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/config.sh"

source "$LMOD_INIT"
module load "$FSL_MODULE"

FUNC_BASE="$OUTPUT_ROOT/func_crop5"
LOG_DIR="$PIPELINE_DIR/logs"
MODEL_OUT="$PIPELINE_DIR/staresina_fix"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DRY_RUN=0
LOO=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --loo)     LOO=1; shift ;;
        *) shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Step 1: Collect labeled blocks and write hand_labels_noise.txt
# ---------------------------------------------------------------------------
ICA_DIRS=()

for label_txt in "$FUNC_BASE"/*/label.txt; do
    [[ -f "$label_txt" ]] || continue
    block_dir=$(dirname "$label_txt")
    block_id=$(basename "$block_dir")
    feat_dir="$block_dir/bold.feat"
    ica_dir="$feat_dir/filtered_func_data.ica"

    if [[ ! -f "$feat_dir/filtered_func_data.nii.gz" ]]; then
        echo "[$block_id] No filtered_func_data.nii.gz, skipping"
        continue
    fi
    if [[ ! -f "$ica_dir/melodic_mix" ]]; then
        echo "[$block_id] MELODIC not complete, skipping"
        continue
    fi

    noise_line=$(grep -E '^\[' "$label_txt" | tail -1)
    if [[ -z "$noise_line" ]]; then
        echo "[$block_id] No noise list in label.txt, skipping"
        continue
    fi

    echo "$noise_line" > "$feat_dir/hand_labels_noise.txt"
    echo "[$block_id] hand_labels_noise.txt: $noise_line"

    ICA_DIRS+=("$feat_dir")
done

echo ""
echo "Found ${#ICA_DIRS[@]} labeled blocks"

if [[ ${#ICA_DIRS[@]} -eq 0 ]]; then
    echo "No labeled blocks found — exiting"
    exit 1
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] Would run fix -f then fix -t on ${#ICA_DIRS[@]} blocks"
    for d in "${ICA_DIRS[@]}"; do echo "  $d"; done
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 2: Feature extraction (required before fix -t in FIX 0.10)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Feature extraction (fix -f)"
echo "============================================================"

for feat_dir in "${ICA_DIRS[@]}"; do
    block_id=$(basename "$(dirname "$feat_dir")")
    log="$LOG_DIR/${block_id}_fix_features.log"
    echo "[$block_id] Running fix -f ..."
    { echo "=== $block_id fix -f ==="; date; } > "$log"
    if fix -f "$feat_dir" 2>&1 | tee -a "$log"; then
        echo "[$block_id] features DONE"
    else
        echo "[$block_id] features FAILED — see $log"
    fi
done

# ---------------------------------------------------------------------------
# Step 3: Train classifier
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Training FIX classifier"
echo " Output: ${MODEL_OUT}.pyfix_model"
echo " Blocks: ${#ICA_DIRS[@]}"
echo "============================================================"

LOO_FLAG=""
[[ $LOO -eq 1 ]] && LOO_FLAG="-l"

log="$LOG_DIR/fix_train.log"
{ echo "=== fix -t ==="; date; printf '%s\n' "${ICA_DIRS[@]}"; } > "$log"

# shellcheck disable=SC2086
if fix -t "$MODEL_OUT" $LOO_FLAG "${ICA_DIRS[@]}" 2>&1 | tee -a "$log"; then
    echo ""
    echo "Training DONE — model: ${MODEL_OUT}.pyfix_model"
else
    echo "Training FAILED — see $log"
    exit 1
fi
