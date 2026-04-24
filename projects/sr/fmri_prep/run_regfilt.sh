#!/usr/bin/env bash
# run_regfilt.sh
# ==============
# Regress out noise ICs from resting-state blocks using fsl_regfilt.
# Reads noise component indices from a label.txt file (produced by manual
# or automated labelling). Expects label.txt to live alongside bold.feat/:
#
#   <OUTPUT_ROOT>/func/<block_id>/label.txt
#
# label.txt format (OHBA convention):
#   Line 1  : path to .ica dir
#   Lines 2+ : <IC>, <label>, <noise_bool>
#   Last line: [list of noise IC indices]  ← used by this script
#
# Output: bold.feat/filtered_func_data_clean.nii.gz
# Skips blocks where the output already exists.
#
# Usage:
#   bash run_regfilt.sh
#   bash run_regfilt.sh --subject sub-001
#   bash run_regfilt.sh --block sub-001_ses-01_scanrun-01_resting_01
#   bash run_regfilt.sh --dry-run

set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/config.sh"

source "$LMOD_INIT"
module load "$FSL_MODULE"

FUNC_BASE="$OUTPUT_ROOT/func"
LOG_DIR="$PIPELINE_DIR/logs"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DRY_RUN=0
SUBJECT_FILTER=""
BLOCK_FILTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        --subject)  SUBJECT_FILTER="$2"; shift 2 ;;
        --block)    BLOCK_FILTER="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Main loop: resting blocks only
# ---------------------------------------------------------------------------
total=0; skipped=0; done_count=0; failed=0

for block_dir in "$FUNC_BASE"/*_resting_*/; do
    [[ -d "$block_dir" ]] || continue
    block_id=$(basename "$block_dir")

    [[ -n "$SUBJECT_FILTER" && "$block_id" != "${SUBJECT_FILTER}"* ]] && continue
    [[ -n "$BLOCK_FILTER"   && "$block_id" != "$BLOCK_FILTER"      ]] && continue

    feat_out="$block_dir/bold.feat"
    filtered="$feat_out/filtered_func_data.nii.gz"
    ica_dir="$feat_out/filtered_func_data.ica"
    label_txt="$block_dir/label.txt"
    output="$feat_out/filtered_func_data_clean.nii.gz"

    if [[ ! -f "$filtered" ]]; then
        echo "[$block_id] FEAT not complete, skipping"
        continue
    fi

    if [[ ! -f "$ica_dir/melodic_mix" ]]; then
        echo "[$block_id] MELODIC not complete (no melodic_mix), skipping"
        continue
    fi

    if [[ ! -f "$label_txt" ]]; then
        echo "[$block_id] No label.txt found, skipping"
        continue
    fi

    total=$(( total + 1 ))

    if [[ -f "$output" ]]; then
        echo "[$block_id] Already cleaned, skipping"
        skipped=$(( skipped + 1 ))
        continue
    fi

    # Extract noise IC indices from the last non-empty line of label.txt
    # Expected format: [6, 9, 15, ...]
    noise_line=$(grep -E '^\[' "$label_txt" | tail -1)
    if [[ -z "$noise_line" ]]; then
        echo "[$block_id] No noise IC list found in label.txt, skipping"
        skipped=$(( skipped + 1 ))
        continue
    fi
    # Strip square brackets -> "6, 9, 15, ..."
    noise_ics="${noise_line//[\[\]]/}"
    noise_ics="$(echo "$noise_ics" | xargs)"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] $block_id — noise ICs: $noise_ics"
        continue
    fi

    echo ""
    echo "============================================================"
    echo " REGFILT: $block_id"
    echo " Noise ICs: $noise_ics"
    echo "============================================================"

    log="$LOG_DIR/${block_id}_regfilt.log"
    { echo "=== $block_id REGFILT ==="; date; echo "Noise ICs: $noise_ics"; } >> "$log"

    if fsl_regfilt \
            -i "$feat_out/filtered_func_data" \
            -o "$feat_out/filtered_func_data_clean" \
            -d "$ica_dir/melodic_mix" \
            -f "$noise_ics" \
        2>&1 | tee -a "$log"
    then
        echo "DONE $(date)" >> "$log"
        echo "[$block_id] DONE"
        done_count=$(( done_count + 1 ))
    else
        echo "FAILED $(date)" >> "$log"
        echo "[$block_id] FAILED — see $log"
        failed=$(( failed + 1 ))
    fi
done

echo ""
echo "=================================================="
echo "Summary: $total total, $skipped skipped, $done_count done, $failed failed"
echo "=================================================="
