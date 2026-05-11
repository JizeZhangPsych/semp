#!/usr/bin/env bash
# apply_fix_40.sh
# ===============
# Apply a trained FIX classifier at threshold 40 (conservative).
# Runs alongside 4c (threshold 20) to produce a second cleaned output.
# Equivalent to: fix <feat_dir> <model> 40 -m -h 100
#
# Output: bold.feat/filtered_func_data_clean_thres40.nii.gz
# Skips blocks where the output already exists.
#
# Usage:
#   bash apply_fix.sh --model <path/to/model.pyfix_model>
#   bash apply_fix.sh --model <model> --subject sub-001
#   bash apply_fix.sh --model <model> --block sub-001_ses-01_scanrun-01_resting_01
#   bash apply_fix.sh --model <model> --dry-run

set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/config.sh"

source "$LMOD_INIT"
module load "$FSL_MODULE"

FUNC_BASE="$OUTPUT_ROOT/func_crop5"
LOG_DIR="$PIPELINE_DIR/logs"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DRY_RUN=0
SUBJECT_FILTER=""
BLOCK_FILTER=""
MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        --subject)  SUBJECT_FILTER="$2"; shift 2 ;;
        --block)    BLOCK_FILTER="$2"; shift 2 ;;
        --model)    MODEL="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Usage: bash apply_fix.sh --model <path/to/model.pyfix_model>"
    exit 1
fi

# ---------------------------------------------------------------------------
# Main loop: resting blocks only
# ---------------------------------------------------------------------------
total=0; skipped=0; done_count=0; failed=0

for block_dir in "$FUNC_BASE"/*_resting_*/; do
    [[ -d "$block_dir" ]] || continue
    block_id=$(basename "$block_dir")

    [[ -n "$SUBJECT_FILTER" && "$block_id" != "${SUBJECT_FILTER}"* ]] && continue
    [[ -n "$BLOCK_FILTER"   && "$block_id" != "$BLOCK_FILTER"      ]] && continue

    feat_dir="$block_dir/bold.feat"
    output="$feat_dir/filtered_func_data_clean_thres40.nii.gz"

    if [[ ! -f "$feat_dir/filtered_func_data.ica/melodic_mix" ]]; then
        echo "[$block_id] MELODIC not complete, skipping"
        continue
    fi

    total=$(( total + 1 ))

    if [[ -f "$output" ]]; then
        echo "[$block_id] Already cleaned, skipping"
        skipped=$(( skipped + 1 ))
        continue
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] $block_id"
        continue
    fi

    echo ""
    echo "============================================================"
    echo " FIX: $block_id"
    echo "============================================================"

    log="$LOG_DIR/${block_id}_fix40.log"
    { echo "=== $block_id FIX thres40 ==="; date; } > "$log"

    # Protect 4c's thres20 output before FIX overwrites it
    fix20="$feat_dir/filtered_func_data_clean_fix20.nii.gz"
    if [[ -f "$feat_dir/filtered_func_data_clean.nii.gz" && ! -f "$fix20" ]]; then
        mv "$feat_dir/filtered_func_data_clean.nii.gz" "$fix20"
        echo "[$block_id] Renamed thres20 output -> $(basename $fix20)"
    fi

    if fix "$feat_dir" "$MODEL" 40 -m -h 100 2>&1 | tee -a "$log"; then
        mv "$feat_dir/filtered_func_data_clean.nii.gz" "$output"
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
