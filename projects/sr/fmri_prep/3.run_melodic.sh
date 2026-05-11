#!/usr/bin/env bash
# run_melodic.sh
# ==============
# MELODIC ICA decomposition for resting-state blocks, run after run_fmri.sh.
#
# For each resting block with a completed FEAT output:
#   - Runs melodic on filtered_func_data
#   - Output: bold.feat/filtered_func_data.ica/
#   - Skips blocks where melodic_IC.nii.gz already exists
#
# Usage:
#   bash run_melodic.sh
#   bash run_melodic.sh --subject sub-001
#   bash run_melodic.sh --block sub-001_ses-01_scanrun-01_resting_01
#   bash run_melodic.sh --dry-run

set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Allow overriding config file via --config <file> (default: config.sh)
CONFIG_FILE="$PIPELINE_DIR/config44.sh"
_prev=""
for _arg in "$@"; do
    if [[ "$_prev" == "--config" ]]; then CONFIG_FILE="$_arg"; fi
    _prev="$_arg"
done
unset _prev _arg
source "$CONFIG_FILE"

[[ -n "$LMOD_INIT" ]] && source "$LMOD_INIT"
module load "$FSL_MODULE"

FUNC_BASE="$OUTPUT_ROOT/func_crop5"
LOG_DIR="$PIPELINE_DIR/logs"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DRY_RUN=0
SUBJECT_FILTER=""
SUBJECT_PARITY=""   # "even" or "odd"
BLOCK_FILTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        --config)   shift 2 ;;  # already consumed above
        --subject)
            case "$2" in
                even|odd) SUBJECT_PARITY="$2" ;;
                *)        SUBJECT_FILTER="$2" ;;
            esac
            shift 2 ;;
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
    if [[ -n "$SUBJECT_PARITY" ]]; then
        sub_num=$(echo "$block_id" | grep -oP '(?<=sub-)\d+')
        remainder=$(( 10#$sub_num % 2 ))
        [[ "$SUBJECT_PARITY" == "even" && $remainder -ne 0 ]] && continue
        [[ "$SUBJECT_PARITY" == "odd"  && $remainder -eq 0 ]] && continue
    fi

    feat_out="$block_dir/bold.feat"
    filtered="$feat_out/filtered_func_data.nii.gz"
    ica_dir="$feat_out/filtered_func_data.ica"

    if [[ ! -f "$filtered" ]]; then
        echo "[$block_id] FEAT not complete, skipping"
        continue
    fi

    total=$(( total + 1 ))

    if [[ -f "$ica_dir/melodic_IC.nii.gz" ]]; then
        echo "[$block_id] MELODIC already complete, skipping"
        skipped=$(( skipped + 1 ))
        continue
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] $block_id"
        continue
    fi

    echo ""
    echo "============================================================"
    echo " MELODIC: $block_id"
    echo "============================================================"

    tr=$(fslval "$filtered" pixdim4 | xargs)
    log="$LOG_DIR/${block_id}_melodic.log"
    { echo "=== $block_id MELODIC ==="; date; } >> "$log"

    if melodic \
            -i "$feat_out/filtered_func_data" \
            -o "$ica_dir" \
            --tr="$tr" \
            --report --Oall \
            --vn \
            --nobet \
            --mmthresh=0.5 \
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
