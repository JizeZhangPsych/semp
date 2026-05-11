#!/usr/bin/env bash
# run_t1.sh
# =========
# T1 preprocessing for Staresina EEG-fMRI data (no SLURM — runs directly).
#
# For each unique subject (sub-XXX) in DATA_ROOT:
#   1. T1 combination : mri_robust_template --satit  (or mri_convert if only 1 T1)
#   2. T1 pipeline    : fsl_anat (bias correction, BET, segmentation)
#
# T1 discovery: one file per scanner run folder (lowest sequence number),
# across all sessions. Exclusions read from t1_combine_exclude.txt.
#
# Output per subject:
#   {OUTPUT_ROOT}/anat/sub-XXX/anat/sub-XXX_T1w.nii.gz
#   {OUTPUT_ROOT}/anat/sub-XXX/anat/sub-XXX_T1w.anat/T1_biascorr_brain.nii.gz
#
# Usage:
#   bash run_t1.sh
#   bash run_t1.sh --subject sub-001
#   bash run_t1.sh --dry-run

set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/config.sh"

# Load modules
source "$LMOD_INIT"
module load "$FS_MODULE"
export FREESURFER_HOME="$FS_HOME"
module load "$FSL_MODULE"

ANAT_BASE="$OUTPUT_ROOT/anat"
LOG_DIR="$PIPELINE_DIR/logs"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Read T1 exclusions
# ---------------------------------------------------------------------------
declare -A EXCLUDED
if [[ -f "$PIPELINE_DIR/t1_combine_exclude.txt" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        entry="${line%%#*}"; entry="${entry//[[:space:]]/}"
        [[ -n "$entry" ]] && EXCLUDED["$entry"]=1
    done < "$PIPELINE_DIR/t1_combine_exclude.txt"
    echo "T1 exclusions (${#EXCLUDED[@]}): $(printf '%s  ' "${!EXCLUDED[@]}")"
fi

# ---------------------------------------------------------------------------
# process_subject SUB T1_FILE [T1_FILE ...]
# ---------------------------------------------------------------------------
process_subject() {
    local sub="$1"; shift
    local t1_files=("$@")
    local out_dir="$ANAT_BASE/$sub/anat"
    local out_t1="$out_dir/${sub}_T1w.nii.gz"
    mkdir -p "$out_dir"

    if [[ ${#t1_files[@]} -eq 1 ]]; then
        echo "  1 T1 — converting"
        mri_convert "${t1_files[0]}" "$out_t1"
    else
        echo "  ${#t1_files[@]} T1s — mri_robust_template"
        mri_robust_template \
            --mov "${t1_files[@]}" \
            --template "$out_t1" \
            --satit
    fi

    echo "  Running fsl_anat ..."
    fsl_anat -i "$out_t1"
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DRY_RUN=0
SUBJECT_FILTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        --subject)  SUBJECT_FILTER="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
total=0; skipped=0; done_count=0; failed=0

for sub_dir in "$DATA_ROOT"/sub-*/; do
    [[ -d "$sub_dir" ]] || continue
    sub=$(basename "$sub_dir")
    [[ -n "$SUBJECT_FILTER" && "$sub" != "$SUBJECT_FILTER" ]] && continue
    total=$(( total + 1 ))

    if [[ -f "$ANAT_BASE/$sub/anat/${sub}_T1w.anat/T1_biascorr_brain.nii.gz" ]]; then
        echo "[$sub] already complete, skipping"
        skipped=$(( skipped + 1 ))
        continue
    fi

    # Collect one T1 per scanner run folder, skip excluded scanruns
    t1_files=()
    for ses_dir in "$sub_dir"ses-*/; do
        [[ -d "$ses_dir" ]] || continue
        ses=$(basename "$ses_dir")
        for run_dir in "$ses_dir"mri/run-*/; do
            [[ -d "$run_dir" ]] || continue
            run=$(basename "$run_dir")
            scanrun_id="${sub}_${ses}_${run}"
            if [[ -n "${EXCLUDED[$scanrun_id]+x}" ]]; then
                echo "  excluded: $scanrun_id"
                continue
            fi
            # First T1 (lowest sequence number) in the W3T folder (.nii or .nii.gz)
            t1=$(find "$run_dir" -maxdepth 2 \
                     \( -name "images_*_t1_mpr_*.nii" -o -name "images_*_t1_mpr_*.nii.gz" \) \
                     2>/dev/null | sort | head -1)
            [[ -n "$t1" ]] && t1_files+=("$t1")
        done
    done

    if [[ ${#t1_files[@]} -eq 0 ]]; then
        echo "[$sub] no T1 files found, skipping"
        skipped=$(( skipped + 1 ))
        continue
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] $sub: ${#t1_files[@]} T1(s)"
        for f in "${t1_files[@]}"; do echo "    $f"; done
        continue
    fi

    echo ""
    echo "=================================================="
    echo " T1 pipeline: $sub  (${#t1_files[@]} T1s)"
    echo "=================================================="

    log="$LOG_DIR/${sub}_t1.log"
    { echo "=== $sub ==="; date; } >> "$log"

    if process_subject "$sub" "${t1_files[@]}" 2>&1 | tee -a "$log"; then
        echo "DONE $(date)" >> "$log"
        echo "[$sub] DONE"
        done_count=$(( done_count + 1 ))
    else
        echo "FAILED $(date)" >> "$log"
        echo "[$sub] FAILED — see $log"
        failed=$(( failed + 1 ))
    fi

done

echo ""
echo "=================================================="
echo "Summary: $total total, $skipped skipped, $done_count done, $failed failed"
echo "=================================================="
