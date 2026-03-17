#!/usr/bin/env bash
# run_fmri.sh
# ===========
# fMRI preprocessing for Staresina EEG-fMRI data (no SLURM — runs directly).
#
# For each BOLD run discovered in DATA_ROOT:
#   1. Symlink raw BOLD + SBRef into the block output directory
#   2. Motion outliers  : fsl_motion_outliers --fd --thresh=0.9
#   3. Fieldmap prep    : BET + erode + fsl_prepare_fieldmap (no GDC)
#   4. FSF generation   : sed substitution into design_preproc.fsf
#   5. FEAT             : preprocessing only
#
# Block IDs: sub-XXX_ses-YY_scanrun-ZZ_TASK_NN
# Continue mode: blocks with bold.feat/filtered_func_data.nii.gz are skipped.
#
# Prerequisites: run_t1.sh must have completed for all subjects first.
#
# Usage:
#   bash run_fmri.sh
#   bash run_fmri.sh --subject sub-001
#   bash run_fmri.sh --task resting
#   bash run_fmri.sh --block sub-001_ses-01_scanrun-01_resting_01
#   bash run_fmri.sh --dry-run

set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/config.sh"

# Load FSL
source "$LMOD_INIT"
module load "$FSL_MODULE"

FUNC_BASE="$OUTPUT_ROOT/func"
ANAT_BASE="$OUTPUT_ROOT/anat"
LOG_DIR="$PIPELINE_DIR/logs"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Ordered task suffix → short label mapping
# Order determines block IDs when the same suffix appears multiple times.
# ---------------------------------------------------------------------------
TASK_SUFFIXES=(
    Resting_State
)

task_label() {
    case "$1" in
        Resting_State)      echo "resting" ;;
        ObScAL_encoding1)   echo "encoding-01" ;;
        ObScAL_encoding2)   echo "encoding-02" ;;
        ObScAL_encoding3)   echo "encoding-03" ;;
        ObScAL_encoding4)   echo "encoding-04" ;;
        ObScAL_retrieval1)  echo "retrieval-01" ;;
        ObScAL_retrieval2)  echo "retrieval-02" ;;
        ObScAL_retrieval3)  echo "retrieval-03" ;;
        ObScAL_retrieval4)  echo "retrieval-04" ;;
        motor_loc)          echo "motorloc" ;;
        ObScAL_func_loc1)   echo "funcloc-01" ;;
        ObScAL_func_loc2)   echo "funcloc-02" ;;
        *)                  echo "$1" ;;
    esac
}

# ---------------------------------------------------------------------------
# process_block BLOCK_ID SUB BOLD_RAW SBREF_RAW FMAP_MAG1 FMAP_PD
# ---------------------------------------------------------------------------
process_block() {
    local block_id="$1" sub="$2" bold_raw="$3" sbref_raw="$4" \
          fmap_mag1="$5" fmap_pd="$6"

    local block_dir="$FUNC_BASE/$block_id"
    local t1_brain="$ANAT_BASE/$sub/anat/${sub}_T1w.anat/T1_biascorr_brain.nii.gz"
    local fproc="$block_dir/fieldmap_proc"
    local mo_dir="$block_dir/motion_assess"
    local bold_lk="$block_dir/bold.nii"
    local sbref_lk="$block_dir/bold_SBREF.nii"
    mkdir -p "$block_dir"

    [[ -f "$t1_brain" ]] || { echo "  ERROR: T1 not found — run run_t1.sh first: $t1_brain" >&2; return 1; }

    # Step 1: symlink raw files (avoids copying large 4D BOLD data)
    [[ -e "$bold_lk" ]]  || ln -s "$bold_raw"  "$bold_lk"
    [[ -e "$sbref_lk" ]] || ln -s "$sbref_raw" "$sbref_lk"

    # Step 2: motion outlier detection
    mkdir -p "$mo_dir"
    if [[ ! -f "$mo_dir/confound.txt" ]]; then
        echo "  Motion outliers ..."
        fsl_motion_outliers \
            -i "$bold_lk" \
            -o "$mo_dir/confound.txt" \
            --fd --thresh=0.9 \
            -p "$mo_dir/fd_plot" -v \
            > "$mo_dir/outlier_output.txt" 2>&1 || true
        [[ -f "$mo_dir/confound.txt" ]] || touch "$mo_dir/confound.txt"
    fi

    # Step 3: fieldmap preparation (no GDC — new scanner)
    mkdir -p "$fproc"
    if [[ ! -f "$fproc/fmap_rads.nii.gz" ]]; then
        echo "  Fieldmap prep ..."
        bet "$fmap_mag1" "$fproc/magnitude1_brain.nii.gz" -f 0.5 -g 0 -R
        fslmaths "$fproc/magnitude1_brain.nii.gz" -ero "$fproc/magnitude1_brain_ero.nii.gz"
        fsl_prepare_fieldmap SIEMENS \
            "$fmap_pd" \
            "$fproc/magnitude1_brain_ero.nii.gz" \
            "$fproc/fmap_rads.nii.gz" \
            "$DELTA_TE"
    fi

    # Step 4: FSF generation
    local feat_out="$block_dir/bold.feat"
    local npts tr
    npts=$(fslval "$bold_lk" dim4)
    tr=$(fslval   "$bold_lk" pixdim4)

    sed \
        -e "s|@@OUTPUTDIR@@|${feat_out}|g" \
        -e "s|@@TR@@|${tr}|g" \
        -e "s|@@NPTS@@|${npts}|g" \
        -e "s|@@FEAT_FILE@@|${bold_lk%.nii}|g" \
        -e "s|@@ALT_EX_FUNC@@|${sbref_lk%.nii}|g" \
        -e "s|@@UNWARP_FILE@@|${fproc}/fmap_rads|g" \
        -e "s|@@UNWARP_MAG@@|${fproc}/magnitude1_brain|g" \
        -e "s|@@HIGHRES@@|${t1_brain%.nii.gz}|g" \
        "$PIPELINE_DIR/design_preproc.fsf" > "$block_dir/bold.fsf"

    if [[ -s "$mo_dir/confound.txt" ]]; then
        sed -i "s|set fmri(confoundevs) 0|set fmri(confoundevs) 1|g" "$block_dir/bold.fsf"
        echo "set confoundev_files(1) \"$mo_dir/confound.txt\"" >> "$block_dir/bold.fsf"
    fi

    # Step 5: FEAT
    echo "  Running FEAT ..."
    feat "$block_dir/bold.fsf"
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DRY_RUN=0
SUBJECT_FILTER=""
TASK_FILTER=""
BLOCK_FILTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1; shift ;;
        --subject)  SUBJECT_FILTER="$2"; shift 2 ;;
        --task)     TASK_FILTER="$2"; shift 2 ;;
        --block)    BLOCK_FILTER="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Main loop: iterate sub → ses → run → W3T → task → BOLD pairs
# ---------------------------------------------------------------------------
total=0; skipped=0; done_count=0; failed=0

for sub_dir in "$DATA_ROOT"/sub-*/; do
    [[ -d "$sub_dir" ]] || continue
    sub=$(basename "$sub_dir")
    [[ -n "$SUBJECT_FILTER" && "$sub" != "$SUBJECT_FILTER" ]] && continue

    for ses_dir in "$sub_dir"ses-*/; do
        [[ -d "$ses_dir" ]] || continue
        ses=$(basename "$ses_dir")

        for run_dir in "$ses_dir"mri/run-*/; do
            [[ -d "$run_dir" ]] || continue
            run=$(basename "$run_dir")
            run_num="${run#run-}"

            # Locate the single W3T_* subdirectory
            w3t_dir=$(find "$run_dir" -maxdepth 1 -name "W3T_*" -type d 2>/dev/null | sort | head -1)
            [[ -z "$w3t_dir" ]] && continue

            # Fieldmap for this scanner run (shared by all blocks within it)
            fmap_mag1=$(find "$w3t_dir" -name "images_*_fieldmap_*_e1.nii"    2>/dev/null | sort | head -1)
            fmap_pd=$(  find "$w3t_dir" -name "images_*_fieldmap_*_e2_ph.nii" 2>/dev/null | sort | head -1)
            if [[ -z "$fmap_mag1" || -z "$fmap_pd" ]]; then
                echo "  WARNING: no fieldmap in $w3t_dir, skipping"
                continue
            fi

            # Iterate task types in defined order
            for task_suffix in "${TASK_SUFFIXES[@]}"; do
                label=$(task_label "$task_suffix")
                [[ -n "$TASK_FILTER" && "$label" != "$TASK_FILTER" ]] && continue

                mapfile -t bold_files < <(
                    find "$w3t_dir" -name "images_*_BOLD_*_${task_suffix}.nii" 2>/dev/null | sort
                )
                [[ ${#bold_files[@]} -eq 0 ]] && continue

                for (( i=0; i<${#bold_files[@]}; i+=2 )); do
                    task_idx=$(( i/2 + 1 ))
                    block_id="${sub}_${ses}_scanrun-${run_num}_${label}_$(printf '%02d' "$task_idx")"
                    [[ -n "$BLOCK_FILTER" && "$block_id" != "$BLOCK_FILTER" ]] && continue

                    total=$(( total + 1 ))

                    if [[ -f "$FUNC_BASE/$block_id/bold.feat/filtered_func_data.nii.gz" ]]; then
                        echo "[$block_id] already complete, skipping"
                        skipped=$(( skipped + 1 ))
                        continue
                    fi

                    sbref="${bold_files[$i]}"
                    bold="${bold_files[$(( i+1 ))]}"

                    if [[ $DRY_RUN -eq 1 ]]; then
                        echo "[DRY-RUN] $block_id"
                        continue
                    fi

                    echo ""
                    echo "============================================================"
                    echo " fMRI pipeline: $block_id"
                    echo "============================================================"

                    log="$LOG_DIR/${block_id}.log"
                    { echo "=== $block_id ==="; date; } >> "$log"

                    if process_block "$block_id" "$sub" "$bold" "$sbref" \
                                     "$fmap_mag1" "$fmap_pd" 2>&1 | tee -a "$log"; then
                        echo "DONE $(date)" >> "$log"
                        echo "[$block_id] DONE"
                        done_count=$(( done_count + 1 ))
                    else
                        echo "FAILED $(date)" >> "$log"
                        echo "[$block_id] FAILED — see $log"
                        failed=$(( failed + 1 ))
                    fi
                done
            done
        done
    done
done

echo ""
echo "=================================================="
echo "Summary: $total total, $skipped skipped, $done_count done, $failed failed"
echo "=================================================="
