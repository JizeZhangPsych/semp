#!/usr/bin/env python3
"""
run_fmri.py
===========
fMRI preprocessing for Staresina EEG-fMRI data on BMRC, submitted via SLURM.

For each block in block_list.txt:
  1. Symlink raw BOLD + SBRef as bold.nii / bold_SBREF.nii
  2. Motion outliers  : fsl_motion_outliers --fd --thresh=0.9
  3. Fieldmap prep    : BET + erode + fsl_prepare_fieldmap (no GDC)
  4. FSF generation   : sed substitution into design_preproc.fsf
  5. FEAT             : preprocessing only

Input BOLD:  {EPI_DIR}/{block}/epi_rest/{block}_epi_rest.nii.gz
Input fmap:  {FMAP_DIR}/{block}/fmap/{block}_magnitude1.nii.gz
Output:      {FUNC_DIR}/{block}/bold.feat/filtered_func_data.nii.gz

SLURM:   BATCH_SIZE blocks per job, run sequentially within each job.
Continue: blocks with filtered_func_data.nii.gz already present are skipped.
Partial outputs are cleaned before resubmission.

Prerequisites: run_t1.py must have completed for all subjects first.

Usage:
  python run_fmri.py
"""

import os
import shutil

import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from config import (MAINDIR, ANAT_DIR, EPI_DIR, FMAP_DIR, FUNC_DIR,
                    PIPELINE_DIR, DELTA_TE, BATCH_SIZE, QUEUE)

os.makedirs("logs",         exist_ok=True)
os.makedirs("subject_logs", exist_ok=True)

with open(f"{MAINDIR}/block_list.txt") as f:
    blocks = [l.strip() for l in f if l.strip()]


def is_complete(block):
    return os.path.isfile(
        f"{FUNC_DIR}/{block}/bold.feat/filtered_func_data.nii.gz"
    )


def cleanup_partial(block):
    block_dir = f"{FUNC_DIR}/{block}"
    for path in [
        f"{block_dir}/bold.feat",
        f"{block_dir}/fieldmap_proc",
        f"{block_dir}/bold.fsf",
    ]:
        if os.path.isdir(path):
            print(f"  Removing partial dir : {path}")
            shutil.rmtree(path)
        elif os.path.isfile(path):
            print(f"  Removing partial file: {path}")
            os.remove(path)
    log = f"subject_logs/{block}.log"
    if os.path.isfile(log):
        os.remove(log)


pending = [b for b in blocks if not is_complete(b)]
print(f"Total: {len(blocks)}  |  complete: {len(blocks)-len(pending)}  |  pending: {len(pending)}")

for b in pending:
    cleanup_partial(b)


def write_job(batch):
    name = f"fmri_{batch[0]}__to__{batch[-1]}"
    block_list_str = " ".join(batch)

    with open("job.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH -J {name}\n")
        f.write(f"#SBATCH -o logs/{name}.out\n")
        f.write(f"#SBATCH -e logs/{name}.err\n")
        f.write(f"#SBATCH -p {QUEUE}\n\n")
        f.write("module use -a /well/win/software/modules\n")
        f.write("module load fsl\n\n")

        f.write(f'EPI_DIR="{EPI_DIR}"\n')
        f.write(f'FMAP_DIR="{FMAP_DIR}"\n')
        f.write(f'FUNC_DIR="{FUNC_DIR}"\n')
        f.write(f'ANAT_DIR="{ANAT_DIR}"\n')
        f.write(f'PIPELINE_DIR="{PIPELINE_DIR}"\n')
        f.write(f'DELTA_TE={DELTA_TE}\n\n')

        # ---- process_block function ----------------------------------------
        f.write("process_block() {\n")
        f.write("  local block=$1\n")
        f.write("  local sub\n")
        f.write('  sub=$(echo "$block" | cut -d_ -f1)\n\n')

        f.write('  local block_dir="$FUNC_DIR/$block"\n')
        f.write('  local t1_brain="$ANAT_DIR/$sub/anat/${sub}_T1w.anat/T1_biascorr_brain.nii.gz"\n')
        f.write('  local fproc="$block_dir/fieldmap_proc"\n')
        f.write('  local mo_dir="$block_dir/motion_assess"\n')
        f.write('  local bold_lk="$block_dir/bold.nii"\n')
        f.write('  local sbref_lk="$block_dir/bold_SBREF.nii"\n')
        f.write('  mkdir -p "$block_dir"\n\n')

        f.write('  [[ -f "$t1_brain" ]] || { echo "ERROR: T1 not found — run run_t1.py first: $t1_brain" >&2; return 1; }\n\n')

        f.write('  # Step 1: symlink raw files (avoids copying large 4D BOLD data)\n')
        f.write('  local bold_raw="$EPI_DIR/$block/epi_rest/${block}_epi_rest.nii.gz"\n')
        f.write('  local sbref_raw="$EPI_DIR/$block/epi_rest/${block}_epi_rest_SBREF.nii.gz"\n')
        f.write('  [[ -e "$bold_lk" ]]  || ln -s "$bold_raw"  "$bold_lk"\n')
        f.write('  [[ -e "$sbref_lk" ]] || ln -s "$sbref_raw" "$sbref_lk"\n\n')

        f.write('  # Step 2: motion outlier detection\n')
        f.write('  mkdir -p "$mo_dir"\n')
        f.write('  if [[ ! -f "$mo_dir/confound.txt" ]]; then\n')
        f.write('    fsl_motion_outliers \\\n')
        f.write('      -i "$bold_lk" \\\n')
        f.write('      -o "$mo_dir/confound.txt" \\\n')
        f.write('      --fd --thresh=0.9 \\\n')
        f.write('      -p "$mo_dir/fd_plot" -v \\\n')
        f.write('      > "$mo_dir/outlier_output.txt" 2>&1 || true\n')
        f.write('    [[ -f "$mo_dir/confound.txt" ]] || touch "$mo_dir/confound.txt"\n')
        f.write('  fi\n\n')

        f.write('  # Step 3: fieldmap preparation (no GDC — new scanner)\n')
        f.write('  mkdir -p "$fproc"\n')
        f.write('  if [[ ! -f "$fproc/fmap_rads.nii.gz" ]]; then\n')
        f.write('    local fmap_mag1="$FMAP_DIR/$block/fmap/${block}_magnitude1.nii.gz"\n')
        f.write('    local fmap_pd="$FMAP_DIR/$block/fmap/${block}_phaseddiff.nii.gz"\n')
        f.write('    bet "$fmap_mag1" "$fproc/magnitude1_brain.nii.gz" -f 0.5 -g 0 -R\n')
        f.write('    fslmaths "$fproc/magnitude1_brain.nii.gz" -ero "$fproc/magnitude1_brain_ero.nii.gz"\n')
        f.write('    fsl_prepare_fieldmap SIEMENS \\\n')
        f.write('      "$fmap_pd" \\\n')
        f.write('      "$fproc/magnitude1_brain_ero.nii.gz" \\\n')
        f.write('      "$fproc/fmap_rads.nii.gz" \\\n')
        f.write('      "$DELTA_TE"\n')
        f.write('  fi\n\n')

        f.write('  # Step 4: FSF generation\n')
        f.write('  local feat_out="$block_dir/bold.feat"\n')
        f.write('  local npts; npts=$(fslval "$bold_lk" dim4)\n')
        f.write('  local tr;   tr=$(fslval   "$bold_lk" pixdim4)\n')
        f.write('  local mni_brain="${FSLDIR}/data/standard/MNI152_T1_2mm_brain"\n\n')

        f.write('  sed \\\n')
        f.write('    -e "s|@@OUTPUTDIR@@|${feat_out}|g" \\\n')
        f.write('    -e "s|@@TR@@|${tr}|g" \\\n')
        f.write('    -e "s|@@NPTS@@|${npts}|g" \\\n')
        f.write('    -e "s|@@FEAT_FILE@@|${bold_lk%.nii}|g" \\\n')
        f.write('    -e "s|@@ALT_EX_FUNC@@|${sbref_lk%.nii}|g" \\\n')
        f.write('    -e "s|@@UNWARP_FILE@@|${fproc}/fmap_rads|g" \\\n')
        f.write('    -e "s|@@UNWARP_MAG@@|${fproc}/magnitude1_brain|g" \\\n')
        f.write('    -e "s|@@HIGHRES@@|${t1_brain%.nii.gz}|g" \\\n')
        f.write('    -e "s|@@MNI_BRAIN@@|${mni_brain}|g" \\\n')
        f.write('    "$PIPELINE_DIR/design_preproc.fsf" > "$block_dir/bold.fsf"\n\n')

        f.write('  if [[ -s "$mo_dir/confound.txt" ]]; then\n')
        f.write('    sed -i "s|set fmri(confoundevs) 0|set fmri(confoundevs) 1|g" "$block_dir/bold.fsf"\n')
        f.write('    echo "set confoundev_files(1) \\"$mo_dir/confound.txt\\"" >> "$block_dir/bold.fsf"\n')
        f.write('  fi\n\n')

        f.write('  # Step 5: FEAT\n')
        f.write('  feat "$block_dir/bold.fsf"\n')
        f.write("}\n\n")

        # ---- main loop over batch -----------------------------------------
        f.write(f"for block in {block_list_str}; do\n")
        f.write('  log="subject_logs/${block}.log"\n')
        f.write('  { echo "=== $block ==="; date; } >> "$log"\n')
        f.write('  if process_block "$block" >> "$log" 2>&1; then\n')
        f.write('    echo "DONE $(date)" >> "$log"\n')
        f.write('  else\n')
        f.write('    echo "FAILED $(date)" >> "$log"\n')
        f.write('  fi\n')
        f.write("done\n")

    os.system("sbatch job.sh")
    os.system("rm job.sh")


for i in range(0, len(pending), BATCH_SIZE):
    batch = pending[i:i + BATCH_SIZE]
    write_job(batch)
