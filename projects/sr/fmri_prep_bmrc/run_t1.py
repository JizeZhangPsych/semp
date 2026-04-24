#!/usr/bin/env python3
"""
run_t1.py
=========
T1 preprocessing for Staresina EEG-fMRI data on BMRC, submitted via SLURM.

For each unique subject (sub-XXX) in session_list.txt:
  1. T1 combination : mri_robust_template --satit  (single T1: cp)
  2. T1 pipeline    : fsl_anat (bias correction, BET, segmentation)

Input T1s:  {ANAT_DIR}/{sess}/anat/{sess}_T1w.nii.gz
Output:     {ANAT_DIR}/{sub}/anat/{sub}_T1w.anat/T1_biascorr_brain.nii.gz

Exclusions: sessions listed in t1_combine_exclude.txt are skipped.
SLURM:      BATCH_SIZE subjects per job, run sequentially within each job.
Continue:   subjects with T1_biascorr_brain.nii.gz already present are skipped.

Usage:
  python run_t1.py
"""

import os
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from config import MAINDIR, ANAT_DIR, PIPELINE_DIR, BATCH_SIZE, QUEUE

os.makedirs("logs",         exist_ok=True)
os.makedirs("subject_logs", exist_ok=True)

# --- Read session list ------------------------------------------------------
with open(f"{MAINDIR}/session_list.txt") as f:
    sessions = [l.strip() for l in f if l.strip()]

# --- Read exclusion list ----------------------------------------------------
excluded = set()
exclude_file = os.path.join(PIPELINE_DIR, "t1_combine_exclude.txt")
if os.path.exists(exclude_file):
    with open(exclude_file) as f:
        excluded = {l.split("#")[0].strip() for l in f
                    if l.strip() and not l.startswith("#")}
    if excluded:
        print(f"Excluding {len(excluded)} session(s): {sorted(excluded)}")

# --- Group sessions by subject ----------------------------------------------
subject_sessions = defaultdict(list)
for sess in sessions:
    sub = sess.split("_")[0]
    subject_sessions[sub].append(sess)


def is_complete(sub):
    return os.path.isfile(
        f"{ANAT_DIR}/{sub}/anat/{sub}_T1w.anat/T1_biascorr_brain.nii.gz"
    )


def write_job(subject_batch, t1_map):
    name = f"T1_{subject_batch[0]}__to__{subject_batch[-1]}"
    with open("job.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH -J {name}\n")
        f.write(f"#SBATCH -o logs/{name}.out\n")
        f.write(f"#SBATCH -e logs/{name}.err\n")
        f.write(f"#SBATCH -p {QUEUE}\n\n")
        f.write("module use -a /well/win/software/modules\n")
        f.write("module load fsl\n")
        f.write("module load FreeSurfer\n\n")

        for sub in subject_batch:
            t1_files = t1_map[sub]
            out_dir  = f"{ANAT_DIR}/{sub}/anat"
            out_t1   = f"{out_dir}/{sub}_T1w.nii.gz"
            log      = f"subject_logs/{sub}_t1.log"

            f.write(f'{{ echo "=== {sub} ==="; date; }} >> "{log}"\n')
            f.write(f'mkdir -p "{out_dir}"\n\n')

            if len(t1_files) == 1:
                f.write(f'cp "{t1_files[0]}" "{out_t1}"\n')
            else:
                mov_args = " \\\n    ".join(f'"{p}"' for p in t1_files)
                f.write(f'mri_robust_template \\\n')
                f.write(f'  --mov {mov_args} \\\n')
                f.write(f'  --template "{out_t1}" \\\n')
                f.write(f'  --satit\n')

            f.write(f'\nfsl_anat -i "{out_t1}" >> "{log}" 2>&1\n')
            f.write(f'echo "DONE $(date)" >> "{log}"\n\n')

    os.system("sbatch job.sh")
    os.system("rm job.sh")


# --- Filter subjects and collect T1 file lists ------------------------------
pending = []
t1_map  = {}

for sub in sorted(subject_sessions):
    if is_complete(sub):
        print(f"{sub}: already complete, skipping")
        continue

    t1_files = []
    for sess in sorted(subject_sessions[sub]):
        if sess in excluded:
            print(f"  {sess}: excluded")
            continue
        t1 = f"{ANAT_DIR}/{sess}/anat/{sess}_T1w.nii.gz"
        if os.path.isfile(t1):
            t1_files.append(t1)
        else:
            print(f"  WARNING: T1w not found for {sess}, skipping")

    if not t1_files:
        print(f"{sub}: no T1w images found after exclusions, skipping")
        continue

    pending.append(sub)
    t1_map[sub] = t1_files
    print(f"{sub}: pending ({len(t1_files)} T1s)")

print(f"\nTotal: {len(subject_sessions)}  |  pending: {len(pending)}")

# --- Submit SLURM jobs in batches -------------------------------------------
for i in range(0, len(pending), BATCH_SIZE):
    batch = pending[i:i + BATCH_SIZE]
    write_job(batch, t1_map)
