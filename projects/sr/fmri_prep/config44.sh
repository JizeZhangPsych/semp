#!/usr/bin/env bash
# config44.sh
# ===========
# Like config.sh but for hbaws44, which has fsl/6.0.7.9 (not 6.0.7.13).
# Source this instead of config.sh when running on hbaws44.

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_ROOT="/ohba/pi/mwoolrich/datasets/staresina/eeg_fmri"
OUTPUT_ROOT="/ohba/pi/mwoolrich/datasets/staresina/eeg_fmri/mr_prep"

# ---------------------------------------------------------------------------
# Scanner parameters
# ---------------------------------------------------------------------------
DELTA_TE=2.46   # echo time difference (ms) for fsl_prepare_fieldmap

# ---------------------------------------------------------------------------
# Software modules
# ---------------------------------------------------------------------------
LMOD_INIT=""           # not needed on hbaws44; module is already a shell function
FSL_MODULE="fsl/6.0.7.9"
FS_MODULE="freesurfer/7.4.1"
FS_HOME=""             # FreeSurfer not used by run_melodic.sh
