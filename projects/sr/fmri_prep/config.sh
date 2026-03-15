#!/usr/bin/env bash
# config.sh
# =========
# Sourced by run_t1.sh and run_fmri.sh.
# Edit here if paths or scanner parameters change.

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_ROOT="/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina"
OUTPUT_ROOT="/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/mr_prep"

# ---------------------------------------------------------------------------
# Scanner parameters
# ---------------------------------------------------------------------------
DELTA_TE=2.46   # echo time difference (ms) for fsl_prepare_fieldmap

# ---------------------------------------------------------------------------
# Software modules
# ---------------------------------------------------------------------------
LMOD_INIT="/usr/share/lmod/lmod/init/bash"
FSL_MODULE="fsl/6.0.7.13"
FS_MODULE="freesurfer/7.4.1"
FS_HOME="/cvmfs/software.fmrib.ox.ac.uk/neuro/el9/software/freesurfer/7.4.1"
