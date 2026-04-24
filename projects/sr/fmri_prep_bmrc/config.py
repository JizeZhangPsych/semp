# config.py
# =========
# Central configuration for fmri_prep_bmrc.
# Edit here if paths or scanner parameters change.

import os

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
MAINDIR    = "/well/woolrich/projects/eeg-fmri_Staresina/mr"
ANAT_DIR   = f"{MAINDIR}/anat"      # per-session T1 input & per-subject T1 output
EPI_DIR    = f"{MAINDIR}/epi_rest"  # raw BOLD/SBRef per block
FMAP_DIR   = f"{MAINDIR}/fmap_rest" # fieldmaps per block
FUNC_DIR   = f"{MAINDIR}/func"      # FEAT output per block

PIPELINE_DIR = os.path.dirname(os.path.realpath(__file__))

# ---------------------------------------------------------------------------
# Scanner parameters
# ---------------------------------------------------------------------------
DELTA_TE = 2.46  # echo time difference (ms) for fsl_prepare_fieldmap

# ---------------------------------------------------------------------------
# SLURM settings
# ---------------------------------------------------------------------------
BATCH_SIZE = 10
QUEUE      = "short"
