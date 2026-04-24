# fMRI Preprocessing Pipeline — Staresina EEG-fMRI Dataset

## Purpose
4-stage FSL pipeline to preprocess resting-state fMRI BOLD data for the Staresina EEG-fMRI dataset.

## Key Paths
- **Raw data:** `/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/`
- **Outputs:** `/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/mr_prep/`
- **Config:** `config.sh` — sourced by all scripts; defines paths, FSL/FreeSurfer modules, scanner params

## Pipeline Stages

| Script | Stage | Input | Output |
|--------|-------|-------|--------|
| `run_t1.sh` | T1 anatomical preprocessing | Raw T1 NIFTIs | `mr_prep/anat/sub-XXX/anat/sub-XXX_T1w.anat/` |
| `run_fmri.sh` | BOLD preprocessing (FEAT) | BOLD + SBRef + fieldmap + T1 brain | `bold.feat/filtered_func_data.nii.gz` |
| `run_melodic.sh` | MELODIC ICA decomposition | `filtered_func_data.nii.gz` | `bold.feat/filtered_func_data.ica/` |
| `run_regfilt.sh` | Noise IC regression | ICA output + `label.txt` | `bold.feat/filtered_func_data_clean.nii.gz` |

## Block ID Convention
`sub-XXX_ses-YY_scanrun-ZZ_resting_NN` — used for both output directories and log filenames.

## Scanner Parameters (Siemens, new scanner)
- DELTA_TE = 2.46 ms, EPI dwell = 0.52 ms, TE = 30 ms
- Unwarp direction: y-, phase-encode: anteroposterior
- No gradient distortion correction (GDC disabled for new scanner)
- No slice timing correction (multiband EPI)

## FEAT Settings (`design_preproc.fsf` template)
- Preprocessing only (no stats/ICA inside FEAT)
- Motion correction: MCFLIRT, FD outlier threshold = 0.9 mm
- Smoothing: 5 mm FWHM, highpass: 100 s
- Registration: BBR to T1 (using SBRef as alternate reference), nonlinear to MNI152 2mm
- Fieldmap unwarping: enabled

## Noise Regression Convention
`label.txt` per block (OHBA format): last line is a Python list of noise IC indices, e.g. `[2, 7, 13]`. Required before running `run_regfilt.sh`.

## Common CLI Options (all scripts)
```bash
--subject sub-001     # Process one subject
--block <block_id>    # Process one block
--task resting        # Filter by task
--dry-run             # Preview without executing
```
All scripts support **continue mode**: skip blocks where the output file already exists.

## Status (as of pipeline development)
- T1 preprocessing: 43 subjects complete
- BOLD FEAT preprocessing: ~130+ blocks, mostly complete
- MELODIC ICA: 22 blocks complete
- Noise regression: pilot stage (awaiting `label.txt` files)
- Excluded T1: `sub-006_ses-01_run-01` (corrupted, listed in `t1_combine_exclude.txt`)

## Known Quirks
- `.nfs*` file in directory is an NFS artifact (duplicate of `run_fmri.sh`), safe to ignore
- MELODIC logs may show `--tr: Couldn't set_value!` warning — non-critical, runs complete successfully
