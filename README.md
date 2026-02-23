# SEMP v2.0: Simultaneous EEG-fMRI Preprocessing Toolbox

Tools for analysing EEG acquired during simultaneous EEG-fMRI experiments.

This package is built on top of [osl-ephys](https://github.com/OHBA-analysis/osl-ephys) and provides EEG-fMRI specific preprocessing and analysis tools. Visualisation of source-space results additionally requires [osl-dynamics](https://github.com/OHBA-analysis/osl-dynamics).

## Package Structure

```
semp/
├── src/semp/               # installable package
│   ├── eeg/                # EEG signal processing and preprocessing wrappers
│   ├── utils/              # pathfinder, I/O, metrics, osl-ephys expansions
│   ├── visualize/          # statistical analysis and visualisation
│   └── fmri/               # fMRI utilities (in development)
└── projects/               # user analysis scripts (not part of the package)
    ├── template/           # step-by-step tutorial for building a new project
    ├── sr/                 # Staresina resting-state EEG-fMRI project
    └── wmt/                # WMT project
```

### What is available in each environment

| Module | osl-ephys env (default) | osl-dynamics env |
|--------|:-----------------------:|:----------------:|
| `semp.eeg` (helpers, metrics) | ✓ | ✓ |
| `semp.eeg` (prep/src wrappers) | ✓ | — |
| `semp.utils` (pathfinder, I/O, metrics) | ✓ | ✓ |
| `semp.utils.osle_expansion` | ✓ | — |
| `semp.visualize` (statistics, array ops) | ✓ | ✓ |
| `semp.visualize.visualize` (power/connectivity maps) | — | ✓ |

## Installation

### Step 1 — Install osl-ephys

Follow the installation instructions on the [osl-ephys GitHub page](https://github.com/OHBA-analysis/osl-ephys) to set up a conda environment with osl-ephys.

### Step 2 — Install osl-dynamics (optional)

osl-dynamics is only required for the visualisation module (`semp.visualize.visualize`). If needed, install it either into the same or a separate conda environment by following the [osl-dynamics GitHub page](https://github.com/OHBA-analysis/osl-dynamics):

```bash
pip install osl-dynamics
```

> **Note:** osl-ephys and osl-dynamics are not designed to coexist in the same conda environment so bugs might occur. semp detects which is installed and enables the appropriate modules automatically.

### Step 3 — Install semp

Inside your conda environment (osl-ephys or osl-dynamics or both), run:

```bash
pip install -e /path/to/semp
```

After installation, verify with:
```python
import semp
# semp v2.0 loaded [osl-ephys only]  (or whichever mode)
```

## Usage

`semp` is a library, not a script runner. Analysis scripts live under `projects/`. Each project folder follows the same pattern:

```
projects/<project>/
├── pathfinder.py     # defines where the data lives on disk
├── helpers.py        # project-specific preprocessing helpers
├── 1.prep.py         # step 1: preprocessing
├── 2.src.py          # step 2: source reconstruction
└── ...
```

See [`projects/template/1.prep.ipynb`](projects/template/1.prep.ipynb) for a step-by-step tutorial on adapting the template into a working pipeline for your own dataset.

### Staresina resting-state dataset

Expected directory structure on disk:

```
/path/to/base/
├── eeg-fmri_Staresina/
│   ├── edfs/           # raw .edf files
│   ├── sub-*/          # raw data (required for polhemus, EEG channel layout, etc.)
│   └── after_prep_sr/  # preprocessed output
```