# SEMP: Simultaneous EEG-fMRI Preprocessing Toolbox

Tools for analysing EEG acquired during simultaneous EEG-fMRI experiments.

This package is built on top of [osl-ephys](https://github.com/OHBA-analysis/osl-ephys) and provides EEG-fMRI specific preprocessing and analysis tools. The NIfTI mask + parcellation atlases used for source-space visualisation come from osl-ephys's bundled `source_recon/files/` directory, so installing osl-ephys is sufficient — no separate atlas download.

The interactive manual-ICA review (browser-based per-IC inspection + label/bad-segment server) lives in the standalone [osl-manual-ica](https://github.com/) package; semp re-exports its `manual_ica` wrapper so existing configs keep working unchanged.

## Package Structure

```
semp/
├── src/semp/               # installable package
│   ├── preprocessing/      # EEG signal processing and preprocessing wrappers
│   ├── source_recon/       # source reconstruction wrappers
│   ├── utils/              # pathfinder, I/O, metrics, parcellation/brain-surface plot helpers
│   └── visualize/          # statistical analysis and visualisation
└── projects/               # user analysis scripts (not part of the package)
    ├── template/           # step-by-step tutorial for building a new project
    ├── sr/                 # Staresina resting-state EEG-fMRI (auto-ICA)
    ├── sr_manual/          # Staresina resting-state EEG-fMRI (manual-ICA)
    └── wmt/                # WMT project
```

## Installation

### Step 1 — Install osl-ephys

Follow the installation instructions on the [osl-ephys GitHub page](https://github.com/OHBA-analysis/osl-ephys) to set up a conda environment with osl-ephys.

### Step 2 — Install osl-manual-ica

semp depends on osl-manual-ica for the interactive ICA-review wrapper. If it
is not yet on PyPI in your environment, install the local checkout in editable
mode first:

```bash
pip install -e /path/to/osl-manual-ica
```

`pip install semp` will pull it automatically once osl-manual-ica is published.

### Step 3 — Install semp

Inside the osl-ephys conda environment, run:

```bash
pip install -e /path/to/semp
```

After installation, verify with:
```python
import semp
# semp v2.0 loaded [osl-ephys + core]  (or [core only] without osl-ephys)
import osl_manual_ica
# manual ICA review wrapper available
```

> **Note:** The brain-surface and connectome plots in
> `semp.utils.parcel_plot` need a few NIfTI files (MNI brain mask +
> parcellation atlases). These are picked up automatically from
> `osl_ephys/source_recon/files/` --- no extra setup required as long as
> osl-ephys is installed. To use a custom atlas, pass an absolute path.

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

Two preprocessing pipelines are provided:

| Project | ICA approach | Output directory |
|---------|:------------:|:----------------:|
| `sr/` | Automated (slice ICA + osl-ephys auto-reject) | `after_prep_sr/` |
| `sr_manual/` | Manual review (`manual_ica` + `apply_ica`) | `after_prep_sr_manual/` |

Expected directory structure on disk:

```
/ohba/pi/mwoolrich/datasets/oxford/staresina/eeg_fmri/
├── edfs/           # raw .edf files
└── after_prep_sr/  # preprocessed output (auto-ICA)

/ohba/pi/mwoolrich/raw_datasets/oxford/staresina/eeg_fmri/
└── sub-*/          # raw data (required for polhemus, EEG channel layout)
```

### Manual ICA workflow (`sr_manual`)

The manual pipeline is a three-stage hand-off between an automated batch and a
human reviewer. The interactive parts (HTML review pages + the small label-saving
HTTP server) live in the [osl-manual-ica](https://github.com/) package;
`semp.preprocessing.manual_ica` is just a re-export of `osl_manual_ica.manual_ica`.

1. **`1.prep.py`** — `run_proc_batch` fits the ICA and writes the per-subject
   review pages to `{target_pth}/ica/{subject}/`. **No ICs are removed yet** —
   the wrapper only fits + plots, so subsequent steps in the same config
   (`interpolate_bads`, `set_eeg_reference`) operate on uncleaned data. The
   actual cleanup happens in step 3.

   ```python
   from semp.preprocessing import manual_ica
   config = {'preproc': [
       ...,
       {'manual_ica': {'n_components': 0.999, 'picks': 'eeg', 'l_freq': 1}},
       ...,
   ]}
   ```

2. **Review in the browser.** Start the bundled server inside the IC root and
   open the page:

   ```bash
   cd {target_pth}/ica
   osl-ica-review 8000          # python -m osl_manual_ica.review_server 8000
   # open http://localhost:8000/<subject>/single_ic.html
   ```

   The page persists `label.txt` (per-IC `good`/`bad`/`unsure`) and `bads.txt`
   (manual bad time segments) to disk via two POST endpoints. By default the
   server binds to `127.0.0.1` only; pass `--host 0.0.0.0` to expose it.

3. **`2.ica.py`** — reads `label.txt` + `bads.txt`, applies the bad-IC list
   and bad-segment annotations, writes `<subject>_after_ica-raw.fif`. It uses
   `osl_manual_ica.parse_label_txt` / `parse_bads_txt` (single source of truth)
   and bounds-checks IC indices against `ica.n_components_` so a typo'd
   `IC999: bad` against a 50-component fit fails loud instead of silently
   propagating to MNE.
