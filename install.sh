#!/usr/bin/env bash
#
# Install semp on top of an existing osl-ephys conda environment WITHOUT
# upgrading or downgrading anything osl-ephys already provides.
#
# Why a constraints file rather than pinning versions in pyproject.toml:
#   osl-ephys is installed from its own fully-pinned conda env file
#   (osl-ephys/envs/hbaws.yml). semp must not perturb that pinned set --
#   pip would otherwise happily pull a newer numpy/scipy/mne and break the
#   numba / source-recon stack. We snapshot whatever osl-ephys actually
#   installed (`pip freeze`) and feed it to pip as a constraints file
#   (`-c`). A constraints file pins versions but installs nothing on its
#   own, so:
#       * every package osl-ephys provided  -> left exactly as-is
#       * every extra package semp needs    -> installed normally
#   This is robust to future osl-ephys updates: we never hard-code which
#   packages it owns, we just snapshot the live env at install time.
#
# Usage:
#   conda activate osle          # the osl-ephys env (see hbaws.yml)
#   ./install.sh [/path/to/osl-manual-ica]
#
# If the osl-manual-ica path is omitted it defaults to ../osl-manual-ica
# relative to this script.

set -euo pipefail

SEMP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMICA_DIR="${1:-$(cd "$SEMP_DIR/.." && pwd)/osl-manual-ica}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: no conda environment is active. 'conda activate <osle-env>' first." >&2
    exit 1
fi

echo ">> Active environment: $CONDA_PREFIX"
echo ">> Checking osl-ephys is importable..."
python -c "import osl_ephys" || {
    echo "ERROR: osl_ephys not importable in this env. Create it first with:" >&2
    echo "       conda env create -f /ohba/pi/mwoolrich/jzhang/osl-ephys/envs/hbaws.yml" >&2
    exit 1
}

LOCK="$(mktemp -t osle-lock.XXXXXX.txt)"
trap 'rm -f "$LOCK"' EXIT

echo ">> Snapshotting current env to constraints file: $LOCK"
# Use `pip list --format=freeze`, NOT `pip freeze`: in a conda env `pip
# freeze` emits conda packages as `name @ file:///home/conda/feedstock_root/
# build_artifacts/...` direct-reference URLs (build-time paths that don't
# exist later), which break when fed back as a constraints file. `pip list
# --format=freeze` emits plain `name==version` for every package.
pip list --format=freeze --exclude-editable > "$LOCK"

echo ">> Installing osl-manual-ica (editable) under constraints..."
pip install -c "$LOCK" -e "$OMICA_DIR"

echo ">> Installing semp (editable) under constraints..."
pip install -c "$LOCK" -e "$SEMP_DIR"

echo ">> Done. Verifying..."
python -c "import semp; import osl_manual_ica; import osl_ephys" \
    && echo ">> OK: semp + osl-manual-ica + osl-ephys all import cleanly."
