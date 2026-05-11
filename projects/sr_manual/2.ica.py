"""Apply manual-ICA decisions to the prepped raw fif files.

For every subject under PREP_PTH that has a finished ``label.txt`` (lines like
``IC000: bad`` / ``good`` / ``unsure``), this script:

  1. Loads ``<subject>/<subject>_preproc-raw.fif``
  2. Loads the saved ICA solution at ``<subject>/<subject>_ica.fif``
  3. Reads ``ica/<subject>/label.txt`` and collects every IC marked ``bad``
     (``unsure`` is kept --- it is "jury's out", flagged but not deleted)
  4. If ``ica/<subject>/bads.txt`` exists, appends each ``start\\tend``
     interval to ``raw.annotations`` as ``BAD_manual`` so the segments are
     ignored by downstream source recon / parcellation
  5. Applies the ICA with those bad ICs in ``ica.exclude``
  6. Writes ``<subject>/<subject>_after_ica-raw.fif`` next to the input

Run from anywhere:
    python 2.ica.py
"""
import sys
import time
from pathlib import Path

import mne

# Single source of truth for label.txt / bads.txt parsing lives in the
# osl-manual-ica package (semp re-uses it instead of maintaining a copy).
from osl_manual_ica import parse_label_txt, parse_bads_txt


# ── paths ──────────────────────────────────────────────────────────────────
PREP_PTH = Path(
    '/ohba/pi/mwoolrich/datasets/oxford/staresina/eeg_fmri/'
    'after_prep_sr_manual'
)
ICA_PTH  = PREP_PTH / 'ica'
OUT_SUFFIX = '_after_ica-raw.fif'      # alongside <subject>_preproc-raw.fif


def process_subject(subject, overwrite=False):
    raw_path   = PREP_PTH / subject / f'{subject}_preproc-raw.fif'
    ica_path   = PREP_PTH / subject / f'{subject}_ica.fif'
    label_path = ICA_PTH  / subject / 'label.txt'
    bads_path  = ICA_PTH  / subject / 'bads.txt'
    out_path   = PREP_PTH / subject / f'{subject}{OUT_SUFFIX}'

    missing = [p for p in (raw_path, ica_path, label_path) if not p.exists()]
    if missing:
        return f'skip --- missing: {[str(p.name) for p in missing]}'
    if out_path.exists() and not overwrite:
        return f'skip --- exists: {out_path.name}'

    bad, n_unsure, n_unlabeled, parse_warnings = parse_label_txt(label_path)
    if parse_warnings:
        return f'skip --- {len(parse_warnings)} bad lines in label.txt: ' \
               + ' | '.join(parse_warnings[:3]) \
               + ('...' if len(parse_warnings) > 3 else '')
    if n_unlabeled:
        return f'skip --- {n_unlabeled} unlabeled ICs (review not finished)'

    bad_segs = parse_bads_txt(bads_path) if bads_path.exists() else []

    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    if bad_segs:
        onsets    = [s for s, _ in bad_segs]
        durations = [d for _, d in bad_segs]
        raw.annotations.append(onsets, durations,
                               ['BAD_manual'] * len(bad_segs))
    ica = mne.preprocessing.read_ica(ica_path, verbose=False)
    # Bounds-check the IC indices against the actual decomposition --- a
    # typo'd label like "IC999: bad" against a 50-component fit would
    # otherwise be silently handed to MNE.
    out_of_range = [i for i in bad if i >= ica.n_components_]
    if out_of_range:
        return (f'skip --- label.txt references IC indices >= '
                f'n_components_={ica.n_components_}: {out_of_range}')
    ica.exclude = list(bad)
    cleaned = ica.apply(raw, verbose=False)         # in-place on the loaded raw
    cleaned.save(out_path, overwrite=overwrite, verbose=False)

    return (
        f'ok --- removed {len(bad)} bad / kept {ica.n_components_ - len(bad)} '
        f'(unsure: {n_unsure}, bad segs: {len(bad_segs)}) -> {out_path.name}'
    )


# ── main ───────────────────────────────────────────────────────────────────
def main(overwrite=False):
    if not ICA_PTH.exists():
        sys.exit(f'ICA folder not found: {ICA_PTH}')

    subjects = sorted(
        p.name for p in ICA_PTH.iterdir()
        if p.is_dir() and (p / 'label.txt').exists()
    )
    print(f'[2.ica] {len(subjects)} subject(s) with label.txt')

    n_done, n_skip = 0, 0
    t0 = time.time()
    for s in subjects:
        try:
            msg = process_subject(s, overwrite=overwrite)
        except Exception as e:
            msg = f'error --- {type(e).__name__}: {e}'
        print(f'  {s}: {msg}')
        if msg.startswith('ok'):
            n_done += 1
        else:
            n_skip += 1
    print(f'[2.ica] {n_done} processed, {n_skip} skipped '
          f'in {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main(overwrite='--overwrite' in sys.argv)
