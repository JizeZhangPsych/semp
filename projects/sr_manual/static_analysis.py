"""Static (whole-recording) network analysis for the manually-cleaned
Staresina resting EEG-fMRI dataset.

Replaces the legacy ``semp/projects/sr/static_analysis.py``:

  * pulls source-recon files via the sr_manual pathfinder (after_src_sr_manual)
  * uses ``semp.visualize`` (StaticVisualizer / _colormap_transparent) instead
    of the now-deleted ``semp_old`` tree
  * implements ``compute_aec`` inline (bandpass -> Hilbert envelope ->
    pairwise Pearson correlation), since osl_dynamics has no built-in AEC and
    the previous helper lived in semp_old
  * caches the per-band, per-subject envelope-correlation matrices on disk so
    re-running visualisation is cheap

Run from anywhere:
    /ohba/pi/mwoolrich/jzhang/conda/envs/general/bin/python static_analysis.py

Use the ``general`` env: it has nilearn 0.10.4 + a working osl_dynamics.
The ``semp_stable`` env can't import ``osl_dynamics.analysis.connectivity``
(missing pqdm) and will crash at import time below.
"""
#%%
import os
import sys
from pathlib import Path

import numpy as np
from scipy.signal import hilbert
from mne.io import read_raw
from mne.filter import filter_data

from osl_dynamics.analysis import static, power, connectivity

from semp.utils import ensure_dir
from semp.utils.io import load_pkl, save_pkl
from semp.visualize.visualize import StaticVisualizer, _colormap_transparent

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathfinder import StaresinaRestPathfinder                         # noqa: E402


#%%
# config -------------------------------------------------------------------
TMP_DIR = Path(
    '/ohba/pi/mwoolrich/jzhang/staresina_proc/sr_manual_static/'
)
ensure_dir(TMP_DIR)
sfreq = 250
src_key = 'src'

FREQ_BANDS = {
    'wide':  (1.5, 20),
    'delta': (1.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 20),
}

# IDs to skip --- inherited from the legacy sr static_analysis. Audit / trim
# as the manual-ICA pipeline matures.
exclude_ids = {
    '8111', '8112', '8121',
    '15111', '15112',
    '17111', '17112',
    '26111', '26112',
    '31111', '31112', '31121', '31211', '31212',
}


# AEC ---------------------------------------------------------------------
def compute_aec(input_data, sfreq, freq_range, tmp_dir=None):
    """Amplitude envelope correlation per subject.

    Parameters
    ----------
    input_data : list of (n_samples, n_parcels) arrays
    sfreq : float
    freq_range : (lo, hi) bandpass in Hz
    tmp_dir : Path | None
        If given, cache the resulting (n_subj, n_parcels, n_parcels) array
        at ``tmp_dir/aec_<lo>-<hi>Hz_n<N>.npy`` so repeat calls are free.

    Returns
    -------
    conn_maps : (n_subj, n_parcels, n_parcels) float32 array
    """
    n_subj = len(input_data)
    cache_path = None
    if tmp_dir is not None:
        ensure_dir(tmp_dir)
        cache_path = (Path(tmp_dir)
                      / f'aec_{freq_range[0]}-{freq_range[1]}Hz_n{n_subj}.npy')
        if cache_path.exists():
            return np.load(cache_path)

    out = []
    lo, hi = float(freq_range[0]), float(freq_range[1])
    for i, x in enumerate(input_data):
        # x: (n_samples, n_parcels) -> filter expects (n_chans, n_samples)
        filt = filter_data(
            x.T.astype(np.float64), sfreq=sfreq, l_freq=lo, h_freq=hi,
            method='iir', iir_params={'order': 5, 'ftype': 'butter'},
            verbose=False,
        )
        env = np.abs(hilbert(filt, axis=-1))           # (n_parcels, n_samples)
        out.append(np.corrcoef(env).astype(np.float32))
        if (i + 1) % 10 == 0:
            print(f'    AEC [{lo}-{hi} Hz]: {i + 1}/{n_subj}')

    conn = np.stack(out, axis=0)
    if cache_path is not None:
        np.save(cache_path, conn)
    return conn


#%%
# [1] load source-space data per subject -----------------------------------
pf = StaresinaRestPathfinder()
print(f'pathfinder: {len(pf)} file_ids total')

input_data, ok_ids = [], []
for file_id, pth_dict in pf.items():
    if file_id in exclude_ids:
        print(f'  excluding {file_id} (in exclude_ids)')
        continue
    src_path = pth_dict.get(src_key)
    if src_path is None or not Path(src_path).exists():
        print(f'  WARNING: {file_id} has no {src_key!r} file --- skipping')
        continue
    raw = read_raw(src_path, preload=True, verbose=False)
    input_data.append(raw.get_data().T)               # (n_samples, n_parcels)
    ok_ids.append(file_id)

print(f'loaded {len(input_data)} subjects')
if not input_data:
    sys.exit('no subjects loaded --- nothing to do')


#%%
# [2] compute (or load) static features ------------------------------------
save_path = TMP_DIR / f'static_features_n{len(ok_ids)}.pkl'

if save_path.exists():
    print(f'(Step 2-1) Loading cached static features from {save_path.name}')
    feats       = load_pkl(save_path)
    freqs       = feats['freqs']
    psds        = feats['psds']
    weights     = feats['weights']
    power_maps  = feats['power_maps']
    conn_maps   = feats['conn_maps']
else:
    print('(Step 2-1) Computing static PSDs (Welch)')
    freqs, psds, weights = static.welch_spectra(
        data=input_data,
        sampling_frequency=sfreq,
        window_length=int(sfreq * 2),
        step_size=int(sfreq),
        frequency_range=[1.5, 45],
        return_weights=True,
        standardize=True,
    )                                                  # (subj, parcels, freqs)

    print('(Step 2-2) Computing static power maps')
    power_maps = {
        band: power.variance_from_spectra(freqs, psds, frequency_range=list(fr))
        for band, fr in FREQ_BANDS.items()
    }                                                  # each: (subj, parcels)

    print('(Step 2-3) Computing static AEC maps (wide band)')
    conn_maps = compute_aec(
        input_data, sfreq, freq_range=FREQ_BANDS['wide'],
        tmp_dir=TMP_DIR / 'aec_cache',
    )                                                  # (subj, parcels, parcels)

    print(f'(Step 2-4) Saving features to {save_path.name}')
    save_pkl({
        'freqs':       freqs,
        'psds':        psds,
        'weights':     weights,
        'power_maps':  power_maps,
        'conn_maps':   conn_maps,
        'subject_ids': ok_ids,
    }, save_path)


#%%
# [3] visualisation --------------------------------------------------------
print('\n*** STEP 3: VISUALIZATION ***')

SV = StaticVisualizer()
cmap_hot_tp = _colormap_transparent('gist_heat')

for bandname, freq_range in FREQ_BANDS.items():
    print(f'  band={bandname}  range={freq_range} Hz')
    TGT_DIR = TMP_DIR / bandname
    ensure_dir(TGT_DIR)

    band_conn = compute_aec(
        input_data, sfreq, freq_range=freq_range,
        tmp_dir=TMP_DIR / 'aec_cache',
    )

    # power map (subject-averaged)
    gpower_all = np.mean(power_maps[bandname], axis=0)   # (parcels,)
    gp_range   = float(np.abs(gpower_all.max() - gpower_all.min()))
    SV.plot_power_map(
        power_map=gpower_all,
        filename=str(TGT_DIR / 'power_map.png'),
        plot_kwargs={
            'vmin': 0,
            'vmax': float(gpower_all.max() + 0.1 * gp_range),
            'symmetric_cbar': False, 'cmap': cmap_hot_tp,
        },
    )

    # connectivity map (subject-averaged, top-5%)
    gconn_all = np.mean(band_conn, axis=0)
    gconn_all = connectivity.threshold(gconn_all, percentile=95)
    SV.plot_aec_conn_map(
        connectivity_map=gconn_all,
        filename=str(TGT_DIR / 'conn_map.png'),
        colormap='Reds',
        plot_kwargs={'edge_vmin': 0, 'edge_vmax': float(np.max(gconn_all))},
    )

# group PSD
gpsd_all = np.mean(psds, axis=(1, 0))                   # (freqs,)
gpsd_sem = np.std(np.mean(psds, axis=1), axis=0) / np.sqrt(len(psds))
SV.plot_psd(
    freqs=freqs, psd=gpsd_all, error=gpsd_sem,
    filename=str(TMP_DIR / 'psd.png'),
)

print('Done.')
