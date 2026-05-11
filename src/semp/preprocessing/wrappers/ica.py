"""ICA wrappers for the semp preprocessing pipeline.

The interactive ``manual_ica`` review wrapper (and its HTML templates) was
extracted into the standalone ``osl-manual-ica`` package so it can be used
by osl-ephys users who don't depend on semp. We re-export it here so
existing semp configs (``{'manual_ica': {...}}``) keep working unchanged.

The two wrappers that remain inline are semp-specific:

* ``slice_ica``   --- ICA on slice-timing harmonics (needs the
                       ``slice_interval`` / ``tr_interval`` keys that
                       semp's ``initialize`` extra_func adds to ``dataset``).
* ``apply_ica``   --- applies a saved ICA solution at the semp-conventional
                       path ``dataset['target_pth']/<subject>/<subject>_ica.fif``.
"""
import copy

import numpy as np
import mne
from mne.preprocessing import ICA
from osl_ephys.utils.logger import log_or_print

from osl_manual_ica import manual_ica   # re-export; keeps semp configs working

from semp.utils import proc_userargs, mean_psd_in_band

__all__ = ['slice_ica', 'manual_ica', 'apply_ica']


def slice_ica(dataset, userargs):
    """Perform ICA on raw data to remove residual slice artifacts.

    Fits ICA on a high-pass filtered copy of raw, identifies components with
    high power at slice-timing harmonics (CTPS-style SNR threshold), and
    removes them from the raw data.
    """
    seed = userargs.get('seed', 42)
    max_iter = userargs.get('max_iter', 'auto')
    n_components = userargs.get('n_components', .999)
    epoch_frange = userargs.get('epoch_frange', [1, 40])
    noise2base_threshold = userargs.get('noise2base_threshold', 4.0)
    noise_window = userargs.get('noise_window', 1.0)
    base_window = userargs.get('base_window', 5.0)

    slice_freq = 1 / dataset['slice_interval']
    tr_freq    = 1 / dataset['tr_interval']

    if 'slice_ica_n2b_threshold' in dataset:
        noise2base_threshold = dataset['slice_ica_n2b_threshold']
        log_or_print(
            f'using noise2base_threshold: {noise2base_threshold} '
            f'defined in initialize()'
        )

    assert base_window > noise_window, \
        'base_window should be greater than noise_window.'

    ica = ICA(n_components=n_components, max_iter=max_iter, random_state=seed)
    ica.fit(copy.deepcopy(dataset['raw']).filter(l_freq=1, h_freq=None),
            picks='eeg')

    data = ica.get_sources(dataset['raw'])._data
    psds, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=dataset['raw'].info['sfreq'],
        fmin=epoch_frange[0],
        fmax=epoch_frange[1],
        n_fft=int(round(dataset['raw'].info['sfreq'] * 20)),
    )

    exclude_list = []
    eps = 1e-10
    harmonics = np.arange(slice_freq, freqs.max(), slice_freq)
    for ic in range(data.shape[0]):
        psd_row = psds[ic]
        for harmonic in harmonics:
            noise = mean_psd_in_band(psd_row, freqs, harmonic, noise_window * tr_freq / 2)
            base  = mean_psd_in_band(psd_row, freqs, harmonic, base_window  * tr_freq / 2)
            base = (base * base_window - noise * noise_window) / (base_window - noise_window)
            if (noise / (base + eps)) > noise2base_threshold:
                exclude_list.append(ic)
                break

    ica.exclude = exclude_list
    dataset['raw'] = ica.apply(dataset['raw'].copy())
    return dataset


def apply_ica(dataset, userargs):
    """Apply a saved ICA solution, removing user-specified bad components."""
    default_args = {
        'bad_ics': [],
        'load_from_disk': False,
    }
    userargs = proc_userargs(userargs, default_args)

    subject  = dataset['subject']
    ica_path = dataset['target_pth'] / subject / f'{subject}_ica.fif'

    if userargs['load_from_disk'] or 'ica' not in dataset:
        log_or_print(f'[apply_ica] Loading ICA from {ica_path}')
        ica = mne.preprocessing.read_ica(str(ica_path))
        dataset['ica'] = ica
    else:
        ica = dataset['ica']

    bad_ics = list(userargs['bad_ics'])
    ica.exclude = bad_ics
    log_or_print(f'[apply_ica] Excluding ICs {bad_ics} and applying to raw.')

    dataset['raw'] = ica.apply(dataset['raw'].copy())
    ica.save(str(ica_path), overwrite=True)
    return dataset
