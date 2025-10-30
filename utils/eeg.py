import os, glob, re, parse, copy
from natsort import natsorted as sorted
import numpy as np
import mne
from string import Formatter
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, welch
from math import floor, ceil, log10

ALL_CHANNEL_LIST = {'grad', 'mag', 'eeg', 'csd', 'stim', 'eog', 'emg', 'ecg', 'ref_meg', 'resp', 'exci', 'ias', 'syst', 'misc', 'seeg', 'dbs', 'bio', 'chpi', 'dipole', 'gof', 'ecog', 'hbo', 'hbr', 'temperature', 'gsr', 'eyetrack'}

def pearson_corr(windows: np.ndarray, template: np.ndarray):
    """
    windows: shape (N, L) — N windows of length L
    template: shape (L,) — single template
    returns: shape (N,) — correlation of each window with the template
    """
    # Normalize template
    return np.array([np.abs(np.corrcoef(window, template)[0,1]) for window in windows])

def correct_trigger(raw, event, event_id, tmin, tmax, template='mid', channel=0, hwin=30):
    """
    Correct the trigger event using Pearson correlation.
    Parameters
    ----------
    raw : mne.io.Raw
        The raw data object.
    event : numpy.ndarray, shaped (n_events, 3).
        The event to correct.
    event_id : int
        The event ID to correct.
    tmin : float
        The start time of the epoch relative to the event onset, in seconds.
    tmax : float
        The end time of the epoch relative to the event onset, in seconds.
    template : str, optional
        The template to use for the correction. Can be 'mid', 'start'. Default is 'mid'.
    channel : int, optional
        The channel to use for the correction. Default is 0.
    hwin : int, optional
        The half window size for the best trigger searching. Default is 30 (timepoints).
    Returns
    -------
    corrected_events : np.ndarray
        The corrected events array. Only contains events with the specified event_id.
    """
    
    event = event[event[:, 2] == event_id, 0]
    new_events = []
    sfreq = raw.info['sfreq']
    data = raw.get_data()[channel]
    tpmin = int(tmin * sfreq)
    tpmax = int(tmax * sfreq)
    
    if template == 'mid':
        tmplt_event_tp = event[event.shape[0] // 2] - raw.first_samp
        tmplt = data[tmplt_event_tp + tpmin:tmplt_event_tp + tpmax+1]
    elif template == 'start':
        try:
            tmplt_event_tp = event[0] - raw.first_samp
            tmplt = data[tmplt_event_tp + tpmin:tmplt_event_tp + tpmax+1]
        except IndexError:
            tmplt_event_tp = event[1] - raw.first_samp
            tmplt = data[tmplt_event_tp + tpmin:tmplt_event_tp + tpmax+1]
    else:
        raise ValueError(f"Template {template} not supported. Use 'mid' or 'start'.")
    best_pos_list = []
    
    for ev_tp in event:
        ev_tp -= raw.first_samp
        win_pos_list = np.arange(ev_tp-hwin, ev_tp+hwin+1)
        win_pos_list = win_pos_list[(win_pos_list+tpmin >= 0) & (win_pos_list+tpmax+1 < len(data))]
        if len(win_pos_list) == 0:
            continue
        
        window_arr = np.stack([data[pos+tpmin:pos+tpmax+1] for pos in win_pos_list])
        corr = pearson_corr(window_arr, tmplt)
        best_pos = win_pos_list[np.argmax(corr)]  
        best_pos_list.append(best_pos-ev_tp)
        
        new_events.append([best_pos + raw.first_samp, 0, event_id])      
    
    return np.array(new_events, dtype=np.int32)
    
    



def pick_indices(mne_obj, picks, return_indices=True):
    if isinstance(picks, str):
        picks = [picks]
    if picks is None or 'all' in picks:
        return np.arange(len(mne_obj.ch_names)) if return_indices else mne_obj.copy()
    
    type_flags = {}
    include_names = []
    for pick in picks:
        if pick in ALL_CHANNEL_LIST:
            type_flags[pick] = True
        else:
            include_names.append(pick)
    
    indices = mne.pick_types(mne_obj.info, include=include_names, **type_flags)
    # return indices if return_indices else raw.copy().pick_channels([raw.ch_names[idx] for idx in indices])

    if return_indices:
        return indices
    
    ret_obj = copy.deepcopy(mne_obj)
    try:
        ret_obj = ret_obj.pick_channels([ret_obj.ch_names[idx] for idx in indices])
    except RuntimeError as e:
        if 'MNE does not load data into main memory to conserve resources' in str(e):
            ret_obj.load_data()
            ret_obj = ret_obj.pick_channels([ret_obj.ch_names[idx] for idx in indices])
        else:
            raise e
    return ret_obj



def find_spurious_channels(psd, freq_range=[1,40], slice_interval=0.07, residual_radius=1, ref_height_factor=2):
    residual_pos = []
    dirty_channels = []
    for idx in range(1, 10):
        if idx/slice_interval > freq_range[1]:
            break
        residual_pos.append(idx/slice_interval)

    psd2db = lambda psd_data: 10*np.log10(np.maximum(psd_data*1e12, np.finfo(float).tiny))
    freq2idx = lambda freq, psd_obj: np.sum(psd_obj.freqs < freq)
    for channel_idx, channel_data in enumerate(psd.get_data(picks='eeg')):
        channel_data = psd2db(channel_data)
        reference_start_idx = 0
        for spurious_freq in residual_pos:
            res_start, res_end = spurious_freq - residual_radius, spurious_freq + residual_radius
            res_start_idx, res_end_idx = freq2idx(res_start, psd), freq2idx(res_end, psd)
            reference_psd_arr = channel_data[reference_start_idx:res_start_idx]
            ref_height = np.mean(reference_psd_arr) + ref_height_factor*np.std(reference_psd_arr)
            peaks, properties = find_peaks(channel_data[res_start_idx:res_end_idx], height=ref_height)
            reference_start_idx = res_end_idx
            if len(peaks) != 0:
                dirty_channels.append(channel_idx)
                break

    return dirty_channels

class Pathfinder:
    def __init__(self, ds_name='staresina', def_fsl_dir='/opt/ohba/software/software/fsl/6.0.7.9', base_dir="/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/", src="/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/", prep="after_prep", recon="after_recon", hmm="after_hmm", raw="edfs", slice="abnormal_slices", debug="debug", **kwargs):
        """
            Generates a class containing for finding paths of EEG-fMRI processing.
            Parameters:
            def_fsl_dir (str): if $FSLDIR not exist, Default directory for FSL software. Defaults to '/opt/ohba/software/software/fsl/6.0.7.9'.
            base_dir (str): Base directory for EEG-fMRI datasets. Defaults to "../datasets/eeg-fmri_Staresina/".
            mne_fdr_name (str): Folder name for MNE processed data. Defaults to "after_mne".
            recon_fdr_name (str): Folder name for reconstructed data. Defaults to "after_recon".
        """
    
        self.fdr = {}
        self.fdr["fsl"] = os.getenv('FSLDIR')
        if self.fdr["fsl"] is None:
            self.fdr["fsl"] = def_fsl_dir

        self.fdr["base"] = base_dir
        self.fdr["src"] = src

        self.fdr["raw"] = os.path.join(self.fdr['src'], raw)

        self.fdr["slice"] = os.path.join(self.fdr['base'], slice)
        self.fdr["prep"] = os.path.join(self.fdr['base'], prep)
        self.fdr["recon"] = os.path.join(self.fdr['base'], recon)
        self.fdr["hmm"] = os.path.join(self.fdr['base'], hmm)
        self.fdr["debug"] = os.path.join(self.fdr['base'], debug)
        
        for key, value in kwargs.items():
            if key in self.fdr.keys():
                raise ValueError(f"Key {key} already exists in fdr dictionary, don't include it in kwargs. Existing keys are {self.fdr.keys()}")
            self.fdr[key] = os.path.join(self.fdr['base'], value)

        try:        
            self.subject_list = sorted(list(set([filename2subj(filename, ds_name) for filename in os.listdir(self.fdr["raw"])])))
        except :
            self.subject_list = []
    
    def get_fdr_dict(self):
        return self.fdr
    
    def get_eeg_file(self, subj_str, data_type, postfix):
        """
            Returns the path to the file given the subject string and the data type.
            Parameters:
            subj_str (str): Subject string.
            data_type (str): Data type. Must be one of the keys in Pathfinder.fdr.
        """
        assert data_type in self.fdr.keys(), f"Data type {data_type} not found in Pathfinder.fdr"
        assert data_type != "base", f"Data type cannot be 'base'"
        subject = parse_subj(subj_str)
        file_name = f'{subject["subj"]}_{subject["ses"]}_{subject["run"]}*{subject["block"]}*' + postfix
        file_pth = glob.glob(os.path.join(self.fdr[data_type], file_name))
        if len(file_pth) == 2:
            file_pth = [pth for pth in file_pth if not 'no-recon' in pth]
        assert len(file_pth) <= 1, f"Found {len(file_pth)} files for {subj_str}: {file_pth}"
        if len(file_pth) == 0:
            print(f"WARNING: {subj_str} not found. skipping...")
            return None
        file_pth = file_pth[0]
        return file_pth
    
    def get_preproc_file(self, subj_str, data_type, postfix='.fif'):
        assert data_type in self.fdr.keys(), f"Data type {data_type} not found in Pathfinder.fdr"
        assert data_type != "base", f"Data type cannot be 'base'"
        file_name = f'{subj_str}_preproc-raw{postfix}'
        file_pth = glob.glob(os.path.join(self.fdr[data_type], subj_str, file_name))
        if len(file_pth) == 0:
            return None
        assert len(file_pth) == 1, f"Found {len(file_pth)} files"
        file_pth = file_pth[0]
        return file_pth
    
    def get_fmri_file(self, subj_str):
        raise NotImplementedError("Not implemented yet")
    
    def get_polhemus_file(self, subj_str):
        raise NotImplementedError("Not implemented yet")
    
def parse_subj(subject, digit_only=False, ds_name='staresina'):
    if ds_name == 'staresina':
        subj_dict = {
            "subj": f"{subject[:-3].zfill(3)}",
            "ses": f"0{subject[-3]}",
            "run": f"0{subject[-2]}",
            "block": f"0{subject[-1]}"
        }    
        if not digit_only:
            subj_dict["subj"] = f"sub-{subj_dict['subj']}"
            subj_dict["ses"] = f"ses-{subj_dict['ses']}"
            subj_dict["run"] = f"run-{subj_dict['run']}"
            subj_dict["block"] = f"block-{subj_dict['block']}"
    elif ds_name == 'irene':
        subj_dict = {
            "subj": f"{subject[:2]}",
            "visit": f"{subject[-2]}",
            "block": f"{subject[-1]}"
        }
        if not digit_only:
            subj_dict["subj"] = f"s{subj_dict['subj']}"
            # visit is used differently in folder name and file name, so we just keep it as is
            subj_dict["block"] = f"block{subj_dict['block']}"
    return subj_dict

def filename2subj(filename, ds_name='staresina'):
    if ds_name == 'staresina':
    
        filename = filename.split('/')[-1]
        match = re.search(r'sub-(\d+).*ses-(\d+).*run-(\d+).*block-(\d+)', filename)
        if match:
            subj = match.group(1).lstrip("0")
            ses = match.group(2).lstrip("0")
            run = match.group(3).lstrip("0")
            block = match.group(4).lstrip("0")
            subject_identifier = subj + ses + run + block
            return subject_identifier
        else:
            match = re.search(r'(\d+)_preproc-raw.fif', filename)
            subject = match.group(1)
            return subject
    elif ds_name == 'lemon':
        filename = filename.split('/')[-1].split('.')[0]
        return filename
    elif ds_name == 'irene':
        filename = filename.split('/')[-1]
        match = re.search(r's(\d\d)_mrEEG(?:_visit)?(\d)_block(\d).*\.cdt', filename)
        try:
            if match:
                subj = match.group(1)
                visit = match.group(2)
                block = match.group(3)
                subject_identifier = subj + visit + block
                return subject_identifier
            else:
                match = re.search(r'(\d+)_preproc-raw.fif', filename)
                subject = match.group(1)
                return subject
        except Exception:
            print(f"Filename {filename} parse error.")

def psd_plot(eeg, name=None, fs=None, picks='eeg', fmin=0, fmax=60, resolution=0.05, figsize=(20,3), save_pth=None, debug=False, dB=False, rc={
        'font.size': 12,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    }):
    """
    Plot the power spectral density (PSD) of a single EEG dataset.
    
    Parameters
    ----------
    eeg : mne.io.Raw or ndarray-like
        EEG data. If ndarray-like, shape should be (channels, time).
    name : str, optional
        Title or filename suffix. Default: None.
    fs : float, optional
        Sampling frequency. If None, taken from Raw, or raised as an error.
    picks : str or list, optional
        Channels to include. Default: 'eeg'.
    fmin, fmax : float
        Frequency range. Default: 0–60 Hz.
    resolution : float
        Resolution for psd map. Default: 0.05 Hz/bin.
    figsize : tuple
        Matplotlib figure size. Default: (20, 3).
    save_pth : str, optional
        Base path to save figure. If None, just show.
    debug : bool
        If True, print debug info.
    dB : bool
        Whether to plot in dB scale.
    """
    verbose = 'INFO' if debug else 'ERROR'
    # matplotlib.rcParams.update(rc)
    if 'mne.io' in str(type(eeg)):  # already Raw
        if fs is None:
            fs = eeg.info['sfreq']
        raw = pick_indices(eeg, picks, return_indices=False)
    else:  # numpy or torch
        eeg = np.array(eeg).copy()
        if fs is None:
            raise ValueError("fs must be provided if eeg is not an mne.io.Raw object.")
        ch_types = picks if picks in ALL_CHANNEL_LIST else 'eeg'
        raw = mne.io.RawArray(eeg, mne.create_info(eeg.shape[0], sfreq=fs, ch_types=ch_types))
    
    n_fft = int(np.round(fs / resolution))
    
    old_level = mne.set_log_level(verbose, return_old_level=True)
    
    with matplotlib.rc_context(rc):
        while True:
            try:
                psd = raw.compute_psd(fmin=fmin, fmax=fmax, n_fft=n_fft, picks=picks)
                plot_out = psd.plot(dB=dB, show=False, picks=picks)
                # Handle both (fig, axes) and fig-only returns
                if isinstance(plot_out, tuple):
                    fig, axes = plot_out
                else:
                    fig, axes = plot_out, plot_out.axes
                break        
            except ValueError as e:
                if 'NaN' in str(e):
                    n_fft = n_fft // 2
                    print(f'WARNING: PSD calculation failed, trying again with a smaller resolution {int(fs / n_fft)} Hz/bin')
                else:
                    raise e
            except RuntimeError as e:
                if 'No plottable channel types found' in str(e):
                    fig = psd.plot(dB=dB, show=False)
                    plot_out = psd.plot(dB=dB, show=False)
                    if isinstance(plot_out, tuple):
                        fig, axes = plot_out
                    else:
                        fig, axes = plot_out, plot_out.axes
                    break
                else:
                    raise e
        # --- Suppress MNE's per-panel channel-type titles (e.g., "EEG") ---
        try:
            for ax in (axes if isinstance(axes, (list, tuple, np.ndarray)) else [axes]):
                ax.set_title('')
        except Exception:
            for ax in fig.axes:
                ax.set_title('')
        fig.set_size_inches(figsize)
        if name is not None:
            # after plotting and getting `fig` and `axes`
            axes_list = axes if isinstance(axes, (list, tuple, np.ndarray)) else [axes]

            # get per-axis positions in figure coordinates
            pos_list = [ax.get_position() for ax in axes_list]

            # combined left and right edge of the plotting area
            left = min(p.x0 for p in pos_list)
            right = max(p.x0 + p.width for p in pos_list)

            center_fig = left + 0.5 * (right - left)
            fig.suptitle(str(name), x=center_fig)   # tweak y to taste
        
        try:
            if save_pth is not None:
                fig.savefig(save_pth.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0)
                plt.close(fig)
            else:
                plt.show()
        except ValueError as e:
            print(f"WARNING: {e}. Plotting without saving to file.")
            print(f"Current psd is : {psd}")

    mne.set_log_level(old_level)
    return psd
        

def temp_plot(eeg, channel, start=0, length=None, fs=None, events=None, event_id=None, event_onset=0, name=None, save_pth=None, figsize=(20,3), ylim=None, event_name=None, rc={
        'font.size': 12,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    }, candidate_tick_counts=[7,9,11], colors = ['r', 'g', 'm', 'c', 'y', 'k', '#7f7f7f']):
    name = name if name is not None else f'{channel}'
    event_name = event_name if event_name is not None else 'event'
    if 'mne.io' in str(type(eeg)):
        if fs is None:
            fs = eeg.info['sfreq']
        data = eeg.get_data(reject_by_annotation='NaN')
        # first_samp = eeg.first_samp
        first_samp = 0
        if isinstance(channel, str):
            channel = eeg.ch_names.index(channel)
    else:
        if fs is None:
            fs = 5000
        data = eeg.copy()
        first_samp = 0
    if length is None:
        length = data.shape[1]
    
    with matplotlib.rc_context(rc):
        plt.figure(figsize=figsize)
        start = int(start)
        length = int(length)
        
        plt.plot(np.arange(start, start + length)/fs, data[channel][start:start+length])
            
        # === y-axis scaling: new requested logic ===
        seg = np.asarray(data[channel][start:start+length], dtype=float)

        # 1) determine raw min/max (respect user-supplied ylim if provided)
        if ylim is None:
            raw_min = np.nanmin(seg) if seg.size > 0 else 0.0
            raw_max = np.nanmax(seg) if seg.size > 0 else 0.0
        else:
            raw_min, raw_max = float(ylim[0]), float(ylim[1])

        # handle degenerate / zero-range signals
        if np.isclose(raw_min, raw_max):
            if np.isclose(raw_min, 0.0):
                raw_min, raw_max = -1.0, 1.0
            else:
                span = abs(raw_min) * 0.1 if abs(raw_min) > 0 else 1.0
                raw_min = raw_min - span
                raw_max = raw_max + span

        # 2) compute med and height (half-range)
        med = 0.5 * (raw_min + raw_max)
        height = raw_max - med   # equals (raw_max - raw_min)/2

        # 3) choose exponent so that 1 < height_scaled < 10 (strict)
        if height <= 0:
            exp = 0
        else:
            exp = int(floor(log10(abs(height))))
            # adjust so the scaled height is strictly < 10 and > 1
            # height_s = height / 10**exp
            # want 1 < height_s < 10  => adjust exp accordingly
            while True:
                height_s = height / (10 ** exp)
                if height_s <= 1:
                    exp -= 1
                    continue
                if height_s >= 10:
                    exp += 1
                    continue
                break

        factor = 10 ** exp
        # scaled (display) coordinates
        raw_min_s = raw_min / factor
        raw_max_s = raw_max / factor
        med_s = med / factor

        # 4) center = closest two-decimal to med_s
        center_s = round(med_s, 2)   # \d.\d\d formatting achieved via rounding to 2 decimals

        # helper to round up step to nearest 0.01 multiple
        def step_round_up(x):
            return np.ceil(x * 100.0) / 100.0

        # try candidate tick counts (7,9,11); pick the candidate that covers data and minimizes span
        candidates = []
        for n_ticks in candidate_tick_counts:
            half = (n_ticks - 1) / 2.0
            # minimal step needed so that center +/- half*step covers raw_min_s..raw_max_s
            # step_needed = max( (center - raw_min_s)/half, (raw_max_s - center)/half )
            # protect half==0 (not possible for n>=7)
            step_needed = 0.0
            left_need = (center_s - raw_min_s) / half
            right_need = (raw_max_s - center_s) / half
            step_needed = max(left_need, right_need, 0.0)

            # if step_needed is zero (rare, e.g., all equal), use minimal step 0.01
            if step_needed <= 0:
                step_needed = 0.01

            # round up step_needed to nearest 0.01 so ticks are of form \d.\d\d
            step = step_round_up(step_needed)

            # If step is too large such that ticks would hit/ exceed -10 or 10, this candidate fails
            # compute candidate limits
            tmin = center_s - half * step
            tmax = center_s + half * step

            # ensure ticks are two-decimal; center_s is already 2-decimal and step is multiple of 0.01
            # Now check that candidate limits (in scaled coords) cover the actual scaled data
            covers = (tmin <= raw_min_s + 1e-12) and (tmax >= raw_max_s - 1e-12)
            # also ensure no tick equals exactly -10 or 10 (per your prior rules — optional but safe)
            if np.any(np.isclose(np.array([tmin, tmax]), -10.0, atol=1e-9)) or np.any(np.isclose(np.array([tmin, tmax]), 10.0, atol=1e-9)):
                covers = False

            # also ensure ticks stay reasonably inside -10..10 (a little margin)
            if tmin <= -10.0 + 1e-12 or tmax >= 10.0 - 1e-12:
                covers = False

            if covers:
                # compute raw-data ylim candidate and span
                ylim_low_raw = tmin * factor
                ylim_high_raw = tmax * factor
                span_raw = ylim_high_raw - ylim_low_raw
                candidates.append({
                    'n': n_ticks,
                    'step': step,
                    'tmin': tmin,
                    'tmax': tmax,
                    'ylim': (ylim_low_raw, ylim_high_raw),
                    'span': span_raw
                })

        # choose best candidate (minimize span)
        chosen = None
        if candidates:
            candidates = sorted(candidates, key=lambda x: x['span'])
            chosen = candidates[0]

        # 5) fallback if none chosen: use n_ticks=7 and ylim = [raw_min, raw_max]
        if chosen is None:
            n_ticks = 7
            # Try to create nice 2-decimal ticks if possible; otherwise simply set ylim=[raw_min,raw_max]
            # Compute step as span/(n_ticks-1)
            if raw_max_s - raw_min_s <= 0:
                step = 0.01
            else:
                step = (raw_max_s - raw_min_s) / (n_ticks - 1)
                # round up to nearest 0.01
                step = max(0.01, step_round_up(step))
            # build ticks centered not necessary; just produce ticks that span [raw_min_s, raw_max_s]
            # build tmin as raw_min_s (rounded down to 2 decimals in index units)
            tmin = round(raw_min_s, 2)
            # ensure tmin is of form with 2 decimals; force as floor to multiple of 0.01
            tmin_idx = int(np.floor(tmin * 100.0))
            tmin = tmin_idx / 100.0
            tmax = tmin + (n_ticks - 1) * step
            ylim_low_raw = tmin * factor
            ylim_high_raw = tmax * factor
            chosen = {
                'n': n_ticks,
                'step': step,
                'tmin': tmin,
                'tmax': tmax,
                'ylim': (ylim_low_raw, ylim_high_raw),
                'span': ylim_high_raw - ylim_low_raw,
                'fallback': True
            }

        # Build ticks from chosen
        step = chosen['step']
        tmin = chosen['tmin']
        tmax = chosen['tmax']
        n_ticks = chosen['n']

        # generate indexes and displayed tick numbers (scaled coords)
        # ensure numerical stability with rounding to 2 decimals
        idxs = np.arange(0, n_ticks)
        disp_ticks = np.round(tmin + idxs * step, 2)   # two-decimal displayed ticks
        ytick_vals = disp_ticks * factor               # convert back to raw-data coords

        # Format labels: always two decimals as requested (\d.\d\d)
        ytick_labels = [f"{v:.2f}" for v in disp_ticks]

        plt.yticks(ytick_vals, ytick_labels)
        # set tight ylim according to chosen (ensure signal is within)
        plt.ylim(chosen['ylim'])

        # y label exactly as requested
        plt.ylabel(fr"Amplitude ($\times 10^{{{exp}}}$ V)")
        plt.xlabel('time (s)')
        # === end new y-axis logic ===
        plt.xlim(start / fs, (start + length) / fs)
        
                # --- event raster: draw small colored markers in rows below the trace ---
        # normalize events/event_id into parallel lists (support both single and list forms)
        if isinstance(events, (list, tuple)) and isinstance(event_id, (list, tuple)):
            ev_list = list(events)
            id_list = list(event_id)
            if isinstance(event_name, (list, tuple)) and len(event_name) == len(ev_list):
                names = list(event_name)
            else:
                names = [event_name] * len(ev_list)
        else:
            ev_list = [events]
            id_list = [event_id]
            names = [event_name]

        # visual params
        marker = '^'
        marker_size = 60
        z = 6

        # compute positions for raster rows (placed below the plotted signal)
        y0, y1 = plt.ylim()
        y_span = y1 - y0
        row_gap = 0.03 * y_span             # gap between signal bottom and first raster row
        row_step = 0.02 * y_span            # vertical spacing between rows
        max_rows = len(ev_list)
        # don't allow raster to extend >20% of plot height; compress if necessary
        if max_rows * row_step + row_gap > 0.20 * y_span:
            row_step = max(0.20 * y_span - row_gap, 0.001 * y_span) / max(1, max_rows)

        any_plotted = False
        for i, (ev_arr, eid, nm) in enumerate(zip(ev_list, id_list, names)):
            if ev_arr is None:
                continue
            try:
                evs = ev_arr.copy()
            except Exception:
                evs = np.array(ev_arr)
            xs = []
            for ev in evs:
                # support both Nx>=1 arrays (ev[0] sample, ev[2] code) and scalar event formats
                try:
                    samp = int(ev[0]) - first_samp
                    code = ev[2] if len(ev) > 2 else None
                except Exception:
                    try:
                        samp = int(ev) - first_samp
                        code = None
                    except Exception:
                        continue
                if samp > start and samp < start + length:
                    if (eid is None) or (code == eid):
                        xs.append(samp / fs + event_onset)
            if len(xs) == 0:
                continue
            # y position for this row (below plot)
            y_row = y0 - row_gap - i * row_step
            # label: use provided name (only once per series)
            label = names[i] if isinstance(names[i], str) else f'event_{i}'
            # draw markers but avoid clipping (also add a thin black edge for contrast)
            plt.scatter(xs, [y_row] * len(xs),
                        marker=marker, s=marker_size,
                        facecolor=colors[i % len(colors)],
                        edgecolors='k', linewidths=0.4,
                        zorder=z, label=label,
                        clip_on=False)

            # ensure markers are not visually cropped: pad bottom ylim based on marker size (points -> data units)
            fig = plt.gcf()
            fig_h_in = fig.get_size_inches()[1]           # figure height in inches
            marker_diam_pts = np.sqrt(marker_size)        # marker "diameter" in points (approx)
            marker_h_in = marker_diam_pts / 72.0          # convert points -> inches (1 pt = 1/72 in)
            # data-units per inch on the y-axis:
            data_per_in = (y_span) / fig_h_in
            # pad in data units; 0.6 factor because triangular marker height < full diameter
            pad_data = marker_h_in * data_per_in * 0.6
            # make sure pad is at least a small fraction of the y-span
            pad_data = max(pad_data, 0.01 * y_span)

        # after looping all rows (once), extend ylim to include lowest row + padding
        if any_plotted:
            min_row = y0 - row_gap - (max_rows - 1) * row_step
            plt.ylim(min_row - pad_data, y1)
        
        if save_pth is not None:
            plt.savefig(save_pth, bbox_inches='tight', pad_inches=0.01)
            plt.close()
        else:
            plt.show()
    

def temp_plot_diff(eeg, eeg2, channel, start=0, length=None, fs=None, events=None, event_id=None, plot_eeg=False, event_onset=0, name='BCG removal', save_pth=None, figsize=(20,3)):
    first_samp = 0
    plt.figure(figsize=figsize)
    if not isinstance(channel, str):
        channel1 = channel2 = channel
    if 'mne.io' in str(type(eeg)):
        if fs is None:
            fs = eeg.info['sfreq']
        data = eeg.get_data(reject_by_annotation='NaN')
        first_samp = eeg.first_samp
        if isinstance(channel, str):
            channel1 = eeg.ch_names.index(channel)
    else:
        if fs is None:
            fs = 5000
        data = eeg.copy()
        first_samp = 0
    if 'mne.io' in str(type(eeg2)):
        data2 = eeg2.get_data(reject_by_annotation='NaN')
        # assert eeg2.first_samp == first_samp, "eeg and eeg2 should have the same first_samp"
        if eeg2.first_samp > first_samp:
            data = data[:, eeg2.first_samp-first_samp:]
        if eeg2.first_samp < first_samp:
            data2 = data2[:, first_samp-eeg2.first_samp:]
            
        if isinstance(channel, str):
            channel2 = eeg2.ch_names.index(channel)
    else:
        data2 = eeg2.copy()
    if length is None:
        length = min(data.shape[1], data2.shape[1])
        
    start = int(start)
    length = int(length)
    if plot_eeg:
        plt.plot(np.arange(start, start + length)/fs, data[channel1][start:start+length], label="Before")
        plt.plot(np.arange(start, start + length)/fs, data2[channel2][start:start+length], label="After", color="orange")
    else:
        data_total_length = min(data.shape[1], data2.shape[1]) 
        plt.plot(np.arange(start, start + length)/fs, (data2[channel2, :data_total_length]-data[channel1, :data_total_length])[start:start+length], label="Difference")
        

    plt.xlabel('time (s)')
    plt.ylabel('Amplitude (V)')
    plt.title(f'Before and After {name}')

    event_labeled=False    
    if events is not None and event_id is not None:
        for event in events.copy():
            event[0] -= first_samp
            if event[0] > start and event[0] < start+length and event[2] == event_id:
                if not event_labeled:
                    plt.axvline((event[0])/fs+event_onset, color='r', label='event')
                    event_labeled=True
                else:
                    plt.axvline((event[0])/fs+event_onset, color='r')
    plt.legend()
    if save_pth is not None:
        plt.savefig(save_pth)
        plt.close()
    else:
        plt.show()
    
def mne_epoch2raw(epoch, raw, ndarray=None, tmin=0, overwrite='new', picks='eeg'):

    """ Convert an mne.Epochs object to an mne.RawArray object by overwriting the raw data.
    Parameters
    ----------
    epoch : mne.Epochs
        The mne.Epochs object containing the epoched data to be converted. If ndarray is not None, the data from this object will not be used, and this object would only work as a template.
    raw : mne.io.Raw
        The mne.Raw object to be overwritten with the epoched data.
    ndarray : numpy.ndarray, optional
        A numpy array containing the data to be used for conversion. If None, the data from the epoch object will be used.
    tmin : float, optional
        The start time of the epoch relative to the event onset, in seconds. Default is 0.
    overwrite : str, optional
        Specifies the behavior when overwriting the raw data. Default is 'new'.
        'new' : The epoch with a larger index will overwrite the epoch with a smaller index.
        'even' : The datapoint closer to the event onset will be retained.
        'obs' : early dirty epochs. for first 11 epochs, use new, for the rest, use even.
    picks : str, optional
        The channels to include in the conversion. Default is 'eeg'. Would raise an AssertionError if the number of channels in the ndarray object does not match the number of channels in "picks" in the raw object.
    Returns
    -------
    raw : mne.io.Raw
        The mne.Raw object with the epoched data overwritten.
    Raises
    ------
    AssertionError
        If the number of channels in the ndarray object does not match the number of channels in "picks" in the raw object.
    """
    
    epoch = pick_indices(epoch, picks, return_indices=False)
    picked_idx = [raw.ch_names.index(ch) for ch in epoch.ch_names]
    
    if len(raw.info['bads']) > 0:
        picked_idx = [idx for idx in picked_idx if raw.ch_names[idx] not in raw.info['bads']]
    
    # get data
    if ndarray is not None:
        processed_data = ndarray
    else:
        processed_data = epoch.get_data()
    
    # check if the number of channels in the ndarray object matches the number of channels in "picks" in the raw object
    assert processed_data.shape[1] == len(picked_idx), f"Number of channels in ndarray ({processed_data.shape[1]}) does not match number of channels in raw object ({len(picked_idx)})"
        
    sfreq = raw.info['sfreq']
    tmin_shift = sfreq*tmin                
    
    old_mid = -100000
    for i, event_onset in enumerate(epoch.events[:,0]):
        start_sample = int(event_onset+tmin_shift) - raw.first_samp
        end_sample = start_sample+processed_data.shape[2]
        epoch_data = processed_data[i]
        if overwrite == 'even' or (overwrite == 'obs' and i > 10):
            mid_sample = start_sample + processed_data.shape[2]//2 + 1
            epoch_divide = (mid_sample + old_mid) // 2 + 1
            if epoch_divide > start_sample:
                epoch_data = epoch_data[:, epoch_divide-start_sample:]
                start_sample = epoch_divide
            old_mid = mid_sample
        
        if end_sample > raw._data.shape[1] or start_sample < 0:
            raise ValueError(f"epoch {i} exceeds raw data bound {raw._data.shape[1]}. ({start_sample}~{end_sample})")

        raw._data[picked_idx, start_sample:end_sample] = epoch_data

    return raw


def pcs_plot(pcs, target_fdr, ch_list, ch_names, info, win_list=None, figsize=(20, 9), strict=True):
    """
    Print the PCA components in a human-readable format.
    3 pcs are printed together in one image, with the first one being the mean.
    pcs: numpy array of shape (len(win)-1+#windows, #channels, len(epoch), #pc) or (#channels, len(epoch), #pc)
    target_fdr: the folder to save the images
    ch_list: the list of NAME of channels to plot
    ch_names: the list of names of ALL channels
    info: the info of the raw data
    strict: if True, assert the number of channels in pcs and ch_names should be the same. else, only a warning would be printed if pcs.shape[-3] < len(ch_names)
    """
    
    fs = info['sfreq']
    bad_chs = info['bads']
    ch_list = [ch for ch in ch_list if (ch not in bad_chs) and (ch in ch_names)]
    ch_names = [ch for ch in ch_names if ch not in bad_chs]
    
    if len(ch_names) != pcs.shape[-3]:
        if strict:
            raise ValueError(f"Number of channels in pcs ({pcs.shape[-3]}) does not match number of channels in ch_names ({len(ch_names)})")
        elif len(ch_names) > pcs.shape[-3]:
            print(f"WARNING: number of ch_names is longer then pcs.shape. the last {len(ch_names)-pcs.shape[-3]} channels would be ignored.")
            ch_names = ch_names[:pcs.shape[-3]]
        else:
            raise ValueError(f"Number of channels in pcs ({pcs.shape[-3]}) is larger than number of channels in ch_names ({len(ch_names)})")
    
    x = np.arange(pcs.shape[-2]) / fs
    n_fft = np.power(2, np.round(np.log2(fs))).astype(np.int64)   # signal length is around 1s
    # psd2db = lambda psd_data: 10*np.log10(np.maximum(psd_data*1e12, np.finfo(float).tiny))
    if len(pcs.shape) == 3:
        for ch_name in ch_list:
            # ch_name = ch_names[ch]
            ch_idx = ch_names.index(ch_name)

            fig, axs = plt.subplots(pcs.shape[-1], 2, figsize=figsize, squeeze=False)
            for npc in range(pcs.shape[-1]):
                axs[npc,0].plot(x, pcs[ch_idx, :, npc], label=f"PC{npc}")
                axs[npc,0].set_title(f"PC{npc} for channel {ch_name}")
                axs[npc, 0].set_xlim([x[0], x[-1]])
                axs[npc,0].legend()
                
                freqs, psd = welch(pcs[ch_idx, :, npc], fs, nperseg=min(n_fft,pcs.shape[-2]))
                axs[npc,1].plot(freqs, psd, label="PSD")
                axs[npc,1].set_title(f"PC{npc} PSD for channel {ch_name}")
                axs[npc, 1].set_xlim([0, 40])
                axs[npc,1].legend()
                
            axs[-1,0].set_xlabel("Time (s)")
            axs[-1,1].set_xlabel("Frequency (Hz)")
            fig.tight_layout()
            fig.savefig(os.path.join(target_fdr, f"pc_{ch_name}.png"))
            plt.close(fig)
    elif len(pcs.shape) == 4:
        for ch_name in ch_list:
            ch_idx = ch_names.index(ch_name)
            for win_idx in (win_list if win_list is not None else np.random.choice(np.arange(pcs.shape[0]), 1)):
                fig, axs = plt.subplots(pcs.shape[-1], 2, figsize=figsize, squeeze=False)
                for npc in range(pcs.shape[-1]):
                    axs[npc, 0].plot(x, pcs[win_idx, ch_idx, :, npc], label=f"PC{npc}")
                    axs[npc, 0].set_title(f"PC{npc} for No. {win_idx} window of channel {ch_name}")
                    axs[npc, 0].set_xlim([x[0], x[-1]])
                    axs[npc, 0].legend()
                    
                    freqs, psd = welch(pcs[win_idx, ch_idx, :, npc], fs, nperseg=min(n_fft, pcs.shape[-2]))
                    axs[npc,1].plot(freqs, psd , label="PSD")
                    axs[npc, 1].set_title(f"PC{npc} PSD for No. {win_idx} window of channel {ch_name}")
                    axs[npc, 1].set_xlim([0, 40])
                    axs[npc, 1].set_ylim(bottom=-20)
                    axs[npc, 1].legend()
                    
                axs[-1,0].set_xlabel("Time (s)")
                axs[-1,1].set_xlabel("Frequency (Hz)")                
                fig.tight_layout()
                fig.savefig(os.path.join(target_fdr, f"pc_{ch_name}_win_{win_idx}.png"))
                plt.close(fig)
    else:
        raise ValueError("pcs should be of shape (#ch, len(ep), #pc) or (len(win)-1+#windows, #ch, len(ep), #pc)")
    
    
class Study:
    """Class for simple file finding and looping.
    
    Parameters
    ----------
    studydir : str
        The study directory with wildcards.
    
    Attributes
    ----------
    studydir : str
        The study directory with wildcards.
    fieldnames : list
        The wildcards in the study directory, i.e., the field names in between {braces}.
    globdir : str
        The study directory with wildcards replaced with *.
    match_files : list
        The files that match the globdir.
    match_values : list
        The values of the field names (i.e., wildcards) for each file.
    fields : dict
        The field names and values for each file.
    
    Notes
    -----
    This class is a simple wrapper around glob and parse. It works something like this:
    
    >>> studydir = '/path/to/study/{subject}/{session}/{subject}_{task}.fif'
    >>> study = Study(studydir)
    
    Get all files in the study directory:
    
    >>> study.get()
    
    Get all files for a particular subject:
    
    >>> study.get(subject='sub-01')
    
    Get all files for a particular subject and session:
    
    >>> study.get(subject='sub-01', session='ses-01')
    
    The fieldnames that are not specified in ``get`` are replaced with wildcards (``*``).
    """
    
    def __init__(self, studydir):
        """
        Notes
        -----
        This class is a simple wrapper around glob and parse. It works something like this:
        
        >>> studydir = '/path/to/study/{subject}/{session}/{subject}_{task}.fif'
        >>> study = Study(studydir)
        
        Get all files in the study directory:
        
        >>> study.get()
        
        Get all files for a particular subject:
        
        >>> study.get(subject='sub-01')
        
        Get all files for a particular subject and session:
        
        >>> study.get(subject='sub-01', session='ses-01')
        
        The fieldnames that are not specified in ``get`` are replaced with wildcards (*).
        """
        self.studydir = studydir

        # Extract field names in between {braces}
        self.fieldnames = [fname for _, fname, _, _ in Formatter().parse(self.studydir) if fname]

        # Replace braces with wildcards
        self.globdir = re.sub("\{.*?\}","*", studydir)

        self.match_files = sorted(glob.glob(self.globdir))
        print('found {} files'.format(len(self.match_files)))

        self.match_files = [ff for ff in self.match_files if parse.parse(self.studydir, ff) is not None]
        print('keeping {} consistent files'.format(len(self.match_files)))

        self.match_values = []
        for fname in self.match_files:
            self.match_values.append(parse.parse(self.studydir, fname).named)

        self.fields = {}
        # Use first file as a reference for keywords
        for key, value in self.match_values[0].items():
            self.fields[key] = [value]
            for d in self.match_values[1:]:
                self.fields[key].append(d[key])

    
    def refresh(self):
        """Refresh the study directory."""
        return self.__init__(self.studydir)
    
    
    def get(self, check_exist=True, **kwargs):
        """Get files from the study directory that match the fieldnames.

        Parameters
        ----------
        check_exist : bool
            Whether to check if the files exist.
        **kwargs : dict
            The field names and values to match.

        Returns
        -------
        out : list
            The files that match the field names and values.

        Notes
        -----
        Example using ``Study`` and ``Study.get()``:
        
        >>> studydir = '/path/to/study/{subject}/{session}/{subject}_{task}.fif'
        >>> study = Study(studydir)
        
        Get all files in the study directory:
        
        >>> study.get()
        
        Get all files for a particular subject:
        
        >>> study.get(subject='sub-01')
        
        Get all files for a particular subject and session:
        
        >>> study.get(subject='sub-01', session='ses-01')
        
        The fieldnames that are not specified in ``get`` are replaced with wildcards (``*``).               
        """
        keywords = {}
        for key in self.fieldnames:
            keywords[key] = kwargs.get(key, '*')

        fname = self.studydir.format(**keywords)
        
        # we only want the valid files
        if check_exist:
            return [ff for ff in glob.glob(fname) if any(ff in ff_valid for ff_valid in self.match_files)]
        else:
            return glob.glob(fname)


class DebugStudy(Study):
    """ If Study find inconsistent file, print the warning out
    """
    def __init__(self, studydir):
        self.studydir = studydir
        self.fieldnames = [fname for _, fname, _, _ in Formatter().parse(self.studydir) if fname]
        self.globdir = re.sub("\{.*?\}","*", studydir)
        self.match_files = sorted(glob.glob(self.globdir))
        print('found {} files'.format(len(self.match_files)))
        tmp_lst = []
        for ff in self.match_files:
            if parse.parse(self.studydir, ff) is None:
                print('Warning: {} does not match the studydir'.format(ff)) # the only changed part: adding a warning
            tmp_lst.append(ff)
        self.match_files = tmp_lst
        print('keeping {} consistent files'.format(len(self.match_files)))
        self.match_values = []
        for fname in self.match_files:
            self.match_values.append(parse.parse(self.studydir, fname).named)
        self.fields = {}
        for key, value in self.match_values[0].items():
            self.fields[key] = [value]
            for d in self.match_values[1:]:
                self.fields[key].append(d[key])

class HeteroStudy:
    """Improved version of osl_ephys.utils.Study that supports heterogeneous file structures.
    """
    def __init__(self, studydir_list):
        """Initialize the HeteroStudy object.
        Parameters:
        studydir_list (list or str): List of study directories. If str, it will be converted to a list.
        """
        
        if isinstance(studydir_list, str):
            studydir_list = [studydir_list]
        
        self.studydir_list = studydir_list
        self.study_list = [DebugStudy(studydir) for studydir in studydir_list]
        self.fields = {}
        for study in self.study_list:
            self.fields.update(study.fields)
    
    def refresh(self):
        """Refresh the study list.
        """
        self.study_list = [DebugStudy(studydir) for studydir in self.studydir_list]
    
    def get(self, check_exist=True, **kwargs):
        """Get files from the study list that match the fieldnames.
        Parameters:
        check_exist (bool): Whether to check if the files exist.
        **kwargs (dict): The field names and values to match.
        Returns:
        out (list): The files that match the field names and values.
        """
        out = []
        for study in self.study_list:
            out += study.get(check_exist=check_exist, **kwargs)
        return out
    

class EEGDictMeta(type):
    def __call__(cls, *args, **kwargs):
        return cls.get(*args, **kwargs)

class SingletonEEG(metaclass=EEGDictMeta):
    """Singleton class for loading an example EEG file once and reusing it.
    This class is used to avoid loading the EEG file multiple times, which can be time-consuming.
    
    Example usage: Irene dataset lack a dev_head_t matrix, so we need to load it from another eeg file.
    """
    
    _raw = None
    _loaded_pth = None
    
    @classmethod
    def get(cls, file_path=None, safe_reload=True, preload=False):
        """Get the singleton instance of an example eeg file.
        
        Parameters:
        file_path (str): Path to the EEG file to load.
        keys (list): List of keys to extract from the EEG file. any key should be a property extractable via raw.key, e.g. ['info', 'first_samp', 'ch_names', '_data', 'annotations'].
        info_keys (list): List of keys to extract from the EEG info.
        safe_reload (bool): If True, when loading this dict again, it will check if the file path and keys are either none or the same as the one loaded before. If not, would raise an error.
        
        Returns:
        ExampleEEGSingletonLoader: The singleton instance of a dictionary, containing all keys from the eeg raw class.
        """
        if cls._raw is None:
            if file_path.endswith('.edf'):
                eeg = mne.io.read_raw_edf(file_path, preload=preload)
            else:
                raise ValueError("Unsupported file format. Only .edf files are supported.")
            
            cls._raw = eeg
            cls._loaded_pth = file_path
            del eeg
        elif safe_reload and file_path is not None:
            if file_path != cls._loaded_pth:
                raise ValueError(f"File path {file_path} does not match the previously loaded file path {cls._loaded_pth}. Use the same file path or None to reload the singleton.")
            
        return cls._raw