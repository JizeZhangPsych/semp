import os, glob, re, copy
from natsort import natsorted as sorted
import numpy as np
import mne
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, welch

ALL_CHANNEL_LIST = {'grad', 'mag', 'eeg', 'csd', 'stim', 'eog', 'emg', 'ecg', 'ref_meg', 'resp', 'exci', 'ias', 'syst', 'misc', 'seeg', 'dbs', 'bio', 'chpi', 'dipole', 'gof', 'ecog', 'hbo', 'hbr', 'temperature', 'gsr', 'eyetrack'}

def mean_psd_in_band(psd_row, freqs, center, half_width):
    mask = (freqs >= (center - half_width)) & (freqs <= (center + half_width))
    if not mask.any():
        # fallback to nearest bin
        return psd_row[np.argmin(np.abs(freqs - center))]
    return psd_row[mask].mean()   # mean of density across bins

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
    Correct all trigger event to **one template** using Pearson correlation.
    The aim is to prevent trigger jitter that can occur due to various factors, mainly asynchronous EEG-fMRI recording.
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
    eeg = copy.deepcopy(eeg)  # avoid modifying original data (e.g., picking channels in-place)
    # matplotlib.rcParams.update(rc)
    if 'mne.io' in str(type(eeg)):  # already Raw
        if fs is None:
            fs = eeg.info['sfreq']
        # raw = pick_indices(eeg, picks, return_indices=False)
        raw = eeg.pick(picks)
    else:  # numpy or torch
        eeg = np.array(eeg)
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
                plot_out = psd.plot(dB=dB, amplitude=True, show=False, picks=picks)
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
                # MNE's psd.plot() attaches a RangeSlider that registers a draw_event
                # callback using copy_from_bbox (blitting), which is unsupported by
                # non-interactive backends like PDF. Disconnect before saving.
                for cid in list(fig.canvas.callbacks.callbacks.get('draw_event', {}).keys()):
                    fig.canvas.mpl_disconnect(cid)
                fig.savefig(save_pth.with_suffix(".pdf"), bbox_inches='tight', pad_inches=0)
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

        # 2) compute med and height (kept for backward readability, but we choose exp from absolute max)
        med = 0.5 * (raw_min + raw_max)
        height = raw_max - med   # equals (raw_max - raw_min)/2

        # 3) choose exponent so that the magnitude of the signal is between 1 and 10
        # Robustly compute absolute max from the actual data slice so NaNs do not propagate
        try:
            vmax_abs = float(np.nanmax([abs(raw_min), abs(raw_max)]))
        except Exception:
            vmax_abs = 0.0

        # final guard: if still non-positive or NaN, use exp=0 (factor=1)
        if np.isnan(vmax_abs) or vmax_abs <= 0.0:
            exp = 0
        else:
            exp = int(np.floor(np.log10(vmax_abs)))
            # adjust exp so that 1 <= vmax_abs / 10**exp < 10
            while True:
                vmax_s = vmax_abs / (10 ** exp)
                if vmax_s < 1:
                    exp -= 1
                    continue
                if vmax_s >= 10:
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
            any_plotted = True

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

        if any_plotted:
            plt.legend(loc='upper right')
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
    An easier way to realize this is to 1) use mne.Epochs.get_data() to get the data in ndarray format, 2) reshape, and then 3) use mne.io.RawArray to create a new Raw object. However, this approach is dangerous if the epoch data have overlapping time windows, which is common in BCG correction, or FASTR GA correction. In this case artifact would appear at the edge of concatenated epochs.
    Instead, this function directly overwrites the data in the raw object, which can avoid the edge artifact issue. The overwrite parameter specifies the behavior when overwriting the raw data. 'new' means the epoch with a larger index will overwrite the epoch with a smaller index, while 'even' means the datapoint closer to the event onset will be retained.
    Parameters
    ----------
    epoch : mne.Epochs
        The mne.Epochs object containing the epoched data to be converted. If ndarray is not None, the data from this object will not be used, only metadata such as events and channel names will be used.
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
        'obs' : early dirty epochs. for first 11 epochs, use new, for the rest, use even. This is following the OBS method in Niazy05. 
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
    
    epoch = epoch.pick(picks)
    picked_idx = [raw.ch_names.index(ch) for ch in epoch.ch_names]
    
    if len(raw.info['bads']) > 0:
        picked_idx = [idx for idx in picked_idx if raw.ch_names[idx] not in raw.info['bads']]
    
    # get data
    processed_data = ndarray if ndarray is not None else epoch.get_data()
    
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


def pcs_plot(pcs, target_fdr, ch_list, ch_names, info, win_list=None, figsize=(20, 9), strict=True, resolution=0.05, psd_lim=(0,50)):
    """
    Print the PCA components in a human-readable format.
    3 pcs are printed together in one image, with the first one being the mean.
    pcs: numpy array of shape (len(win)-1+#windows, #channels, len(epoch), #pc) or (#channels, len(epoch), #pc)
    target_fdr: the folder to save the images
    ch_list: the list of NAME of channels to plot
    ch_names: the list of names of ALL channels
    info: the info of the raw data
    strict: if True, assert the number of channels in pcs and ch_names should be the same. else, only a warning would be printed if pcs.shape[-3] < len(ch_names)
    resolution: the resolution for psd plot, in Hz/bin. Default is 0.05, which means 0.05Hz per bin in the psd plot. If the psd calculation fails due to too high resolution, the function will automatically reduce the resolution by half until it succeeds.
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
    n_fft = int(np.round(fs / resolution))
    if len(pcs.shape) == 3:
        for ch_name in ch_list:
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
                axs[npc, 1].set_xlim(psd_lim)
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
                    axs[npc, 1].set_xlim(psd_lim)
                    axs[npc, 1].legend()
                    
                axs[-1,0].set_xlabel("Time (s)")
                axs[-1,1].set_xlabel("Frequency (Hz)")                
                fig.tight_layout()
                fig.savefig(os.path.join(target_fdr, f"pc_{ch_name}_win_{win_idx}.png"))
                plt.close(fig)
    else:
        raise ValueError("pcs should be of shape (#ch, len(ep), #pc) or (len(win)-1+#windows, #ch, len(ep), #pc)")
    
class EEGDictMeta(type):
    def __call__(cls, *args, **kwargs):
        return cls.get(*args, **kwargs)

class SingletonEEG(metaclass=EEGDictMeta):
    """Singleton class for loading an example EEG file once and reusing it.
    This class is used to avoid loading the EEG file multiple times, which can be time-consuming.
    
    Example usage: some dataset lack a dev_head_t matrix, so we need to load it from another eeg file.
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