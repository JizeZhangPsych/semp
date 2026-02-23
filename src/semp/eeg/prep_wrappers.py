import os, re, copy
from datetime import datetime
import numpy as np
from scipy.stats import kurtosis
from scipy.io import loadmat
import torch
from functools import partial
# from scipy.linalg import lstsq as scipy_lstsq
from torch.linalg import lstsq

import mne
from osl_ephys.report.preproc_report import plot_channel_dists # usage: plot_channel_dists(raw, savebase)
from osl_ephys.utils.logger import log_or_print
from semp.utils import ensure_dir, proc_userargs, mean_psd_in_band
from .metric import EEGTracer, psd_band_ratio, psd_band_stat
from .helpers import psd_plot, temp_plot, temp_plot_diff, mne_epoch2raw, pcs_plot, SingletonEEG, correct_trigger
from mne.preprocessing import ICA


def voltage_correction(dataset, userargs):
    """Corrects scaling if EOG/ECG/EMG channels are stored in µV but marked as volts."""
    
    ratio_threshold = userargs.get('ratio_threshold', 1000)  # Default threshold for scaling down EOG/ECG channels
    picks = userargs.get('picks', ['eog', 'ecg', 'emg'])  # Channels to check for scaling
    
    # Get EEG stats
    eeg_data = dataset['raw'].get_data(picks='eeg', units='V')
    eeg_rms = np.sqrt(np.mean(eeg_data**2, axis=1))
    ref_rms = np.median(eeg_rms)
    ref_rms = ref_rms

    for ch_type in picks:
        pick_idx = mne.pick_types(dataset['raw'].info, eeg=False,
                        eog=(ch_type=='eog'),
                        ecg=(ch_type=='ecg'),
                        emg=(ch_type=='emg'),)
        if len(pick_idx) == 0:
            continue
        
        data = dataset['raw'].get_data(picks=ch_type, units='V')
        
        ch_rms = np.mean(np.sqrt(np.mean(data**2, axis=1)))
        
        if (ch_rms / ref_rms) > ratio_threshold:
            print(f"Warning: {ch_type.upper()} channels have a high RMS value ({ch_rms*1e6:.2f} µV) compared to EEG RMS ({ref_rms*1e6:.2f} µV). Scaling down by 1e6.")
            dataset['raw']._data[pick_idx] /= 1e6  # Scale down from µV to V

    return dataset

def cleanup(dataset, userargs):
    keywords = userargs.get('keywords', ['_noise_'])
    epoch_unload = userargs.get('epoch_unload', True)
    
    pop_keys = []
    for k in dataset.keys():
        if epoch_unload and '_ep' in k:
            if isinstance(dataset[k], mne.Epochs):
                dataset[k].preload = False
                dataset[k]._data = None
            
        for keyword in keywords:
            if keyword in k:
                pop_keys.append(k)
                break
    for k in pop_keys:
        dataset.pop(k)
    return dataset

def mid_crop(dataset, userargs):
    """Crops the raw data to the middle of the recording."""
    length = userargs.get('length', None)  # Length of the crop in seconds    
    edge = userargs.get('edge', None)  # Edge to leave out from both sides in seconds
    
    if length is None and edge is not None:
        tmin = dataset['raw'].times[0] + edge
        tmax = dataset['raw'].times[-1] - edge
    elif length is not None and edge is None:
        tmin = dataset['raw'].times[0]
        tmax = dataset['raw'].times[-1]
        
        if length > (tmax - tmin):
            raise ValueError(f"Length {length} seconds is longer than the recording duration {tmax - tmin} seconds.")
        
        mid = (tmin + tmax) / 2
        tmin = mid - length / 2
        tmax = mid + length / 2
    else:
        raise ValueError("Please provide either 'length' or 'edge', not both.")
        
    dataset['raw'].crop(tmin=tmin, tmax=tmax)
    return dataset

def init_tracer(dataset, userargs):
    """Initializes the EEGTracer with specific metrics for the dataset."""
    
    tracer_kwargs = {        
        "psd_mean": partial(psd_band_stat, band=[1, 40], fn=np.mean),
        "psd_kurtosis": partial(psd_band_stat, band=[1, 40], fn=kurtosis),
        "psd_maxmed_ratio": partial(psd_band_ratio, band1=[1, 40], fn1=np.max, band2=[1, 40], fn2=np.median),
        "psd_alpha_mean": partial(psd_band_stat, band='alpha', fn=np.mean),
        "psd_alpha_kurtosis": partial(psd_band_stat, band='alpha', fn=kurtosis),
        "psd_alpha_maxmed_ratio": partial(psd_band_ratio, band1='alpha', fn1=np.max, band2='alpha', fn2=np.median),
        "psd_beta_mean": partial(psd_band_stat, band='beta', fn=np.mean),
        "psd_beta_kurtosis": partial(psd_band_stat, band='beta', fn=kurtosis),
        "psd_beta_maxmed_ratio": partial(psd_band_ratio, band1='beta', fn1=np.max, band2='beta', fn2=np.median),
    }
    
    tracer_kwargs.update(dataset.get('tracer', {}))  # Update with any pre-set tracer metrics from dataset
    tracer_kwargs.update(userargs)
    
    dataset['tracer'] = EEGTracer(**tracer_kwargs)
    
    return dataset

def summary(dataset, userargs):
    """Generates a summary of the dataset, including basic statistics and channel information.
    Currently only plot the tracer checkpoints."""
    
    subject = dataset['subject']
    if 'tracer' in dataset:
        dataset['tracer'].plot(save_pth=dataset['target_pth'] / "ckpt" / subject, show=False)
        
    # dataset.pop('tracer', None)  # Remove tracer from dataset after plotting
    return dataset
    
def ckpt_report(dataset, userargs):
    """a function for debugging the preprocessing steps.
        strictly requires Python >=3.7, for dict keys ordering

    Args:
        dataset (dict): the dict containing raw data and metadata
        userargs (dict): a dictionary containing the optional arguments

    Returns:
        dataset: the updated dataset with the extra metadata
    """
    default_args = {
        'ckpt_name': datetime.now().strftime("%H:%M:%S"),
        'resolution': 0.05,
        'max_freq': 50,
        'key_to_print': None,
        'always_print': ['EKG'],    # must be name, 'eeg' is not allowed
        'std_channel_pick': 'eeg',
        'print_pcs': True,
        'print_noise': True,
        'dB': False,  # whether to plot psd in dB scale
        'focus_range': [100, 110],  # in seconds, for temp_plot
        'log_tracer': True,
        'psd_figsize': (10, 3),
    }
    userargs = proc_userargs(userargs, default_args)
    
    fs = dataset['raw'].info['sfreq']
    subject = dataset['subject']
    save_fdr = dataset['target_pth'] / "ckpt" / subject / userargs['ckpt_name']
    ensure_dir(save_fdr)
    
    if userargs['key_to_print'] is None:
        userargs['print_noise'] = userargs['print_pcs'] = False
    
    if f"picks_{userargs['key_to_print']}" in dataset:
        picks = dataset[f"picks_{userargs['key_to_print']}"]
        psd = psd_plot(dataset['raw'], resolution=userargs['resolution'], fs=fs, figsize=userargs['psd_figsize'], fmax=userargs['max_freq'], picks=picks, dB=userargs['dB'])
        psd_plot(dataset['raw'], resolution=userargs['resolution'], fs=fs, figsize=userargs['psd_figsize'], fmax=userargs['max_freq'], save_pth=save_fdr / f"psd.pdf", picks='eeg', dB=userargs['dB'])
    else:
        picks = 'eeg'
        psd = psd_plot(dataset['raw'], resolution=userargs['resolution'], fs=fs, figsize=userargs['psd_figsize'], fmax=userargs['max_freq'], save_pth=save_fdr / f"psd.pdf", picks='eeg', dB=userargs['dB'])
    
    std = np.mean(np.std(dataset['raw'].get_data(picks=userargs['std_channel_pick'], reject_by_annotation='omit'), axis=1))
    plot_channel_dists(dataset['raw'], str(save_fdr / f"std={std:.4e}.pdf"))
    
    def print_ch(ch_name):
        try:
            extra_str = "channels"
            print_fdr = save_fdr / extra_str
            ensure_dir(print_fdr)
            
            psd_plot(dataset['raw'], resolution=userargs['resolution'], fs=fs, figsize=userargs['psd_figsize'], fmax=userargs['max_freq'], picks=ch_name, save_pth=print_fdr / f"{ch_name}_psd.pdf", dB=userargs['dB'])
            temp_plot(dataset['raw'], ch_name, fs=fs, save_pth=print_fdr / f"{ch_name}.pdf", name=ch_name)
            temp_plot(dataset['raw'], ch_name, fs=fs, start=userargs['focus_range'][0]*fs, length=(userargs['focus_range'][1]-userargs['focus_range'][0])*fs, save_pth=print_fdr / f"{ch_name}_{userargs['focus_range'][0]}-{userargs['focus_range'][1]}.pdf", name=ch_name)
            
            ### Compare with last checkpoint if exists
            if 'last_ckpt_raw' in dataset:
                if dataset['last_ckpt_raw'].info['sfreq'] != dataset['raw'].info['sfreq']:
                    dataset['last_ckpt_raw'].resample(dataset['raw'].info['sfreq'])
                temp_plot_diff(dataset['last_ckpt_raw'], dataset['raw'], ch_name, fs=fs, save_pth=print_fdr / f"{ch_name}_diff.pdf",name=ch_name)
                temp_plot_diff(dataset['last_ckpt_raw'], dataset['raw'], ch_name, fs=fs, start=userargs['focus_range'][0]*fs, length=(userargs['focus_range'][1]-userargs['focus_range'][0])*fs, save_pth=print_fdr / f"{ch_name}_{userargs['focus_range'][0]}-{userargs['focus_range'][1]}_diff.pdf", name=ch_name)
        except Exception as e:
            print(f"Error in printing channel {ch_name}: {e}")
            pass
    
    channel_to_print = psd.ch_names
    if 'last_ckpt_print_ch' in dataset and set(dataset['last_ckpt_print_ch']).issubset(set(channel_to_print)):
        channel_to_print = dataset['last_ckpt_print_ch']
    elif len(channel_to_print) > 3:
        channel_to_print = np.random.choice(channel_to_print, 3, replace=False)
        dataset['last_ckpt_print_ch'] = channel_to_print
    
    channel_to_print = np.unique(np.concatenate([np.array(userargs['always_print']), channel_to_print]))
    for ch_name in channel_to_print:
        print_ch(ch_name)
    
    ### Print PCs of OBS or AAS if requested. Sliding window OBS is also supported, although this is not a commonly used method.
    if userargs['print_pcs']:
        pc_fdr_name = dataset['target_pth'] / "ckpt" / subject / f"pc_{userargs['key_to_print']}"
        ensure_dir(pc_fdr_name)
        
        pcs_plot(dataset[f"pc_{userargs['key_to_print']}"], pc_fdr_name, channel_to_print, psd.ch_names, info=psd.info, resolution=userargs['resolution'], psd_lim=(0, userargs['max_freq']))
    
    ### Print noise components if requested, could be somewhat redundant with the diff plot, but can provide more insights into the nature of the noise
    if userargs['print_noise']:
        noise_fdr_name = dataset['target_pth'] / "ckpt" / subject / f"noise_{userargs['key_to_print']}"
        ensure_dir(noise_fdr_name)
        for ch_name in channel_to_print:
            psd_plot(dataset[f"noise_{userargs['key_to_print']}"], resolution=userargs['resolution'], fs=fs, figsize=userargs['psd_figsize'], fmax=userargs['max_freq'], picks=[ch_name], save_pth=noise_fdr_name / f"{ch_name}_psd.pdf", dB=userargs['dB'])
            temp_plot(dataset[f"noise_{userargs['key_to_print']}"], ch_name, fs=fs, save_pth=noise_fdr_name / f"{ch_name}.pdf", name=ch_name)
        
        psd_plot(dataset[f"noise_{userargs['key_to_print']}"], resolution=userargs['resolution'], fs=fs, figsize=userargs['psd_figsize'], fmax=userargs['max_freq'], save_pth=noise_fdr_name / f"noise_psd.pdf", picks='eeg', dB=userargs['dB'])
    
    ### Store the current raw data for diff comparison in the next checkpoint
    dataset['last_ckpt_raw'] = copy.deepcopy(dataset['raw'])
    
    ### Log tracer metrics if requested
    if userargs['log_tracer']:
        if picks in ['eeg', 'all', 'data'] or 'eeg' in picks:
            dataset['tracer'].checkpoint(dataset['raw'].get_data(picks='eeg'), name=userargs['ckpt_name'])
            dataset['tracer'].checkpoint_psd(psd, name=userargs['ckpt_name'])
        else:
            log_or_print(f"Warning: EEG channels not fully included in picks for checkpointing tracer. Current picks: {picks}. Tracer logging skipped for this checkpoint.")
    
    return dataset

def crop_TR(dataset, userargs):
    """
    Crops the dataset to the TRs of the fMRI data.
    userargs{event_reference: bool} - If True, after cropping, the event would be overwritten to the event in dataset["raw"].
    """
    TR = userargs.get('TR', 1.14)
    tmin = userargs.get('tmin', -0.04*1.14)
    event_name = userargs.get('event_name', None)
    num_edge_TR = userargs.get('num_edge_TR', 0)

    freq = dataset['raw'].info['sfreq']
    if event_name is None:
        assert 'tr_event_key' in dataset, "Please provide the event name for cropping TR or set 'tr_event_key' in the dataset."
        event_name = dataset['tr_event_key']

    if isinstance(event_name, list):
        for event in event_name:
            if str(event) in mne.events_from_annotations(dataset['raw'])[1]:
                event_name = event
                break
        else:
            raise ValueError(f"None of the provided event names {event_name} are found in the raw annotations {list(mne.events_from_annotations(dataset['raw'])[1].keys())}. Please check the event names or the raw annotations.")

    def crop_eeg_to_tr(eeg, tmin, num_edge_TR=0):           
        trig = mne.events_from_annotations(eeg)[1][str(event_name)]
        
        start_point = end_point = -1
        for timepoint, _, trig_value in mne.events_from_annotations(eeg)[0]:
            if trig_value == trig:
                if start_point == -1:
                    start_point = timepoint - eeg.first_samp
                end_point = timepoint+TR*freq - eeg.first_samp
        
        new_tmin = max(start_point/freq+tmin+num_edge_TR*TR, 0)
        tmax = end_point/freq-num_edge_TR*TR
        try:
            new_tmin = max(new_tmin, eeg.tmin)
            tmax = min(tmax, eeg.tmax)
        except AttributeError as e:
            if 'object has no attribute' in str(e):
                log_or_print(f"Warning: {e}")
            else:
                raise e
        eeg = eeg.crop(tmin=new_tmin, tmax=tmax, include_tmax=False)  
        return eeg
    
    dataset["raw"] = crop_eeg_to_tr(dataset["raw"], tmin=tmin, num_edge_TR=num_edge_TR)
    return dataset 

def crop_by_epoch(dataset, userargs):
    """
    Crops the dataset to the epochs of the EEG data.
    """
    
    epoch_name = userargs.get('epoch_name', 'sim_ep')
    num_edge_epoch = userargs.get('num_edge_epoch', 0)
    
    epoch = dataset[epoch_name]
    events = copy.deepcopy(epoch.events)
    events = events[np.argsort(events[:, 0])]  # sort events by timepoint
    
    start_point = events[0,0] - dataset['raw'].first_samp
    end_point = events[-1,0] + epoch.tmax*dataset['raw'].info['sfreq'] - dataset['raw'].first_samp
    
    edge_time_crop = num_edge_epoch*(epoch.tmax-epoch.tmin)
    new_tmin = max(start_point/dataset['raw'].info['sfreq']+epoch.tmin, dataset['raw'].tmin) + edge_time_crop
    tmax = min(end_point/dataset['raw'].info['sfreq'], dataset['raw'].tmax) - edge_time_crop
    
    dataset["raw"] = dataset["raw"].crop(tmin=new_tmin, tmax=tmax, include_tmax=False)  
    return dataset

def set_channel_type_raw(dataset, userargs):
    remove_trigger = userargs.get('remove_trigger', True)
    
    dataset["raw"].set_channel_types({'VEOG': 'eog'})
    dataset["raw"].set_channel_types({'HEOG': 'eog'})
    dataset["raw"].set_channel_types({'EKG': 'ecg'})
    dataset["raw"].set_channel_types({'EMG': 'emg'})
    
    if 'Trigger' in dataset['raw'].ch_names:
        if remove_trigger:
            dataset['raw'].drop_channels(['Trigger'])
        else:
            dataset["raw"].set_channel_types({'Trigger': 'misc'})
    return dataset

def create_epoch(dataset, userargs):
    event = userargs.get('event', 'TR')
    tmin = userargs.get('tmin', -0.04*1.14)    # remember changing 1.14 to 0.07 if event = slice!
    tmax = userargs.get('tmax', 0.97*1.14)      # note that the 'tmax' is in a matlab style, i.e. tmax-tmin is not the length of the epoch, but +1 timepoint
    random = userargs.get('random', False)
    event_name = userargs.get('event_name', None)
    epoch_name_diy = userargs.get('epoch_name', None)   # if None, will be set to event + '_ep' or event + '_ep_rand' if random is True
    correct_trig = userargs.get('correct_trig', False)  # whether to correct the trigger event using pearson correlation. only works for 'TR' event.
    
    if event == 'TR':
        epoch_name = 'tr_ep' if not random else 'tr_ep_rand'
        if epoch_name in dataset:
            events = dataset[epoch_name].events
            events[: ,0] //= int(dataset[epoch_name].info['sfreq'] // dataset['raw'].info['sfreq'])
            events[:, 0] = events[:, 0].astype(np.int64)
            event_id = list(dataset[epoch_name].event_id.values())[0]
        else:
            if event_name is None:
                assert 'tr_event_key' in dataset, "Please provide the event name for cropping TR or set 'tr_event_key' in the dataset."
                event_name = dataset['tr_event_key']
            if isinstance(event_name, list):
                for event in event_name:
                    if str(event) in mne.events_from_annotations(dataset['raw'])[1]:
                        event_name = event
                        break
                else:
                    raise ValueError(f"None of the provided event names {event_name} are found in the raw annotations {list(mne.events_from_annotations(dataset['raw'])[1].keys())}. Please check the event names or the raw annotations.")
            event_id = mne.events_from_annotations(dataset['raw'])[1][str(event_name)]
            events = mne.events_from_annotations(dataset['raw'])[0]
            if correct_trig:
                events = correct_trigger(dataset['raw'], events, event_id, tmin=tmin, tmax=tmax, template='mid', channel=0, hwin=3)
            if random:
                tr_tp_list = events[events[:,-1]==event_id][:,0]            
                rand_tp_list = np.sort(np.random.choice(np.arange(np.min(tr_tp_list), np.max(tr_tp_list)), size=len(tr_tp_list), replace=False))
                events = rand_tp_list.reshape(-1, 1)
                events = np.concatenate([events, np.zeros_like(events), np.ones_like(events)], axis=1)
        
    elif event == 'He':  # tmin and tmax are not used
        if event_name is None:
            assert 'he_event_key' in dataset, "Please provide the event name for cropping He or set 'he_event_key' in the dataset."
            event_name = dataset['he_event_key']
        for name in event_name:
            if name in mne.events_from_annotations(dataset['raw'])[1]:
                event_id = mne.events_from_annotations(dataset['raw'])[1][name]
                break
        else:
            raise ValueError(f"None of the provided event names {event_name} are found in the raw annotations {list(mne.events_from_annotations(dataset['raw'])[1].keys())}. Please check the event names or the raw annotations.")            
            
        events = mne.events_from_annotations(dataset['raw'])[0]
        he_tp_list = events[events[:,-1]==event_id][:,0]
        time_diff = np.diff(he_tp_list)
        tmax = min(np.median(time_diff)*1.02, np.max(time_diff)) / dataset['raw'].info['sfreq']
        tmin = 0
        epoch_name = 'he_ep'
        
        if random:
            epoch_name = 'he_ep_rand'
            rand_tp_list = np.sort(np.random.choice(np.arange(np.min(he_tp_list), np.max(he_tp_list)), size=len(he_tp_list), replace=False))
            events = rand_tp_list.reshape(-1, 1)
            events = np.concatenate([events, np.zeros_like(events), np.ones_like(events)], axis=1)
    elif event == 'simulate':
        ### WARNING: random in this case represents the percentage of noise in the epoch timepoints, not the random sampling of the events.
        epoch_diff = tmax*dataset['raw'].info['sfreq']
        rand_range = int(epoch_diff*random)
        
        tp_list = np.arange(dataset['raw'].first_samp, dataset['raw'].last_samp, epoch_diff)
        if rand_range > 0:
            tp_list = tp_list + np.random.randint(-rand_range, rand_range, size=len(tp_list))
        events = tp_list.reshape(-1, 1).astype(np.int64)
        events = np.concatenate([events, np.zeros_like(events), np.ones_like(events)], axis=1)
        epoch_name = 'sim_ep'
        event_id = 1
    else:
        raise ValueError(f"Event {event} not recognized.")

    if epoch_name_diy is not None:
        epoch_name = epoch_name_diy
    dataset[epoch_name] = mne.Epochs(dataset['raw'], events=events, tmin=tmin, tmax=tmax, event_id=event_id, baseline=None, proj=False, preload=True)
    
    return dataset

def epoch_ssp(dataset, userargs):
    ssp = userargs.get('ssp', 0)
    epoch_key = userargs.get('epoch_key', 'tr_ep')
    apply = userargs.get('apply', False)  # whether to apply all projections including the SSP. 
    
    proj = mne.compute_proj_epochs(dataset[epoch_key], n_grad=0, n_mag=0, n_eeg=ssp, verbose=True)
    dataset['raw'].add_proj(proj)
    
    if apply: 
        dataset['raw'].apply_proj()
        
    # TODO: add option to save the SSP components & noise in the dataset for visualization in the ckpt_report step. This can help with deciding whether to apply the SSP or not, and how many components to use. Currently this function is just realized in AAS and OBS.
    return dataset

def epoch_aas(dataset, userargs):
    epoch_key = userargs.get('epoch_key', 'tr_ep')
    window_length = userargs.get('window_length', 10)
    picks = userargs.get('picks', 'eeg')
    overwrite = userargs.get('overwrite', 'new')
    fit = userargs.get('fit', False)  # if False, standard AAS is used. if True, the avg template is fitted to the data first and then subtracted.
    pre_pad = userargs.get('pre_pad', 0.5)  # in percentage, the padding before the first epoch. 1-pre_pad is the padding after the last epoch.
    
    orig_data = torch.tensor(dataset[epoch_key].get_data(picks=picks))  # 29+#win, #ch, len(ep)
    spurious_data = orig_data.unfold(0, window_length, 1)  # #win, #ch, len(ep), len(win)=#ep

    all_pcs = torch.mean(spurious_data, dim=-1).unsqueeze(-1)  # #win, #ch, len(ep), 1
    
    pre_padding = int(pre_pad * (window_length-1))
    post_padding = window_length - pre_padding - 1

    if pre_padding > 0 and post_padding > 0:
        pre_padding = torch.repeat_interleave(all_pcs[0:1], pre_padding, dim=0)
        post_padding = torch.repeat_interleave(all_pcs[-1:], post_padding, dim=0)
        all_pcs = torch.cat([pre_padding, all_pcs, post_padding], dim=0)
    elif pre_padding > 0:
        pre_padding = torch.repeat_interleave(all_pcs[0:1], pre_padding, dim=0)
        all_pcs = torch.cat([pre_padding, all_pcs], dim=0)
    else:
        post_padding = torch.repeat_interleave(all_pcs[-1:], post_padding, dim=0)
        all_pcs = torch.cat([all_pcs, post_padding], dim=0)

    if fit:
        noise = lstsq(all_pcs, orig_data)[0].unsqueeze(-1)   # 29+#win, #ch, #pc, 1
        noise = (all_pcs @ noise)[...,0]
        cleaned = np.array(orig_data - noise)
    else:
        cleaned = np.array(orig_data - all_pcs.squeeze())
    
    pc_name = f"pc_{epoch_key}"
    noise_name = f"noise_{epoch_key}"
    picks_name = f"picks_{epoch_key}"

    # assert pc_name not in dataset, f"pc_name {pc_name} already exists in dataset. Please use a different name."
    # assert noise_name not in dataset, f"noise_name {noise_name} already exists in dataset. Please use a different name."
    # assert picks_name not in dataset, f"picks_name {picks_name} already exists in dataset. Please use a different name."
    while True:
        if pc_name in dataset:
            pc_name = pc_name + "_"
            continue
        if noise_name in dataset:
            noise_name = noise_name + "_"
            continue
        if picks_name in dataset:
            picks_name = picks_name + "_"
            continue
        break
        
    dataset[noise_name] = copy.deepcopy(dataset['raw'].get_data())
    
    dataset[pc_name] = all_pcs
    dataset[picks_name] = picks
    dataset['raw'] = mne_epoch2raw(dataset[epoch_key], dataset['raw'], cleaned, tmin=dataset[epoch_key].tmin, overwrite=overwrite, picks=picks)
    dataset[noise_name] = dataset[noise_name] - dataset['raw'].get_data()
    dataset[noise_name] = mne.io.RawArray(dataset[noise_name], dataset['raw'].info, first_samp=dataset['raw'].first_samp)

    return dataset
    
def epoch_obs(dataset, userargs):
    epoch_key = userargs.get('epoch_key', 'tr_ep')
    npc = userargs.get('npc', 3)
    picks = userargs.get('picks', 'eeg')
    overwrite = userargs.get('overwrite', 'even')
    remove_mean = userargs.get('remove_mean', True)    # Niazy obs does not remove mean like standard pca, as their slice epoched obs could have <0.1s epochs, removing mean would definitely remove signal. From our experience, when using volume epoch obs & not remove_mean, step noise could appear at epoch boarders. TL;DR: tune this to True when using tr_ep or he_ep, tune this to False when using slice_ep.
    pc_from_spurious = userargs.get('pc_from_spurious', True)  # if True, the PC is calculated from all events, else it is calculated from the safe epochs. This parameter is only used for BCG correction, where the heartbeat detection could mistake residual GA / motion as heartbeats.
    apply_to_spurious = userargs.get('apply_to_spurious', True)  # if True, the PC is applied to the spurious events, else it is not.
    screen_high_power = userargs.get('screen_high_power', None)  # if True, the epochs with high power would not be used for PC calculation. If None, no screening is performed. If false, only the epochs with high power would be used for PC calculation.
    
    
    if pc_from_spurious:
        orig_data = torch.tensor(dataset[epoch_key].get_data(picks=picks)) # #ep, #ch, len(ep)
    else:
        orig_data = torch.tensor(dataset[f"{epoch_key}_safe"].get_data(picks=picks))
        # screener = dataset[f"{epoch_key}_screener"]
        # orig_data = orig_data[screener]
    
    if not screen_high_power is None:
        epoch_power = torch.sum(orig_data**2, dim=(1,2)) # #ep
        power_med = epoch_power.median()
        power_mad = torch.median(torch.abs(epoch_power - power_med))
        threshold = power_med + 3*power_mad
        orig_data = orig_data[epoch_power < threshold] if screen_high_power else orig_data[epoch_power >= threshold]
        
    # orig_data = orig_data.reshape(*orig_data.shape[1:], orig_data.shape[0])
    orig_data = orig_data.permute(1, 2, 0)  # #ch, len(ep), #ep
    
    # epoch_std = orig_data.std(dim=(0,1))
    # normal_ep = epoch_std < (epoch_std.mean() + 3*epoch_std.std())
    # orig_data = orig_data[..., normal_ep]  # #ch, len(ep), #ep
    
    pca_mean = torch.mean(orig_data, dim=1) * int(remove_mean)    # #ch, #ep
    dirty_data = orig_data - pca_mean.unsqueeze(1)
    U, S, _ = torch.pca_lowrank(dirty_data)   # #ch, len(ep), q;    # #ch, q
    all_pcs = U[..., :npc]*S[..., None, :npc]
        
    del orig_data, dirty_data, U, S  # free memory
    if apply_to_spurious:
        orig_data = torch.tensor(dataset[epoch_key].get_data(picks=picks))  # #ep, #ch, len(ep)
    else:
        orig_data = torch.tensor(dataset[f"{epoch_key}_safe"].get_data(picks=picks))
    orig_data = orig_data.permute(1, 2, 0)  # #ch, len(ep), #ep
    pca_mean = torch.mean(orig_data, dim=1) * int(remove_mean)    # #ch, #ep
    dirty_data = orig_data - pca_mean.unsqueeze(1)  # #ch, len(ep), #ep
    noise = lstsq(all_pcs, dirty_data)[0]   # #ch, #pc, #ep
    noise = all_pcs @ noise + pca_mean.unsqueeze(1)  # #ch, len(ep), #ep
    
    cleaned = np.array((orig_data - noise).permute(2, 0, 1))
        
    pc_name = f"pc_{epoch_key}"
    noise_name = f"noise_{epoch_key}"
    picks_name = f"picks_{epoch_key}"
        
    # To avoid overwriting existing keys in the dataset, we append underscores until we find a unique key for pc_name, noise_name, and picks_name. e.g. if you run a pipeline with AAS+OBS, the AAS step would create keys 'pc_tr_ep' and 'noise_tr_ep', then the OBS step would create keys 'pc_tr_ep_' and 'noise_tr_ep_' to avoid overwriting the AAS results.
    while True:
        if pc_name in dataset:
            pc_name = pc_name + "_"
            continue
        if noise_name in dataset:
            noise_name = noise_name + "_"
            continue
        if picks_name in dataset:
            picks_name = picks_name + "_"
            continue
        break
        
    dataset[noise_name] = copy.deepcopy(dataset['raw'].get_data())
    dataset[pc_name] = all_pcs
    dataset[picks_name] = picks
    dataset['raw'] = mne_epoch2raw(dataset[epoch_key], dataset['raw'], cleaned, tmin=dataset[epoch_key].tmin, overwrite=overwrite, picks=picks)
    dataset[noise_name] = dataset[noise_name] - dataset['raw'].get_data()
    dataset[noise_name] = mne.io.RawArray(dataset[noise_name], dataset['raw'].info, first_samp=dataset['raw'].first_samp)
    return dataset

def slice_ica(dataset, userargs):
    """perform ICA on raw data the TR epochs to remove residual slice artifacts.

    This function applies ICA to the TR epochs in the dataset, excluding components based on the PSD.
    It modifies the raw data in the dataset by applying the ICA solution.

    Args:
        dataset (_type_): _description_
        userargs (_type_): _description_

    Returns:
        _type_: _description_
    """
    seed = userargs.get('seed', 42)
    max_iter = userargs.get('max_iter', 'auto')
    n_components = userargs.get('n_components', .999) # number of components to keep, default is 0.999, which means keep all components that explain at least 99.9% of the variance
    epoch_frange = userargs.get('epoch_frange', [1,40])
    noise2base_threshold = userargs.get('noise2base_threshold', 4.0)  # noise to base threshold to identify slice artifact.
    noise_window = userargs.get('noise_window', 1.0)  # TR freqs around each slice harmonics to consider as "noise" for SNR estimate
    base_window = userargs.get('base_window', 5.0)    # TR freqs around each slice harmonics to consider as "base" for SNR estimate
    
    slice_freq = 1/dataset['slice_interval']
    tr_freq = 1/dataset['tr_interval']
    
    if "slice_ica_n2b_threshold" in dataset:
        noise2base_threshold = dataset["slice_ica_n2b_threshold"]
        log_or_print(f"using noise2base_threshold: {noise2base_threshold} defined in initialize()")
    
    assert base_window > noise_window, "base_window should be greater than noise_window."

    ica = ICA(n_components=n_components, max_iter=max_iter, random_state=seed)
    ica.fit(copy.deepcopy(dataset['raw']).filter(l_freq=1, h_freq=None), picks='eeg')

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
            noise = mean_psd_in_band(psd_row, freqs, harmonic, noise_window*tr_freq/2)
            base = mean_psd_in_band(psd_row, freqs, harmonic, base_window*tr_freq/2)
            base = (base * base_window - noise * noise_window) / (base_window - noise_window)
            if (noise / (base+eps)) > noise2base_threshold:
                exclude_list.append(ic)
                break
    
    ica.exclude = exclude_list
    # dataset['raw_before_ica'] = copy.deepcopy(dataset['raw'])
    dataset['raw'] = ica.apply(dataset['raw'].copy())
    return dataset

# ---------- Timer wrappers below ----------

class _TimerRegistry:
    """
    Singleton registry to store running timers and their history across subjects/runs.
    No threading or dataset dependency — purely global, group-level stats.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_TimerRegistry, cls).__new__(cls)
            cls._instance._running = {}   # timer_idx -> {'start': datetime, 'meta': ...}
            cls._instance._history = {}   # timer_idx -> [float durations in seconds]
        return cls._instance

    def start(self, timer_idx, meta=None):
        """Start a timer; raise if already exists."""
        if timer_idx in self._running:
            raise AssertionError(f"Timer '{timer_idx}' already running.")
        self._running[timer_idx] = {'start': datetime.now(), 'meta': meta}

    def end(self, timer_idx):
        """End a timer, return elapsed seconds. Raise if timer not running."""
        if timer_idx not in self._running:
            raise KeyError(f"Timer '{timer_idx}' not found (not running).")
        rec = self._running.pop(timer_idx)
        elapsed = (datetime.now() - rec['start']).total_seconds()
        self._history.setdefault(timer_idx, []).append(elapsed)
        return elapsed

    def get_history(self, timer_idx):
        """Return the list of historical durations for timer_idx (may be empty)."""
        return list(self._history.get(timer_idx, []))

    def clear_history(self, timer_idx=None):
        """Clear history for one timer or all if timer_idx is None."""
        if timer_idx is None:
            self._history.clear()
        else:
            self._history.pop(timer_idx, None)


# module-level singleton instance
_timer_registry = _TimerRegistry()


# ---------- User-facing wrapper functions ----------
def start_timer(dataset, userargs):
    """
    Start a named timer.

    Required userargs:
      - timer_idx: unique identifier for this timer (string/int/other hashable)

    Optional userargs:
      - meta: any object to store alongside the start (for debugging / tracing)
    """
    timer_idx = userargs.get('timer_idx', None)
    if timer_idx is None:
        raise ValueError("start_timer requires 'timer_idx' in userargs.")
    meta = userargs.get('meta', None)

    _timer_registry.start(timer_idx, meta=meta)
    log_or_print(f"[TIMER] Started timer '{timer_idx}'.")
    return dataset


def end_timer(dataset, userargs):
    """
    End a named timer, log its elapsed time and group-level history stats (mean ± std).

    Required userargs:
      - timer_idx: identifier for the timer to end

    Optional userargs:
      - print_stats: bool (default True) whether to log mean±std after appending this run
      - fmt: format for printing numeric values (default '%.3f')
      - ddof: degrees of freedom for std calculation (default 0 -> population std)
    """
    timer_idx = userargs.get('timer_idx', None)
    if timer_idx is None:
        raise ValueError("end_timer requires 'timer_idx' in userargs.")

    print_stats = userargs.get('print_stats', True)
    fmt = userargs.get('fmt', '%.3f')
    ddof = userargs.get('ddof', 0)

    # End timer or raise if not exists
    elapsed = _timer_registry.end(timer_idx)

    # compute history stats
    hist = _timer_registry.get_history(timer_idx)
    hist_arr = np.array(hist, dtype=np.float64)
    n = hist_arr.size
    mean = float(hist_arr.mean()) if n > 0 else float('nan')
    std = float(hist_arr.std(ddof=ddof)) if n > 0 else float('nan')

    # Log the single-run elapsed and the updated history summary
    log_or_print(f"[TIMER] Ended timer '{timer_idx}'. Elapsed = {fmt % elapsed} s.")
    if print_stats:
        log_or_print(f"[TIMER] History (n={n}) for '{timer_idx}': mean = {fmt % mean} s, std = {fmt % std} s.")

    return dataset