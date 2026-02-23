import numpy as np
import os, re
from pathlib import Path
import mne
from .pathfinder import WorkingMemoryPathfinder
from functools import partial
from semp.eeg import SingletonEEG
from semp.utils import psd_band_ratio

def initialize(dataset, userargs):
    """ Initialize dataset with helper functions.
    
    Args:
        dataset (dict): The dataset dictionary.
        userargs (dict): User arguments.
    
    Returns:
        dict: The initialized dataset.
    
    """
    ### You would want these keys in userargs for the prep wrappers to function, so we add them to dataset for easy access. You can also add other fields if needed.
    dataset['slice_interval'] = userargs.get('slice_interval', 0.061)
    dataset['tr_interval'] = userargs.get('tr_interval', 1)
    dataset['tr_event_key'] = userargs.get('tr_event_key', ['100005', '1200002'])
    dataset['he_event_key'] = userargs.get('he_event_key', ['128', '132', '192', '196'])
    dataset['target_pth'] = userargs.get('target_pth', Path("/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/after_prep_sr"))
    
    dataset['pf'] = userargs.get('pf', WorkingMemoryPathfinder())
    dataset['subject'] = dataset['pf'].filename2id(dataset['raw'].filenames[0], kind='rest')
    dataset['raw'].info['dev_head_t'] = SingletonEEG("/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/edfs/sub-003_ses-01_run-01_block-01_task-resting_convert.cdt.edf").info['dev_head_t']

    if int(dataset['subject'][:2]) >= 14:
        dataset['raw'].set_channel_types({'VEOG': 'eog', 'HEOG': 'eog', 'EKG': 'ecg', 'EMG': 'emg'})
    else:
        dataset['raw'].set_channel_types({'VEO': 'eog', 'HEO': 'eog', 'EKG': 'ecg'})

    ### other initialization steps if needed
    dataset['orig_sfreq'] = dataset['raw'].info['sfreq']
    dataset['raw'].drop_channels(['F11', 'F12', 'FT11', 'FT12', 'CB1', 'CB2', 'M1', 'M2', 'Cb1', 'Cb2'], on_missing='warn')
    if 'Trigger' in dataset['raw'].ch_names:
        dataset['raw'].drop_channels(['Trigger'])
        
    rename_dict = {}
    for ch in dataset['raw'].ch_names:
        new_ch = ch
        if ch.startswith('FP'):
            new_ch = new_ch.replace('FP', 'Fp')
        if ch.endswith('Z'):
            new_ch = new_ch.replace('Z', 'z')
        rename_dict[ch] = new_ch
    dataset['raw'].rename_channels(rename_dict)

def set_channel_montage(dataset, userargs):
    template = mne.io.read_raw_fif("/ohba/pi/knobre/irene/data_for_jize/clean/visit1/s02/s02_block1_curry_bcg_cleaner_ica_raw.fif")
    rename_dict = {}
    for ch in template.ch_names:
        new_ch = ch
        if ch.startswith('FP'):
            new_ch = new_ch.replace('FP', 'Fp')
        if ch.endswith('Z'):
            new_ch = new_ch.replace('Z', 'z')
        rename_dict[ch] = new_ch
    template.rename_channels(rename_dict)
    montage = template.get_montage()
    
    IRENE_RESCALE = np.array((0.8, 0.92, 1.1))
    LPA_CHANGE = np.array((5, -10, -40)) * 1e-3
    RPA_CHANGE = np.array((-5, -10, -40)) * 1e-3
    NAS_CHANGE = np.array((0, -5, -33)) * 1e-3
    
    for ch_dict in montage.dig:
        if ch_dict['kind'] == mne.io.constants.FIFF.FIFFV_POINT_EEG:
            ch_dict['r'] *= IRENE_RESCALE
        
        elif ch_dict['kind'] == mne.io.constants.FIFF.FIFFV_POINT_CARDINAL:
            ch_dict['r'] *= IRENE_RESCALE
            if ch_dict['ident'] == mne.io.constants.FIFF.FIFFV_POINT_LPA:
                ch_dict['r'] += LPA_CHANGE
            elif ch_dict['ident'] == mne.io.constants.FIFF.FIFFV_POINT_RPA:
                ch_dict['r'] += RPA_CHANGE
            elif ch_dict['ident'] == mne.io.constants.FIFF.FIFFV_POINT_NASION:
                ch_dict['r'] += NAS_CHANGE
    
    dataset["raw"].set_montage(montage)

    return dataset

def create_slice_epoch(dataset, userargs):
    """ Create epochs for each fMRI slice.
    
    Args:
        dataset (dict): The dataset dictionary.
        userargs (dict): User arguments, should contain 'slice_interval', 'tr_interval', and 'tr_event_key'.
        
    Returns:
        dict: The dataset with slice epochs added. 
    """
    
    tmin = userargs.get('tmin', -0.04*1.14)    # remember changing 1.14 to 0.07 if event = slice!
    tmax = userargs.get('tmax', 0.97*1.14)      # note that the 'tmax' is in a matlab style, i.e. tmax-tmin is not the length of the epoch, but +1 timepoint
    random = userargs.get('random', False)
    event_name = userargs.get('event_name', None)
    epoch_name_diy = userargs.get('epoch_name', None)   # if None, will be set to event + '_ep' or event + '_ep_rand' if random is True
    l_freq = userargs.get('l_freq', None)  # low frequency cutoff for the bandpass filter
    h_freq = userargs.get('h_freq', None)
    ssp = userargs.get('ssp', 0)
    
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
    if random:
        epoch_name = 'slice_ep_rand'
        
        tp_list = mne.events_from_annotations(dataset['raw'])[0]
        tp_list = tp_list[tp_list[:,2] == event_id][:,0]
        rand_tp_list = np.sort(np.random.choice(np.arange(np.min(tp_list), np.max(tp_list)), size=16*len(tp_list), replace=False))
        events = rand_tp_list.reshape(-1, 1)
    else:
        epoch_name = 'slice_ep'
        
        subject = dataset['subject']
        abnormal_mat_fp = os.path.join(dataset['pf'].fdr['slice'], f"{subject}.mat")
        if os.path.exists(abnormal_mat_fp):
            onset = loadmat(abnormal_mat_fp)['UniqueTiming']
        else:
            onset = np.arange(0, 0.07*16, 0.07)
        onset = (onset * dataset['raw'].info['sfreq']).astype(np.int64)
        
        slice_tp_list = []
        for timepoint, _, trig_value in mne.events_from_annotations(dataset['raw'])[0]:
            if trig_value == event_id:
                slice_tp_list.append(timepoint + onset)
        events = np.concatenate(slice_tp_list).reshape(-1, 1)
    events = np.concatenate([events, np.zeros_like(events), np.ones_like(events)], axis=1)
    event_id = 1
        
    if epoch_name_diy is not None:
        epoch_name = epoch_name_diy
    if l_freq is not None or h_freq is not None:
        ep_raw = copy.deepcopy(dataset['raw'])
        ep_raw = ep_raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params={"order": 4, "ftype": "butter"})
    else:
        ep_raw = dataset['raw']
    dataset[epoch_name] = mne.Epochs(ep_raw, events=events, tmin=tmin, tmax=tmax, event_id=event_id, baseline=None, proj=False)
    
    if ssp > 0: 
        proj = mne.compute_proj_epochs(dataset[epoch_name], n_grad=0, n_mag=0, n_eeg=ssp, verbose=True)
        dataset['raw'].add_proj(proj)
        dataset['raw'].apply_proj()
    
    return dataset