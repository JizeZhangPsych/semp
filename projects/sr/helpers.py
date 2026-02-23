import numpy as np
import os, re
from pathlib import Path
import mne
from pathfinder import StaresinaRestPathfinder
from functools import partial
from semp.utils import psd_band_ratio
from semp.eeg import SingletonEEG

spurious_subject_list = ['13121', '8111', '8112', '8121', '17111', '17112', '31111', '31112', '31121']

def initialize(dataset, userargs):
    """ Initialize dataset with helper functions.
    
    Args:
        dataset (dict): The dataset dictionary.
        userargs (dict): User arguments.
    
    Returns:
        dict: The initialized dataset.
    
    """
    ### You would want these keys in userargs for the prep wrappers to function, so we add them to dataset for easy access. You can also add other fields if needed.
    dataset['slice_interval'] = userargs.get('slice_interval', 0.07)
    dataset['tr_interval'] = userargs.get('tr_interval', 1.14)
    dataset['tr_event_key'] = userargs.get('tr_event_key', ['1200002'])
    dataset['he_event_key'] = userargs.get('he_event_key', ['128', '132', '192', '196'])
    dataset['target_pth'] = userargs.get('target_pth', Path("/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/after_prep_sr"))
    
    dataset['pf'] = userargs['pf'] if 'pf' in userargs else StaresinaRestPathfinder()
    dataset['subject'] = dataset['pf'].filename2id(dataset['raw'].filenames[0], kind='rest')
    
    ### other initialization steps if needed
    dataset['orig_sfreq'] = dataset['raw'].info['sfreq']
    
    if 'Trigger' in dataset['raw'].ch_names:
        dataset['raw'].drop_channels(['Trigger'])
    
    if dataset['subject'] == '2111':   # radiographer error, one more session recorded after 400s
        try:
            dataset['raw'] = dataset['raw'].crop(tmin=0, tmax=400)  
        except ValueError:
            print("Warning: Subject 2111 has no data after 400s, so no cropping is needed.")
    if dataset['subject'] == '4121':   # Accidental overwriting of resting stage EEG file after computer prompted an overwriting towards the end of the recording. So a very small file is recorded.
        raise Exception("Subject 4121 has its eeg file corrupted: Accidental overwriting of resting stage EEG file after computer prompted an overwriting towards the end of the recording. So a very small file is recorded.")
    if dataset['subject'] == '31212':   # incorrect event triggering after 311s
        try:
            dataset['raw'] = dataset['raw'].crop(tmin=0, tmax=311)  
        except ValueError:
            print("Warning: Subject 31212 has no data after 311s, so no cropping is needed.")
    if dataset['subject'] == '27212':   # forgot one TR event trigger at onset=18.4582s
        dataset['raw'].annotations.append(18.4582, 0, '1200002')
    if dataset['subject'] == '17121':   # forgot one TR event trigger at onset=59.8632s
        dataset['raw'].annotations.append(59.8632, 0, '1200002')
    if dataset['subject'] == '15112':   # noisy TR, requires harsher slica_ica
        dataset['slice_ica_n2b_threshold'] = 1.8
    if dataset['subject'].startswith('811'):
        dataset['slice_ica_n2b_threshold'] = 1.5
    dataset['raw'].drop_channels(['F11', 'F12', 'FT11', 'FT12', 'Cb1', 'Cb2'], on_missing='warn')
    print("Warning: F11, F12, FT11, FT12, Cb1, Cb2 are dropped from the raw data, as no gel is used in these channels.")
    
    ### tracer initialization based on slice_interval
    dataset['tracer'] = {
        'psd_slice': partial(psd_band_ratio, band1=[1/dataset['slice_interval']-1, 1/dataset['slice_interval']+1], band2='beta', fn1=np.mean),
        'psd_2slice': partial(psd_band_ratio, band1=[2/dataset['slice_interval']-1, 2/dataset['slice_interval']+1], band2=[20, 35], fn1=np.mean),
    }
    
    return dataset


def set_channel_montage(dataset, userargs):
    correct_sign = userargs.get('correct_sign', True)
    
    dpo = dataset['pf'][dataset['subject']]['dpo']
    with open(dpo, 'r') as f:
        dpo_content = f.read()
    dpo_content = re.sub(r"#.*?\n", "\n", dpo_content)  # Remove comments
    labels_list_match = re.search(r"LABELS START_LIST([\s\S]*?)LABELS END_LIST", dpo_content)
    assert labels_list_match is not None
    labels_data = labels_list_match.group(1).strip().splitlines()
    sensors_list_match = re.search(r"SENSORS START_LIST([\s\S]*?)SENSORS END_LIST", dpo_content)
    assert sensors_list_match is not None
    sensors_data = sensors_list_match.group(1).strip().splitlines()
    sensors = np.array([line.split() for line in sensors_data], dtype=np.float64)
    
    if correct_sign:
        sign_x = (sensors[labels_data.index('C6')][0] > sensors[labels_data.index('C5')][0]) * 2 - 1
        sign_y = (sensors[labels_data.index('Fz')][1] > sensors[labels_data.index('Pz')][1]) * 2 - 1
        sign_z = (sensors[labels_data.index('Cz')][2] > np.mean(sensors[:,2])) * 2 - 1
        sensors = [[sign_x*x,sign_y*y,sign_z*z] for x,y,z in sensors] 

    sensors = np.array(sensors)*1e-3
    ch_pos = {ch: loc for ch, loc in zip(labels_data, sensors)}
    
    ch_to_drop = ['F11', 'F12', 'FT11', 'FT12', 'Cb1', 'Cb2']
    for ch in ch_to_drop:
        if ch in ch_pos:
            ch_pos.pop(ch)
    
    custom_montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    dataset["raw"].set_montage(custom_montage)
    
    return dataset

