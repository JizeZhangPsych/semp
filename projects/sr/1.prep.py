#%%
import numpy as np
from osl_ephys.preprocessing import run_proc_chain, run_proc_batch
from pathlib import Path
from pathfinder import StaresinaRestPathfinder

from semp.eeg import crop_TR, epoch_aas, epoch_obs, create_epoch, ckpt_report, slice_ica, init_tracer, summary, mid_crop, set_channel_type_raw, voltage_correction
from helpers import initialize, set_channel_montage

continue_interrupt = True
target_pth = Path("/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/after_prep_sr")
pf = StaresinaRestPathfinder()

config = {
    'preproc': [
        {'initialize': {'target_pth': target_pth, 'pf': pf}},
        {'init_tracer': {}},
        {'set_channel_types': {'VEOG': 'eog', 'HEOG': 'eog', 'EKG': 'ecg', 'EMG': 'emg'}},
        {'set_channel_montage': {}},
        {'notch_filter': {'freqs': '50 100'}},
        {'crop_TR': {'tmin': 0, 'TR': 1.14}},
        {'ckpt_report': {'ckpt_name': 'raw', 'focus_range': [0, 10], 'dB': False}},
        {'create_epoch': {'event': 'TR', 'tmin': 0, 'tmax': 1.14, 'correct_trig': True}},
        {'epoch_aas': {'epoch_key': 'tr_ep', 'overwrite': 'new', 'picks': 'all', 'window_length': 30, 'fit': False}},
        {'voltage_correction': {}},
        {'ckpt_report': {'ckpt_name': 'after_aas_removal', 'key_to_print': 'tr_ep', 'dB': False}},
        {'filter': {'l_freq': 0.5, 'h_freq': 125, 'method': 'iir', 'iir_params': {'order': 5, 'ftype': 'butter'}}},
        {'mid_crop': {'edge': 5}},
        {'resample': {'sfreq': 250}},
        {'ckpt_report': {'ckpt_name': 'after_filt', 'dB': False}},
        {'bad_segments': {'segment_len': 500, 'picks': 'eeg', 'significance_level': 0.1, 'detect_zeros': False}},
        {'bad_segments': {'segment_len': 500, 'picks': 'eeg', 'mode': 'diff', 'significance_level': 0.1, 'detect_zeros': False}},
        {'bad_channels': {'picks': 'eeg', 'significance_level': 0.1}},
        {'bad_segments': {'segment_len': 2500, 'picks': 'eog', 'detect_zeros': False}},
        {'slice_ica': {}},
        {'ckpt_report': {'ckpt_name': 'after_bads_trica', 'dB': False}},
        {'ica_raw': {'n_components': 0.999, 'picks': 'eeg', 'l_freq': 1}},
        {'ica_autoreject': {'eogmeasure':'correlation', 'eogthreshold' : 0.35, 
                            'ecgmethod':'ctps', 'ecgthreshold': 0.1, 'apply': True}},
        {'bad_channels': {'picks': 'eeg', 'significance_level': 0.1}},
        {'interpolate_bads': {}},
        {'ckpt_report': {'ckpt_name': 'after_ica', 'dB': False}},
        {'set_eeg_reference': {'projection': True}},
        {'summary': {}}
    ]
}

subject_list = list(pf.keys())
file_list = [pf[subject]['rest'] for subject in subject_list]

if continue_interrupt:
    full_file_list = file_list.copy()
    file_list = []
    full_subject_list = subject_list.copy()
    subject_list = []
    
    # Check if files already processed
    finished_list = target_pth.glob(f'*/*_preproc-raw.fif')
    finished_list = [full_string.parts[-2] for full_string in finished_list]
    error_list = target_pth.glob(f'logs/*.error.log')
    error_list = [full_string.parts[-1].split('_')[0] for full_string in error_list]
    for subject, preproc in zip(full_subject_list, full_file_list):
        if subject in finished_list:
            print(f"WARNING: {subject} already finished, skipping")
        elif subject in error_list:
            print(f"WARNING: {subject} had an error, skipping")
        else:
            file_list.append(preproc)
            subject_list.append(subject)
    
run_proc_batch(config, file_list, subjects=subject_list, outdir=str(target_pth), gen_report=False, overwrite=True, extra_funcs=[set_channel_montage, crop_TR, set_channel_type_raw, epoch_obs, create_epoch, ckpt_report, initialize, epoch_aas, voltage_correction, summary, init_tracer, mid_crop, slice_ica])
