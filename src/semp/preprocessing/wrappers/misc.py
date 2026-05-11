import numpy as np
import mne

from semp.utils import proc_userargs


def voltage_correction(dataset, userargs):
    """Corrects scaling if EOG/ECG/EMG channels are stored in µV but marked as volts."""

    ratio_threshold = userargs.get('ratio_threshold', 1000)  # Default threshold for scaling down EOG/ECG channels
    picks = userargs.get('picks', ['eog', 'ecg', 'emg'])  # Channels to check for scaling

    # Get EEG stats
    eeg_data = dataset['raw'].get_data(picks='eeg', units='V')
    eeg_rms = np.sqrt(np.mean(eeg_data**2, axis=1))
    ref_rms = np.median(eeg_rms)

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
