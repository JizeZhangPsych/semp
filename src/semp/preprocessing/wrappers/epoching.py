import copy

import numpy as np
import mne
from osl_ephys.utils.logger import log_or_print

from ..helpers import correct_trigger


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
