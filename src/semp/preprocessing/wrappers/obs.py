import copy

import numpy as np
import torch
from torch.linalg import lstsq
import mne

from ..helpers import mne_epoch2raw


@torch.no_grad()
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
        orig_data = torch.tensor(dataset[epoch_key].get_data(picks=picks))  # #ep, #ch, len(ep)
    else:
        orig_data = torch.tensor(dataset[f"{epoch_key}_safe"].get_data(picks=picks))

    if screen_high_power is not None:
        epoch_power = torch.sum(orig_data**2, dim=(1,2))  # #ep
        power_med = epoch_power.median()
        power_mad = torch.median(torch.abs(epoch_power - power_med))
        threshold = power_med + 3*power_mad
        orig_data = orig_data[epoch_power < threshold] if screen_high_power else orig_data[epoch_power >= threshold]

    orig_data = orig_data.permute(1, 2, 0)  # #ch, len(ep), #ep

    pca_mean = torch.mean(orig_data, dim=1) * int(remove_mean)    # #ch, #ep
    dirty_data = orig_data - pca_mean.unsqueeze(1)
    U, S, _ = torch.linalg.svd(dirty_data, full_matrices=False)  # #ch, len(ep), K;  #ch, K;  #ch, K, #ep
    all_pcs = U[..., :npc] * S[..., None, :npc]

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

    # To avoid overwriting existing keys (e.g. if AAS already ran), append underscores.
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
