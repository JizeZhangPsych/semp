import copy

import numpy as np
import torch
from torch.linalg import lstsq
import mne

from ..helpers import mne_epoch2raw


@torch.no_grad()
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

    # To avoid overwriting existing keys, append underscores until a unique key is found.
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
