def epoch_sw_pca(dataset, userargs):
    epoch_key = userargs.get('epoch_key', 'tr_ep')
    npc = userargs.get('npc', 3)
    window_length = userargs.get('window_length', 30)
    force_mean_pc0 = userargs.get('force_mean', True)   # note that this mean has length of epoch_length, while the remove_mean remove the mean with length #epoch
    picks = userargs.get('picks', 'eeg')
    overwrite = userargs.get('overwrite', 'new')
    do_align = userargs.get('do_align', False)
    remove_mean = userargs.get('remove_mean', True)    # bcg obs does not remove mean with length #epoch like pca. WARNING: DO NOT use volume obs or slice pca!
    spurious_event = userargs.get('spurious_event', False)  # if True, epoch_key is used for removal, epoch_key + "_screener" is used for PC calculation
    pre_pad = userargs.get('pre_pad', 0.5)  # in percentage, the padding before the first epoch. 1-pre_pad is the padding after the last epoch.
    
    assert not do_align, "Alignment not implemented yet."
    orig_data = torch.tensor(dataset[epoch_key].get_data(picks=picks))  # 29+#win, #ch, len(ep)

    if spurious_event:
        raise NotImplementedError("Spurious event not implemented yet, please set spurious_event=False.")
        screener = dataset[f"{epoch_key}_screener"]
        epoch_std = orig_data[screener].std(dim=(1,2))  
        within_3_std = epoch_std < (epoch_std.mean()+3*epoch_std.std())
        screener = screener[within_3_std]
    
    pca_mean = torch.mean(orig_data, dim=2) * int(remove_mean)    # 29+#win, #ch
    det_orig_data = orig_data - pca_mean.unsqueeze(2)
    spurious_data = det_orig_data.unfold(0, window_length, 1)  # #win, #ch, len(ep), len(win)=#ep
    if force_mean_pc0:
        pc0 = torch.mean(spurious_data, dim=-1).unsqueeze(-1)  # #win, #ch, len(ep), 1
        detrended = spurious_data - pc0
        U, S, _ = torch.pca_lowrank(detrended)
        all_pcs = U[..., :npc-1]*S[..., None, :npc-1]
        all_pcs = torch.cat([pc0, all_pcs], -1)
    else: 
        U, S, _ = torch.pca_lowrank(spurious_data)
        all_pcs = U[..., :npc]*S[..., None, :npc]   # #win, #ch, len(ep), #pc
    
    
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
    
    # padding = torch.repeat_interleave(all_pcs[0:1], window_length-1, dim=0)
    # all_pcs = torch.cat([padding, all_pcs], dim=0)    # 29+#win, #ch, len(ep), #pc
    
    if spurious_event:
        pass
    
    noise = lstsq(all_pcs, det_orig_data)[0].unsqueeze(-1)   # 29+#win, #ch, #pc, 1
    noise = (all_pcs @ noise)[...,0] + pca_mean.unsqueeze(-1)  # 29+#win, #ch, len(ep)  # [...,0] means squeezing the last dim, not squeeze() for the case #ch=1
    cleaned = np.array(orig_data - noise)
        
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

def epoch_sw_pca_bievent(dataset, userargs):
    epoch_key = userargs.get('epoch_key', 'bcg_ep')
    npc = userargs.get('npc', 3)
    window_length = userargs.get('window_length', 10)
    force_mean_pc0 = userargs.get('force_mean', True)   # note that this mean has length of epoch_length, while the remove_mean remove the mean with length #epoch
    picks = userargs.get('picks', 'eeg')
    overwrite = userargs.get('overwrite', 'even')
    
    orig_data = torch.tensor(dataset[f"{epoch_key}_safe"].get_data(picks=picks))  # #ep, #ch, len(ep)
    
    assert window_length % 2 == 0
    pca_mean = torch.mean(orig_data, dim=2)
    spurious_data = orig_data.unfold(0, window_length, 1)  # #win, #ch, len(ep), len(win)=#ep
    if force_mean_pc0:
        pc0 = torch.mean(spurious_data, dim=-1).unsqueeze(-1)  # #win, #ch, len(ep), 1
        detrended = spurious_data - pc0
        U, S, _ = torch.pca_lowrank(detrended)
        all_pcs = U[..., :npc-1]*S[..., None, :npc-1]
        all_pcs = torch.cat([pc0, all_pcs], -1)
    else: 
        U, S, _ = torch.pca_lowrank(spurious_data)
        all_pcs = U[..., :npc]*S[..., None, :npc]   # #win, #ch, len(ep), #pc
    
    orig_data = torch.tensor(dataset[f"{epoch_key}"].get_data(picks=picks))  # #ep, #ch, len(ep)
    safe_idx = []
    spurious_event = dataset[epoch_key].events
    safe_event = dataset[f"{epoch_key}_safe"].events
    assert spurious_event.shape[0] == orig_data.shape[0], f"Spurious event shape {spurious_event.shape} does not match spurious data shape {orig_data.shape}."
    for idx, _ in enumerate(spurious_event):
        if spurious_event[idx, 0] in safe_event[:,0]:
            safe_idx.append(idx)
    safe_idx = torch.tensor(safe_idx)
    
    all_pcs_padded = torch.empty(orig_data.shape[0], *all_pcs.shape[1:], dtype=orig_data.dtype)  # #win, #ch, len(ep), #pc
    for idx in range(orig_data.shape[0]):
        dist = torch.abs(safe_idx-idx)
        window_idx = safe_idx[torch.argmin(dist)]
        window_idx = min(list(safe_idx).index(window_idx)-window_length//2, 0)
        window_idx = max(window_idx, all_pcs.shape[0]-1)
        all_pcs_padded[idx] = all_pcs[window_idx]
    
    noise = lstsq(all_pcs_padded, orig_data)[0].unsqueeze(-1)   # 29+#win, #ch, #pc, 1
    noise = (all_pcs_padded @ noise)[...,0]  # 29+#win, #ch, len(ep)  # [...,0] means squeezing the last dim, not squeeze() for the case #ch=1
    cleaned = np.array(orig_data - noise)
        
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
    
    dataset[pc_name] = all_pcs_padded
    dataset[picks_name] = picks
    dataset['raw'] = mne_epoch2raw(dataset[epoch_key], dataset['raw'], cleaned, tmin=dataset[epoch_key].tmin, overwrite=overwrite, picks=picks)
    dataset[noise_name] = dataset[noise_name] - dataset['raw'].get_data()
    dataset[noise_name] = mne.io.RawArray(dataset[noise_name], dataset['raw'].info, first_samp=dataset['raw'].first_samp)

    return dataset

def epoch_aas_bievent_mask(dataset, userargs):
    epoch_key = userargs.get('epoch_key', 'bcg_ep')
    window_length = userargs.get('window_length', 10)
    picks = userargs.get('picks', 'eeg')
    overwrite = userargs.get('overwrite', 'new')
    fit = userargs.get('fit', False)  # if False, standard AAS is used. if True, the avg template is fitted to the data first and then subtracted.
    
    orig_data = torch.tensor(dataset[epoch_key].get_data(picks=picks))  # 29+#win, #ch, len(ep)
    # spurious_data = orig_data.unfold(0, window_length, 1)  # #win, #ch, len(ep), len(win)=#ep
    
    unsafe_idx = []
    spurious_event = dataset[epoch_key].events
    safe_event = dataset[f"{epoch_key}_safe"].events
    assert spurious_event.shape[0] == orig_data.shape[0], f"Spurious event shape {spurious_event.shape} does not match spurious data shape {orig_data.shape}."
    for idx, _ in enumerate(spurious_event):
        if not (spurious_event[idx, 0] in safe_event[:,0]):
            unsafe_idx.append(idx)
    unsafe_idx = torch.tensor(unsafe_idx)

    all_pcs = torch.empty_like(orig_data)  # 29+#win, #ch, len(ep)
    for idx in range(orig_data.shape[0]):
        tmp_window_length = window_length
        while True:
            win_left = max(idx - tmp_window_length//2, 0)
            win_left = min(win_left, orig_data.shape[0] - tmp_window_length)
            win_idx = list(range(win_left, win_left + tmp_window_length))
            win_idx = [i for i in win_idx if i not in unsafe_idx]
            if len(win_idx) != 0:
                tmp_window_length = window_length
                break
            tmp_window_length *= 2
        all_pcs[idx] = torch.mean(orig_data[win_idx], dim=0)  # #ch, len(ep)

    all_pcs = all_pcs.unsqueeze(-1)  # 29+#win, #ch, len(ep), 1    


    if fit:
        noise = lstsq(all_pcs, orig_data)[0].unsqueeze(-1)   # 29+#win, #ch, #pc, 1
        noise = (all_pcs @ noise)[...,0]
        cleaned = np.array(orig_data - noise)
    else:
        cleaned = np.array(orig_data - all_pcs.squeeze())
    
    pc_name = f"pc_{epoch_key}"
    noise_name = f"noise_{epoch_key}"
    picks_name = f"picks_{epoch_key}"

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

def epoch_aas_bievent(dataset, userargs):
    epoch_key = userargs.get('epoch_key', 'bcg_ep')
    window_length = userargs.get('window_length', 10)
    picks = userargs.get('picks', 'eeg')
    overwrite = userargs.get('overwrite', 'new')
    fit = userargs.get('fit', False)  # if False, standard AAS is used. if True, the avg template is fitted to the data first and then subtracted.
    
    orig_data = torch.tensor(dataset[epoch_key].get_data(picks=picks))  # 29+#win, #ch, len(ep)
    # spurious_data = orig_data.unfold(0, window_length, 1)  # #win, #ch, len(ep), len(win)=#ep
    
    safe_idx = []
    spurious_event = dataset[epoch_key].events
    safe_event = dataset[f"{epoch_key}_safe"].events
    assert spurious_event.shape[0] == orig_data.shape[0], f"Spurious event shape {spurious_event.shape} does not match spurious data shape {orig_data.shape}."
    for idx, _ in enumerate(spurious_event):
        if spurious_event[idx, 0] in safe_event[:,0]:
            safe_idx.append(idx)
    safe_idx = torch.tensor(safe_idx)

    all_pcs = torch.empty_like(orig_data)  # 29+#win, #ch, len(ep)
    for idx in range(orig_data.shape[0]):
        dist = torch.abs(safe_idx-idx)
        sorted_safe = safe_idx[torch.argsort(dist)]
        window_idx = sorted_safe[:window_length]
        all_pcs[idx] = torch.mean(orig_data[window_idx], dim=0)  # #ch, len(ep)

    all_pcs = all_pcs.unsqueeze(-1)  # 29+#win, #ch, len(ep), 1    

    # all_pcs = torch.mean(spurious_data, dim=-1).unsqueeze(-1)  # #win, #ch, len(ep), 1
    
    # pre_padding = int(pre_pad * (window_length-1))
    # post_padding = window_length - pre_padding - 1

    # if pre_padding > 0 and post_padding > 0:
    #     pre_padding = torch.repeat_interleave(all_pcs[0:1], pre_padding, dim=0)
    #     post_padding = torch.repeat_interleave(all_pcs[-1:], post_padding, dim=0)
    #     all_pcs = torch.cat([pre_padding, all_pcs, post_padding], dim=0)
    # elif pre_padding > 0:
    #     pre_padding = torch.repeat_interleave(all_pcs[0:1], pre_padding, dim=0)
    #     all_pcs = torch.cat([pre_padding, all_pcs], dim=0)
    # else:
    #     post_padding = torch.repeat_interleave(all_pcs[-1:], post_padding, dim=0)
    #     all_pcs = torch.cat([all_pcs, post_padding], dim=0)

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

def impulse_removal(dataset, userargs):
    picks = userargs.get('picks', 'all')
    thres = userargs.get('thres', 3.0)
    iteration = userargs.get('iteration', 1)  # number of iterations to perform impulse removal, 0 or negative means infinite iterations
    
    bad_exist = False
    data = dataset['raw'].get_data(picks=picks)  # #ch, #timepoints
    data_abs_diff = np.abs(data[:, 1:] - data[:, :-1])  # #ch, #timepoints-1
    avg, std = np.mean(data_abs_diff, axis=1), np.std(data_abs_diff, axis=1)  # #ch,
    thres = avg + thres * std  # #ch,
    
    mask = data_abs_diff > thres[:, None]  # #ch, #timepoints-1
    mask = np.concatenate([np.zeros((mask.shape[0], 1), dtype=bool), mask], axis=1)  # #ch, #timepoints
    
    interp_data = data.copy()
    for ch in range(data.shape[0]):
        bad_idx = np.where(mask[ch])[0]
        good_idx = np.where(~mask[ch])[0]
        if len(bad_idx) > 0:
            interp_data[ch, bad_idx] = np.interp(bad_idx, good_idx, data[ch, good_idx])
            bad_exist = True

    dataset['raw']._data[pick_indices(dataset['raw'], picks)] = interp_data  # Update the raw data with the interpolated data
    
    if not bad_exist or iteration == 1:
        return dataset
    else:
        userargs['iteration'] = iteration - 1
        return impulse_removal(dataset, userargs)    

def epoch_impulse_removal(dataset, userargs):
    epoch_key = userargs.get('epoch_key', 'tr_ep')
    overwrite = userargs.get('overwrite', 'new')
    picks = userargs.get('picks', 'all')
    thres = userargs.get('thres', 3.0)
    
    orig_data = dataset[epoch_key].get_data(picks=picks)  # #ep, #ch, len(ep)
    abs_diff_data = np.abs(orig_data[:, :, 1:] - orig_data[:, :, :-1])  # #ep, #ch, len(ep)-1
    avg, std = np.mean(abs_diff_data, axis=-1), np.std(abs_diff_data, axis=-1)
    thres = avg + thres * std  # #ep, #ch
    mask = abs_diff_data > thres[:, :, None]  # #ep, #ch, len(ep)-1
    interp_data = orig_data.copy()
    
    for ep in range(orig_data.shape[0]):
        for ch in range(orig_data.shape[1]):
            ep_data = orig_data[ep, ch, :]
            ep_mask = np.concatenate([mask[ep, ch, :], np.zeros((1,), dtype=bool)])  # #timepoints
            
            bad_idx  = np.flatnonzero(ep_mask)
            good_idx = np.flatnonzero(~ep_mask)
            
            if bad_idx.size > 0:
                interp_data[ep, ch, bad_idx] = np.interp(bad_idx, good_idx, ep_data[good_idx])
    
    dataset['raw'] = mne_epoch2raw(dataset[epoch_key], dataset['raw'], interp_data, tmin=dataset[epoch_key].tmin, overwrite=overwrite, picks=picks)
    return dataset



def build_aecg(dataset, userargs):
    l_freq = userargs.get('l_freq', 1)
    h_freq = userargs.get('h_freq', 40)
    num_components = userargs.get('num_components', 3)
    new_ecg_name = userargs.get('new_ecg_name', 'AECG')

    fs = dataset['raw'].info['sfreq']

    ecg_data = copy.deepcopy(dataset['raw']).pick('eeg', exclude='bads').get_data(picks='eeg', reject_by_annotation='NaN')
    ecg_data = mne.filter.filter_data(ecg_data, sfreq=fs, l_freq=l_freq, h_freq=h_freq, verbose=False)
    
    ecg_data = np.nan_to_num(ecg_data, nan=0.0)
    ecg_data = torch.tensor(ecg_data)
    ecg_data = ecg_data - torch.mean(ecg_data, dim=0).unsqueeze(0)  # #ch, #timepoints
    U, S, _ = torch.pca_lowrank(ecg_data.T)   # [#timepoints, q]; [q,]
    ecg_data = (U[:, :num_components]*S[None, :num_components]).sum(dim=-1)  # #timepoints, 1
    ecg_data = np.array(ecg_data.reshape(1, -1))  # convert to numpy array with shape (#timepoints, 1)
    ecg_data = ecg_data.squeeze()
    new_info = mne.create_info(ch_names=[new_ecg_name], sfreq=fs, ch_types=['ecg'])
    dataset['raw'].add_channels([mne.io.RawArray(ecg_data[None, :], new_info, first_samp=dataset['raw'].first_samp)], force_update_info=True)
    return dataset
    
def bcg_removal(dataset, userargs):
    method = userargs.get('method', 'obs')
    npc = userargs.get('npc', 3)
    overwrite = userargs.get('overwrite', 'obs')
    filt = userargs.get('filt', [1, 40])
    remove_mean = userargs.get('remove_mean', True)
    picks = userargs.get('picks', 'eeg')
    filt_fit_target = userargs.get('filt_fit_target', False)

    if method == 'obs':
        dataset = epoch_pca(dataset, userargs={'epoch_key': 'bcg_ep', 'npc': npc, 'force_mean': True, 'overwrite': overwrite, 'filt': filt, 'tmin':dataset['bcg_ep'].tmin, 'remove_mean': remove_mean, 'picks': picks, 'filt_fit_target': filt_fit_target})
    else:
        raise NotImplementedError(f"Method {method} not implemented.")
    return dataset

def bcg_ep_ica(dataset, userargs):
    """perform ICA on the BCG epochs to remove artifacts.

    This function applies ICA to the BCG epochs in the dataset, excluding components based on their explained variance ratio.
    It modifies the raw data in the dataset by applying the ICA solution.

    Args:
        dataset (_type_): _description_
        userargs (_type_): _description_

    Returns:
        _type_: _description_
    """
    picks = userargs.get('picks', 'eeg')
    seed = userargs.get('seed', 42)
    max_iter = userargs.get('max_iter', 'auto')
    n_components = userargs.get('n_components', .999) # number of components to keep, default is 0.999, which means keep all components that explain at least 99.9% of the variance
    qrs_event_id = userargs.get('qrs_event_id', 999)
    downsample = userargs.get('downsample', 1)    # only set this if the "bcg_ep" is not downsampled!!!
    flatten_epoch_frange = userargs.get('flatten_epoch_frange', [1,40])  # the flattened epochs will be band pass filtered to the specified frequency range, e.g. [1, 40] for 1-40Hz bandpass filtering
    foi = userargs.get('freq_of_interest', [6, 12])  # the frequency range where you're sure not expecting any peaks within. default set to [6,12] Hz
    peak_threshold = userargs.get('peak_threshold', None) # threshold for peak detection in psd, default is 3, which means only peaks with at least 3 times the value of the mean of the surrounding frequencies will be considered as peaks. None if peak detection is not desired.
    ctps_threshold = userargs.get('ctps_threshold', 0.15)
    l_freq = userargs.get('l_freq', 1)
    h_freq = userargs.get('h_freq', None)
    
    assert 'bcg_ep' in dataset, "Please run qrs_detect first to create bcg_ep."
    
    ev = copy.deepcopy(dataset["bcg_ep"].events)
    
    ev[:,0] = ev[:,0] / downsample
    ev = ev.astype(np.int64)
    bcg_ep = mne.Epochs(copy.deepcopy(dataset['raw']).filter(l_freq, h_freq), events=ev, tmin=dataset["bcg_ep"].tmin, tmax=dataset["bcg_ep"].tmax, event_id=qrs_event_id, baseline=None, proj=False)
    bcg_ep.load_data()  # Ensure the data is loaded before applying ICA
    
    ica = ICA(n_components=n_components, max_iter=max_iter, random_state=seed)
    ica.fit(bcg_ep)
    
    if peak_threshold is None:
        exclude_list = []
    else:
        data = ica._transform_epochs(bcg_ep, concatenate=True)
        data = mne.io.RawArray(data, mne.create_info(data.shape[0], sfreq=dataset['raw'].info['sfreq'], ch_types='eeg'))
        psd = data.compute_psd(fmin=flatten_epoch_frange[0], fmax=flatten_epoch_frange[1], n_fft=int(np.round(dataset['raw'].info['sfreq']*20)))  # 20: resolution of 20 bins within 1Hz, 0.05 later = 1/20, 0.1 = 2/20
        
        epoch_freq = 1 / (bcg_ep.tmax-bcg_ep.tmin+1/dataset['raw'].info['sfreq'])
        
        ratio = []
        for idx in range(int(np.floor(foi[0]/epoch_freq)), int(np.ceil(foi[1]/epoch_freq))):
            ratio.append(psd.get_data(fmin=idx*epoch_freq-0.05, fmax=idx*epoch_freq+0.05).max(axis=1) / psd.get_data(fmin=(idx-1)*epoch_freq+0.1, fmax=(idx+1)*epoch_freq-0.1).mean(axis=1))
        ratio = np.array(ratio)
        exclude_list = np.unique(np.where(np.array(ratio)>peak_threshold)[1])
    
    if ctps_threshold is not None:
        # exclude_list.extend(np.where(ica.find_bads_ecg(bcg_ep)[1]>ctps_threshold)[0])
        exclude_list = np.union1d(exclude_list, ica.find_bads_ecg(bcg_ep, threshold=ctps_threshold)[0])
    # exclude_list = np.unique(exclude_list)    
    
    ica.exclude = exclude_list
    dataset['raw_before_ica'] = copy.deepcopy(dataset['raw'])
    dataset['raw'] = ica.apply(dataset['raw'])
    return dataset

def slice_ep_ica(dataset, userargs):
    """perform ICA on raw data the TR epochs to remove residual slice artifacts.

    This function applies ICA to the TR epochs in the dataset, excluding components based on the PSD.
    It modifies the raw data in the dataset by applying the ICA solution.

    Args:
        dataset (_type_): _description_
        userargs (_type_): _description_

    Returns:
        _type_: _description_
    """
    picks = userargs.get('picks', 'eeg')
    seed = userargs.get('seed', 42)
    max_iter = userargs.get('max_iter', 'auto')
    n_components = userargs.get('n_components', .999) # number of components to keep, default is 0.999, which means keep all components that explain at least 99.9% of the variance
    downsample = userargs.get('downsample', 20)    # only set this if the "tr_ep" is not downsampled!!!
    flatten_epoch_frange = userargs.get('flatten_epoch_frange', [1,40])  # the flattened epochs will be band pass filtered to the specified frequency range, e.g. [1, 40] for 1-40Hz bandpass filtering
    l_freq = userargs.get('l_freq', 1)
    event_name = userargs.get('event_name', None)
    h_freq = userargs.get('h_freq', None)
    noise2base_threshold = userargs.get('noise2base_threshold', 4.0)  # noise to base threshold to identify slice artifact.
    noise_window = userargs.get('noise_window', 1.0)  # TR freqs around each slice harmonics to consider as "noise" for SNR estimate
    base_window = userargs.get('base_window', 5.0)    # TR freqs around each slice harmonics to consider as "base" for SNR estimate
    
    slice_freq = 1/dataset['slice_interval']
    tr_freq = 1/dataset['tr_interval']
    
    if "slice_ica_n2b_threshold" in dataset:
        noise2base_threshold = dataset["slice_ica_n2b_threshold"]
        log_or_print(f"using noise2base_threshold: {noise2base_threshold} defined in initialize()")
    
    assert base_window > noise_window, "base_window should be greater than noise_window."
    assert 'tr_ep' in dataset, "Please run create_epoch first to create tr_ep."
    if event_name is None:
        if 'tr_event_key' in dataset:
            event_name = dataset['tr_event_key']
        else:
            event_name = '1200002'
    if isinstance(event_name, list):
        for event in event_name:
            if str(event) in mne.events_from_annotations(dataset['raw'])[1]:
                event_name = event
                break
        else:
            event_name = 'TR'
    try:
        event_id = mne.events_from_annotations(dataset['raw'])[1][str(event_name)]
    except KeyError:
        event_id = mne.events_from_annotations(dataset['raw'])[1]['TR']

    ev = copy.deepcopy(dataset["tr_ep"].events)

    ev[:, 0] = np.round(ev[:, 0] / float(downsample)).astype(int)
    ev = ev.astype(np.int64)
    tr_ep = mne.Epochs(copy.deepcopy(dataset['raw']).filter(l_freq, h_freq), events=ev, tmin=dataset["tr_ep"].tmin, tmax=dataset["tr_ep"].tmax, event_id=event_id, baseline=None, proj=False)
    tr_ep.load_data()  # Ensure the data is loaded before applying ICA
    
    ica = ICA(n_components=n_components, max_iter=max_iter, random_state=seed)
    ica.fit(tr_ep)

    data = ica._transform_epochs(tr_ep, concatenate=True)
    psds, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=dataset['raw'].info['sfreq'],
        fmin=flatten_epoch_frange[0],
        fmax=flatten_epoch_frange[1],
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