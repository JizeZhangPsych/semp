import mne


def epoch_ssp(dataset, userargs):
    ssp = userargs.get('ssp', 0)
    epoch_key = userargs.get('epoch_key', 'tr_ep')
    apply = userargs.get('apply', False)  # whether to apply all projections including the SSP.

    proj = mne.compute_proj_epochs(dataset[epoch_key], n_grad=0, n_mag=0, n_eeg=ssp, verbose=True)
    dataset['raw'].add_proj(proj)

    if apply:
        dataset['raw'].apply_proj()

    # TODO: add option to save the SSP components & noise in the dataset for visualization in the ckpt_report step.
    return dataset
