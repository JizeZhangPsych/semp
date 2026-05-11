import os, re
import numpy as np
import mne
import matplotlib.pyplot as plt
from osl_ephys import source_recon
from osl_ephys.utils.logger import log_or_print
from scipy.signal import welch
from nilearn.plotting import plot_markers
from osl_ephys.source_recon.parcellation import parcel_centers
from semp.utils import ensure_dir
            
def plot_parc(outdir, subject, parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz", nperseg=4096, nfft=None, filename=None, freq_range=[0,45], figsize=(15,3)):
    ensure_dir(f"{outdir}/{subject}/parc_psd")
    if filename is None:
        filename = f"{outdir}/{subject}/parc_psd/psd_{nperseg}"
    else:
        filename = os.path.join(outdir, subject, "parc_psd", filename)
    parcel_data = mne.io.read_raw_fif(f'{outdir}/{subject}/parc/parc-raw.fif', preload=True)
    parc_ts = parcel_data.get_data()
    fs = parcel_data.info['sfreq']
    
    if parc_ts.ndim == 3:
        # Calculate PSD for each epoch individually and average
        psd = []
        for i in range(parc_ts.shape[-1]):
            f, p = welch(parc_ts[..., i], fs=fs, nperseg=nperseg, nfft=nfft)
            psd.append(p)
        psd = np.mean(psd, axis=0)
    else:
        # Calcualte PSD of continuous data
        f, psd = welch(parc_ts, fs=fs, nperseg=nperseg, nfft=nfft)

    n_parcels = psd.shape[0]

    if freq_range is None:
        freq_range = [f[0], f[-1]]

    # Re-order to use colour to indicate anterior->posterior location
    parc_centers = parcel_centers(parcellation_file)
    order = np.argsort(parc_centers[:, 1])
    parc_centers = parc_centers[order]
    psd = psd[order]

    # Plot PSD
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    cmap = plt.cm.viridis_r
    for i in reversed(range(n_parcels)):
        ax.plot(f, psd[i], c=cmap(i / n_parcels))
    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    ax.set_ylabel("PSD (a.u.)", fontsize=14)
    ax.set_xlim(freq_range[0], freq_range[1])
    ax.tick_params(axis="both", labelsize=14)
    plt.tight_layout()

    # Plot parcel topomap
    inside_ax = ax.inset_axes([0.45, 0.55, 0.5, 0.55])
    plot_markers(np.arange(n_parcels), parc_centers, node_size=12, colorbar=False, axes=inside_ax)

    # Save
    log_or_print(f"saving {filename}.png")
    plt.savefig(filename+".png")
    plt.close()
    
    for i in reversed(range(n_parcels)):
        plt.figure(figsize=figsize)
        
        plt.plot(f, psd[i])
        plt.xlim(freq_range[0], freq_range[1])
        plot_markers(np.arange(n_parcels), parc_centers, node_size=12, colorbar=False, axes=inside_ax)
        # Save
        plt.savefig(filename + f"_{i}.png")
        plt.close()
    