"""
Standalone reimplementation of osl_dynamics utilities used by semp.visualize.

Copied and adapted from osl-dynamics (https://github.com/OHBA-analysis/osl-dynamics).
This module removes osl_dynamics as a hard dependency while preserving the same API.

Covered functionality:
  - files: check_exists, mask_directory, parcellation_directory
  - parcellation: Parcellation class, parcel_vector_to_voxel_grid
  - plotting: plot_line, plot_psd_topo, plot_brain_surface
  - power: variance_from_spectra, power_save
  - connectivity: connectivity_save
"""

import logging
import os
import warnings
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import nilearn
from nilearn import image
from nilearn.plotting import plot_markers, plot_img_on_surf, plot_connectome


# We rely on nilearn >= 0.10 (resample_to_img exists). The optional
# ``copy_header`` and ``force_resample`` kwargs are feature-detected at
# call sites (see _resample_kwargs below). Bump this floor if anything in
# this module uses kwargs that are gone in older nilearn.
_NILEARN_MIN = (0, 10)
_nilearn_ver_tuple = tuple(int(x) for x in nilearn.__version__.split('.')[:2]
                           if x.isdigit())
if _nilearn_ver_tuple and _nilearn_ver_tuple < _NILEARN_MIN:
    warnings.warn(
        f'semp.utils.osld_extension: nilearn=={nilearn.__version__} is older '
        f'than the supported floor {_NILEARN_MIN[0]}.{_NILEARN_MIN[1]}; '
        f'plotting may fail.', stacklevel=2,
    )

try:
    from tqdm.auto import trange
except ImportError:
    trange = range

_logger = logging.getLogger(__name__)
logging.getLogger("matplotlib.category").setLevel(logging.ERROR)


# ──────────────────────────────────────────────────────────────────────────────
# Directory constants — point to NIfTI files bundled inside osl_dynamics.
# If osl_dynamics is installed the paths are resolved dynamically; otherwise
# users must supply absolute paths to mask/parcellation files.
# ──────────────────────────────────────────────────────────────────────────────
try:
    import importlib.util as _ilu
    _spec = _ilu.find_spec("osl_dynamics")
    if _spec is not None:
        _osld_base = Path(_spec.origin).parent
        mask_directory = str(_osld_base / "files" / "mask")
        parcellation_directory = str(_osld_base / "files" / "parcellation")
    else:
        mask_directory = ""
        parcellation_directory = ""
except Exception:
    mask_directory = ""
    parcellation_directory = ""


# ──────────────────────────────────────────────────────────────────────────────
# File utilities  (osl_dynamics.files.functions)
# ──────────────────────────────────────────────────────────────────────────────
def check_exists(filename, directory=""):
    """Look for a file on disk, falling back to a bundled-files directory."""
    if not os.path.exists(filename):
        if directory and os.path.exists(f"{directory}/{filename}"):
            filename = f"{directory}/{filename}"
        else:
            raise FileNotFoundError(filename)
    return filename


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers (osl_dynamics.utils.misc / array_ops / analysis.spectral)
# ──────────────────────────────────────────────────────────────────────────────
def _override_dict_defaults(default_dict, override_dict=None):
    if override_dict is None:
        override_dict = {}
    return {**default_dict, **override_dict}


def _validate_array(array, correct_dimensionality, allow_dimensions, error_message):
    array = np.array(array)
    for dimensionality in allow_dimensions:
        if array.ndim == dimensionality:
            for _ in range(correct_dimensionality - dimensionality):
                array = array[np.newaxis, ...]
    if array.ndim != correct_dimensionality:
        raise ValueError(error_message)
    return array


def _get_frequency_args_range(frequencies, frequency_range):
    f_min_arg = np.argwhere(frequencies >= frequency_range[0])[0, 0]
    f_max_arg = np.argwhere(frequencies <= frequency_range[1])[-1, 0]
    if f_max_arg <= f_min_arg:
        raise ValueError("Cannot select requested frequency range.")
    return [f_min_arg, f_max_arg]


# ──────────────────────────────────────────────────────────────────────────────
# Parcellation  (osl_dynamics.utils.parcellation)
# ──────────────────────────────────────────────────────────────────────────────
class Parcellation:
    """Read a parcellation NIfTI and expose parcel geometry.

    Parameters
    ----------
    file : str or Parcellation
        Path to parcellation file, or an existing Parcellation object.
    """

    def __init__(self, file):
        if isinstance(file, Parcellation):
            self.__dict__.update(file.__dict__)
            return
        self.file = check_exists(file, parcellation_directory)

        parcellation = nib.load(self.file)

        if parcellation.ndim == 3:
            parcellation_grid = parcellation.get_fdata()
            unique_values = np.unique(parcellation_grid)[1:]
            parcellation_grid = np.array(
                [(parcellation_grid == value).astype(int) for value in unique_values]
            )
            parcellation_grid = np.rollaxis(parcellation_grid, 0, 4)
            parcellation = nib.Nifti1Image(
                parcellation_grid, parcellation.affine, parcellation.header
            )

        self.parcellation = parcellation
        self.dims = self.parcellation.shape[:3]
        self.n_parcels = self.parcellation.shape[3]

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.file)})"

    def data(self):
        return self.parcellation.get_fdata()

    def nonzero(self):
        return [np.nonzero(self.data()[..., i]) for i in range(self.n_parcels)]

    def nonzero_coords(self):
        return [
            nib.affines.apply_affine(self.parcellation.affine, np.array(nz).T)
            for nz in self.nonzero()
        ]

    def weights(self):
        return [self.data()[..., i][nz] for i, nz in enumerate(self.nonzero())]

    def roi_centers(self):
        """Centroid of each parcel in MNI coordinates."""
        return np.array([
            np.average(c, weights=w, axis=0)
            for c, w in zip(self.nonzero_coords(), self.weights())
        ])


def parcel_vector_to_voxel_grid(
    mask_file, parcellation_file, vector, remove_subcortical_voxels=False
):
    """Map a (n_parcels,) vector of parcel values to a 3-D voxel grid.

    Parameters
    ----------
    mask_file : str
        Brain mask NIFTI file.
    parcellation_file : str
        Parcellation NIFTI file.
    vector : np.ndarray
        Shape (n_parcels,).
    remove_subcortical_voxels : bool, optional
        Set subcortical voxels to NaN (only valid for 8 mm grids).

    Returns
    -------
    voxel_grid : np.ndarray
        Shape (x, y, z).
    """
    logging.getLogger("nibabel.global").setLevel(logging.ERROR)

    mask_file = check_exists(mask_file, mask_directory)
    parcellation_file = check_exists(parcellation_file, parcellation_directory)

    mask = nib.load(mask_file)
    mask_grid = mask.get_fdata().ravel(order="F")
    non_zero_voxels = mask_grid != 0

    parcellation = nib.load(parcellation_file)
    parcellation_grid = parcellation.get_fdata()
    if parcellation_grid.ndim == 3:
        unique_values = np.unique(parcellation_grid)[1:]
        parcellation_grid = np.array(
            [(parcellation_grid == value).astype(int) for value in unique_values]
        )
        parcellation_grid = np.rollaxis(parcellation_grid, 0, 4)
        parcellation = nib.Nifti1Image(
            parcellation_grid, parcellation.affine, parcellation.header
        )

    # NB: ``copy_header`` and ``force_resample`` were added in nilearn 0.11;
    # pass them only if the installed version supports them so this works on
    # both the older "general" env (nilearn 0.10.4) and newer envs.
    _kw = {'interpolation': 'nearest'}
    import inspect as _inspect
    _params = _inspect.signature(image.resample_to_img).parameters
    if 'force_resample' in _params:
        _kw['force_resample'] = True
    if 'copy_header' in _params:
        _kw['copy_header'] = True
    parcellation = image.resample_to_img(parcellation, mask, **_kw)
    parcellation_grid = parcellation.get_fdata()
    n_parcels = parcellation.shape[-1]

    if vector.shape[0] != n_parcels:
        _logger.error("parcellation_file has a different number of parcels to the vector")

    voxel_weights = parcellation_grid.reshape(-1, n_parcels, order="F")[non_zero_voxels]
    voxel_weights /= voxel_weights.max(axis=0, keepdims=True)
    voxel_values = voxel_weights @ vector

    voxel_grid = np.zeros(mask_grid.shape[0])
    voxel_grid[non_zero_voxels] = voxel_values
    voxel_grid = voxel_grid.reshape(
        mask.shape[0], mask.shape[1], mask.shape[2], order="F"
    )

    if remove_subcortical_voxels:
        if voxel_grid.shape != (23, 27, 23):
            raise ValueError(
                "remove_subcortical_voxels=True is only compatible with 8x8x8 mm voxel grids."
            )
        for xx in range(10, 13):
            for yy in range(12, 19):
                if yy > 15 or yy < 13:
                    for zz in range(10, 11):
                        if voxel_grid[xx, yy, zz] == 0:
                            voxel_grid[xx, yy, zz] = np.nan
                else:
                    for zz in range(7, 12):
                        if voxel_grid[xx, yy, zz] == 0:
                            voxel_grid[xx, yy, zz] = np.nan
        warnings.filterwarnings("ignore", message="Mean of empty slice")

    return voxel_grid


# ──────────────────────────────────────────────────────────────────────────────
# Plotting utilities  (osl_dynamics.utils.plotting)
# ──────────────────────────────────────────────────────────────────────────────
def plot_line(
    x,
    y,
    labels=None,
    legend_loc=1,
    errors=None,
    x_range=None,
    y_range=None,
    x_label=None,
    y_label=None,
    title=None,
    plot_kwargs=None,
    fig_kwargs=None,
    ax=None,
    filename=None,
):
    """Basic line plot (mirrors osl_dynamics.utils.plotting.plot_line)."""
    if len(x) != len(y):
        raise ValueError("Different number of x and y arrays given.")

    if x_range is None:
        x_range = [None, None]
    if y_range is None:
        y_range = [None, None]

    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]
        elif len(labels) != len(x):
            raise ValueError("Incorrect number of lines or labels passed.")
        add_legend = True
    else:
        labels = [None] * len(x)
        add_legend = False

    if errors is None:
        errors_min = [None] * len(x)
        errors_max = [None] * len(x)
    elif len(errors) != 2:
        raise ValueError("Errors must be [[y_min1,...], [y_max1,...]].")
    elif len(errors[0]) != len(x) or len(errors[1]) != len(x):
        raise ValueError("Incorrect number of errors passed.")
    else:
        errors_min = errors[0]
        errors_max = errors[1]

    if ax is not None:
        if filename is not None:
            raise ValueError("Use fig.savefig() instead of the filename argument.")
        if isinstance(ax, np.ndarray):
            raise ValueError("Only pass one axis.")

    if fig_kwargs is None:
        fig_kwargs = {}
    fig_kwargs = _override_dict_defaults({"figsize": (7, 4)}, fig_kwargs)
    if plot_kwargs is None:
        plot_kwargs = {}

    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(**fig_kwargs)

    for x_data, y_data, label, e_min, e_max in zip(
        x, y, labels, errors_min, errors_max
    ):
        ax.plot(x_data, y_data, label=label, **plot_kwargs)
        if e_min is not None:
            ax.fill_between(x_data, e_min, e_max, alpha=0.3)

    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if add_legend:
        ax.legend(loc=legend_loc)

    if filename is not None:
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
    elif create_fig:
        return fig, ax


def plot_psd_topo(
    f,
    psd,
    only_show=None,
    parcellation_file=None,
    frequency_range=None,
    topomap_pos=None,
    cmap="viridis",
    fig_kwargs=None,
    ax=None,
    filename=None,
):
    """Plot PSDs for parcels with an optional topomap inset."""
    if frequency_range is None:
        frequency_range = [f[0], f[-1]]
    if topomap_pos is None:
        topomap_pos = [0.45, 0.55, 0.5, 0.55]
    if fig_kwargs is None:
        fig_kwargs = {}

    if parcellation_file is not None:
        parc = Parcellation(parcellation_file)
        roi_centers = parc.roi_centers()
        order = np.argsort(roi_centers[:, 1])
        roi_centers = roi_centers[order]
        psd = np.copy(psd)[order]

    n_parcels = psd.shape[0]
    if only_show is None:
        only_show = np.arange(n_parcels)

    ax_passed = ax is not None
    if not ax_passed:
        fig, ax = plt.subplots(**fig_kwargs)

    cmap_obj = plt.get_cmap(cmap + "_r")
    for i in reversed(range(n_parcels)):
        if i in only_show:
            ax.plot(f, psd[i], c=cmap_obj(i / n_parcels))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (a.u.)")
    ax.set_xlim(frequency_range[0], frequency_range[-1])
    plt.tight_layout()

    if parcellation_file is not None:
        inside_ax = ax.inset_axes(topomap_pos)
        plot_markers(
            np.arange(parc.n_parcels),
            roi_centers,
            node_size=12,
            node_cmap=cmap_obj,
            colorbar=False,
            axes=inside_ax,
        )

    if filename is not None and not ax_passed:
        fig.savefig(filename)
        plt.close(fig)
    elif not ax_passed:
        return fig, ax


def plot_brain_surface(
    values,
    mask_file,
    parcellation_file,
    title=None,
    cmap="cold_hot",
    colorbar=True,
    symmetric_cbar=True,
    cbar_tick_format="%.2g",
    cbar_fontsize=24,
    cbar_label=None,
    vmin=None,
    vmax=None,
    hemispheres=None,
    views=None,
    bg_on_data=False,
    threshold=None,
    remove_subcortical_voxels=False,
    filename=None,
    show_plot=None,
):
    """Plot a 2-D heat map on the surface of the brain."""
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)
    if symmetric_cbar:
        vmax = np.max([vmax, -vmin])
        vmin = -vmax
    if hemispheres is None:
        hemispheres = ["left", "right"]
    if views is None:
        views = ["lateral"]

    if filename is not None:
        allowed = [".png", ".svg", ".pdf"]
        if not any(ext in filename for ext in allowed):
            raise ValueError(f"filename must use one of: {' '.join(allowed)}")

    show_plot = filename is None

    mask_file = check_exists(mask_file, mask_directory)
    parcellation_file = check_exists(parcellation_file, parcellation_directory)

    values = parcel_vector_to_voxel_grid(
        mask_file, parcellation_file, values, remove_subcortical_voxels
    )

    mask = nib.load(mask_file)
    nii = nib.Nifti1Image(values, mask.affine, mask.header)

    fig, ax = plot_img_on_surf(
        nii,
        output_file=None,
        colorbar=False,
        cmap=cmap,
        symmetric_cbar=symmetric_cbar,
        vmin=vmin,
        vmax=vmax,
        hemispheres=hemispheres,
        views=views,
        bg_on_data=bg_on_data,
        threshold=threshold,
    )

    if views == ["lateral"]:
        fig.suptitle(title, fontsize=26, y=0.97)
        if colorbar:
            cbar_ax = fig.add_axes([0.25, 0.2, 0.5, 0.05])
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
            )
            cbar = fig.colorbar(
                sm, cax=cbar_ax, orientation="horizontal", format=cbar_tick_format
            )
            cbar.ax.tick_params(labelsize=cbar_fontsize)
            cbar.set_label(cbar_label, fontsize=cbar_fontsize)
    else:
        fig.suptitle(title, fontsize=18, y=0.98)
        if colorbar:
            cbar_ax = fig.add_axes([0.3, 0.1, 0.4, 0.04])
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
            )
            cbar = fig.colorbar(
                sm, cax=cbar_ax, orientation="horizontal", format=cbar_tick_format
            )
            cbar.ax.tick_params(labelsize=16)
            cbar.set_label(cbar_label, fontsize=16)

    if filename is not None:
        fig.savefig(filename)
        if not show_plot:
            plt.close(fig)
    else:
        return fig, ax


# ──────────────────────────────────────────────────────────────────────────────
# Power utilities  (osl_dynamics.analysis.power)
# ──────────────────────────────────────────────────────────────────────────────
def variance_from_spectra(
    frequencies,
    power_spectra,
    components=None,
    frequency_range=None,
    method="mean",
):
    """Calculate variance from power spectra over a frequency band."""
    if power_spectra.ndim == 2:
        power_spectra = power_spectra[np.newaxis, np.newaxis, ...]
        n_sessions, n_modes, n_channels, n_freq = power_spectra.shape

    elif power_spectra.shape[-2] != power_spectra.shape[-3]:
        error_message = (
            "A (n_channels, n_freq), (n_modes, n_channels, n_freq) or "
            "(n_sessions, n_modes, n_channels, n_freq) array must be passed."
        )
        power_spectra = _validate_array(
            power_spectra,
            correct_dimensionality=4,
            allow_dimensions=[2, 3],
            error_message=error_message,
        )
        n_sessions, n_modes, n_channels, n_freq = power_spectra.shape

    else:
        error_message = (
            "A (n_channels, n_channels, n_freq), "
            "(n_modes, n_channels, n_channels, n_freq) or "
            "(n_sessions, n_modes, n_channels, n_channels, n_freq) "
            "array must be passed."
        )
        power_spectra = _validate_array(
            power_spectra,
            correct_dimensionality=5,
            allow_dimensions=[3, 4],
            error_message=error_message,
        )
        n_sessions, n_modes, n_channels, n_channels, n_freq = power_spectra.shape

    if components is not None and frequency_range is not None:
        raise ValueError(
            "Only one of components or frequency_range can be passed."
        )
    if frequency_range is not None and frequencies is None:
        raise ValueError(
            "If frequency_range is passed, frequencies must also be passed."
        )
    if method not in ["mean", "sum", "integral"]:
        raise ValueError("method should be 'mean', 'sum' or 'integral'.")

    n_components = 1 if components is None else components.shape[0]

    var = []
    for i in range(n_sessions):
        if power_spectra.shape[-2] == power_spectra.shape[-3]:
            psd = power_spectra[i, :, range(n_channels), range(n_channels)]
            psd = np.swapaxes(psd, 0, 1)
        else:
            psd = power_spectra[i]
        psd = psd.reshape(-1, n_freq)
        psd = psd.real

        if components is not None:
            p = components @ psd.T
            for j in range(n_components):
                p[j] /= np.sum(components[j])
        else:
            if frequency_range is None:
                if method == "sum":
                    p = np.sum(psd, axis=-1)
                elif method == "integral":
                    df = frequencies[1] - frequencies[0]
                    p = np.sum(psd * df, axis=-1)
                else:
                    p = np.mean(psd, axis=-1)
            else:
                [min_arg, max_arg] = _get_frequency_args_range(frequencies, frequency_range)
                if method == "sum":
                    p = np.sum(psd[..., min_arg:max_arg], axis=-1)
                elif method == "integral":
                    df = frequencies[1] - frequencies[0]
                    p = np.sum(psd[..., min_arg:max_arg] * df, axis=-1)
                else:
                    p = np.mean(psd[..., min_arg:max_arg], axis=-1)

        p = p.reshape(n_components, n_modes, n_channels)
        var.append(p)

    return np.squeeze(var)


def power_save(
    power_map,
    mask_file,
    parcellation_file,
    filename=None,
    component=0,
    subtract_mean=False,
    mean_weights=None,
    plot_kwargs=None,
    combined=False,
    titles=None,
    n_rows=1,
):
    """Save power maps as NIfTI files or brain-surface images.

    Parameters
    ----------
    power_map : np.ndarray
        Shape (n_components, n_modes, n_channels), (n_modes, n_channels) or
        (n_channels,).
    mask_file : str
        Brain mask file.
    parcellation_file : str
        Parcellation file.
    filename : str, optional
        Output path.  Extension determines format: .nii/.nii.gz for NIfTI,
        .png/.svg/.pdf for images.  None returns figure objects.
    """
    if plot_kwargs is None:
        plot_kwargs = {}

    if filename is not None:
        allowed = [".nii", ".nii.gz", ".png", ".svg", ".pdf"]
        if not any(ext in filename for ext in allowed):
            raise ValueError(f"filename must use one of: {' '.join(allowed)}")

    mask_file = check_exists(mask_file, mask_directory)
    parcellation_file = check_exists(parcellation_file, parcellation_directory)

    power_map = np.copy(power_map)
    power_map = np.squeeze(power_map)
    if power_map.ndim > 1:
        if power_map.shape[-1] == power_map.shape[-2]:
            power_map = np.copy(np.diagonal(power_map, axis1=-2, axis2=-1))
            if power_map.ndim == 1:
                power_map = power_map[np.newaxis, ...]
    else:
        power_map = power_map[np.newaxis, ...]

    power_map = _validate_array(
        power_map,
        correct_dimensionality=3,
        allow_dimensions=[2],
        error_message="power_map.shape is incorrect",
    )

    n_modes = power_map.shape[1]
    if n_modes == 1:
        subtract_mean = False
    if subtract_mean:
        power_map -= np.average(power_map, axis=1, weights=mean_weights)[
            :, np.newaxis, ...
        ]

    power_map = power_map[component]

    if titles is None:
        titles = [None] * n_modes
    elif len(titles) != n_modes:
        raise ValueError(
            f"Number of titles ({len(titles)}) does not match the number "
            f"of power maps ({n_modes})."
        )

    if filename is None:
        figures, axes = [], []
        for i in trange(n_modes, desc="Saving images"):
            fig, ax = plot_brain_surface(
                power_map[i],
                mask_file=mask_file,
                parcellation_file=parcellation_file,
                title=titles[i],
                **plot_kwargs,
            )
            figures.append(fig)
            axes.append(ax)
        return figures, axes

    else:
        if ".nii" in filename:
            power_map_vox = [
                parcel_vector_to_voxel_grid(mask_file, parcellation_file, p)
                for p in power_map
            ]
            power_map_vox = np.moveaxis(power_map_vox, 0, -1)
            _logger.info(f"Saving {filename}")
            mask = nib.load(mask_file)
            nii = nib.Nifti1Image(power_map_vox, mask.affine, mask.header)
            nib.save(nii, filename)

        else:
            output_files = []
            for i in trange(n_modes, desc="Saving images"):
                output_file = "{fn.parent}/{fn.stem}{i:0{w}d}{fn.suffix}".format(
                    fn=Path(filename), i=i, w=len(str(n_modes))
                )
                plot_brain_surface(
                    power_map[i],
                    mask_file=mask_file,
                    parcellation_file=parcellation_file,
                    title=titles[i],
                    filename=output_file,
                    **plot_kwargs,
                )
                output_files.append(output_file)

            if combined:
                n_columns = -(n_modes // -n_rows)
                fig, axes = plt.subplots(
                    n_rows, n_columns, figsize=(n_columns * 5, n_rows * 5)
                )
                for i, ax in enumerate(axes.flatten()):
                    ax.axis("off")
                    if i < n_modes:
                        ax.imshow(plt.imread(output_files[i]))
                fig.tight_layout()
                fig.savefig(filename)
                for output_file in output_files:
                    os.remove(output_file)


# ──────────────────────────────────────────────────────────────────────────────
# Connectivity utilities  (osl_dynamics.analysis.connectivity)
# ──────────────────────────────────────────────────────────────────────────────
def connectivity_save(
    connectivity_map,
    parcellation_file,
    filename=None,
    component=None,
    threshold=0,
    plot_kwargs=None,
    axes=None,
    combined=False,
    titles=None,
    n_rows=1,
):
    """Save connectivity maps as image files (wraps nilearn.plot_connectome)."""
    logging.getLogger("nibabel.global").setLevel(logging.ERROR)

    connectivity_map = np.copy(connectivity_map)
    error_message = (
        "Dimensionality of connectivity_map must be 3 or 4, "
        f"got ndim={connectivity_map.ndim}."
    )
    connectivity_map = _validate_array(
        connectivity_map,
        correct_dimensionality=4,
        allow_dimensions=[2, 3],
        error_message=error_message,
    )

    if isinstance(threshold, (float, int)):
        threshold = np.array([threshold] * connectivity_map.shape[1])

    if np.any(threshold > 1) or np.any(threshold < 0):
        raise ValueError("threshold must be between 0 and 1.")

    if component is None:
        component = 0

    parcellation = Parcellation(parcellation_file)
    conn_map = connectivity_map[component]

    for c in conn_map:
        np.fill_diagonal(c, 0)

    default_plot_kwargs = {"node_size": 10, "node_color": "black"}
    n_modes = conn_map.shape[0]
    axes = axes or [None] * n_modes
    output_files = []

    for i in trange(n_modes, desc="Saving images"):
        kwargs = _override_dict_defaults(default_plot_kwargs, plot_kwargs)

        output_file = (
            None
            if filename is None
            else "{fn.parent}/{fn.stem}{i:0{w}d}{fn.suffix}".format(
                fn=Path(filename), i=i, w=len(str(n_modes))
            )
        )

        kwargs["colorbar"] = np.any(
            conn_map[i][~np.eye(conn_map[i].shape[-1], dtype=bool)] != 0
        )

        plot_connectome(
            conn_map[i],
            parcellation.roi_centers(),
            edge_threshold=f"{threshold[i] * 100}%",
            output_file=output_file,
            axes=axes[i],
            **kwargs,
        )
        output_files.append(output_file)

    if combined:
        if filename is None:
            raise ValueError("filename must be passed to save the combined image.")
        n_columns = -(n_modes // -n_rows)
        titles = titles or [None] * n_modes
        fig, axes = plt.subplots(
            n_rows, n_columns, figsize=(n_columns * 5, n_rows * 5)
        )
        for i, ax in enumerate(axes.flatten()):
            ax.axis("off")
            if i < n_modes:
                ax.imshow(plt.imread(output_files[i]))
                ax.set_title(titles[i], fontsize=20)
        fig.tight_layout()
        fig.savefig(filename)
        for output_file in output_files:
            os.remove(output_file)
