# --- Core: always available ---
from .helpers import (
    mean_psd_in_band,
    pearson_corr,
    correct_trigger,
    psd_plot,
    temp_plot,
    temp_plot_diff,
    mne_epoch2raw,
    pcs_plot,
    EEGDictMeta,
    SingletonEEG,
)
from .metric import EEGTracer, psd_band_ratio, psd_band_stat

# extension.py is reserved for future use; not imported here.

# --- OSL-Ephys dependent: preprocessing wrappers ---
from .wrappers import (
    voltage_correction,
    cleanup,
    mid_crop,
    set_channel_type_raw,
    init_tracer,
    summary,
    ckpt_report,
    crop_TR,
    crop_by_epoch,
    create_epoch,
    epoch_ssp,
    epoch_aas,
    epoch_obs,
    slice_ica,
    apply_ica,
    start_timer,
    end_timer,
)

# --- Auto-discovered registry of semp wrappers for run_proc_batch ---
# osl-ephys's find_func walks extra_funcs by name, so any (dataset, userargs)
# function passed there can be referenced from the YAML/dict config without an
# explicit import in the user script. We collect every public callable from
# semp.preprocessing.wrappers.* (skipping private _names) so callers using the
# semp.run_proc_batch / run_proc_chain shims don't have to spell each wrapper
# out by hand.
def _collect_semp_wrappers():
    import inspect
    funcs = []
    seen = set()
    try:
        from . import wrappers as _w
    except ImportError:
        return funcs
    for name in dir(_w):
        if name.startswith('_'):
            continue
        obj = getattr(_w, name)
        if inspect.isfunction(obj) and obj.__name__ not in seen:
            seen.add(obj.__name__)
            funcs.append(obj)
    return funcs


SEMP_EXTRA_FUNCS = _collect_semp_wrappers()
