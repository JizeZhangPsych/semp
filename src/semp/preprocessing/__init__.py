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
try:
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
        manual_ica,
        apply_ica,
        _TimerRegistry,
        start_timer,
        end_timer,
    )
except ImportError:
    pass
