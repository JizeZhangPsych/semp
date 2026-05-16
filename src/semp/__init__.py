"""semp v2.0: Simultaneous EEG-fMRI Preprocessing Toolbox

Available functionality depends on the installed environment:
  - Core (always): semp.utils, semp.preprocessing (helpers/metrics/extensions),
                   semp.visualize
  - OSL-Ephys env: semp.preprocessing (prep wrappers), semp.source_recon

osl-ephys is the primary dependency. The bundled NIfTI mask / parcellation
files used by ``semp.utils.parcel_plot`` and ``semp.visualize`` come from
osl-ephys's ``source_recon/files/`` directory, so installing osl-ephys is
sufficient --- no separate atlas download is required.
"""

import warnings
import importlib.util

# --- Detect installed optional environments ---
HAS_OSLE = importlib.util.find_spec("osl_ephys") is not None

# --- Always import subpackages (each handles its own missing deps internally) ---
from semp import utils, preprocessing, source_recon, visualize

# Top-level shims around osl-ephys batch runners that auto-inject every semp
# wrapper into extra_funcs (so configs can reference `crop_TR`, `epoch_aas`,
# `manual_ica`, ... without a per-script import list).
if HAS_OSLE:
    from semp.preprocessing.batch import run_proc_batch, run_proc_chain
    __all__ = ["utils", "preprocessing", "source_recon", "visualize", "HAS_OSLE",
               "run_proc_batch", "run_proc_chain"]
else:
    __all__ = ["utils", "preprocessing", "source_recon", "visualize", "HAS_OSLE"]

# --- Warn about missing environments ---
if not HAS_OSLE:
    warnings.warn(
        "osl-ephys not found. EEG preprocessing/wrappers, source_recon, and "
        "the bundled NIfTI atlas files are unavailable. "
        "Install from: https://github.com/OHBA-analysis/osl-ephys",
        UserWarning,
        stacklevel=2,
    )

# --- Mode summary ---
_mode = "osl-ephys + core" if HAS_OSLE else "core only"
print(f"semp v2.0 loaded [{_mode}]")
