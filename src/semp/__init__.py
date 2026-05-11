"""semp v2.0: Simultaneous EEG-fMRI Preprocessing Toolbox

Available functionality depends on the installed environment:
  - Core (always): semp.utils, semp.preprocessing (helpers/metrics/extensions),
                   semp.visualize (all — osl_dynamics no longer required)
  - OSL-Ephys env: semp.preprocessing (prep wrappers), semp.source_recon,
                   semp.utils.osle_expansion

osl-ephys is the primary optional dependency.  osl-dynamics is no longer a
dependency — its needed routines are bundled in semp.utils.osld_extension.
"""

import warnings
import importlib.util

# --- Detect installed optional environments ---
HAS_OSLE = importlib.util.find_spec("osl_ephys") is not None
HAS_OSLD = importlib.util.find_spec("osl_dynamics") is not None  # kept for info only

# --- Always import subpackages (each handles its own missing deps internally) ---
from semp import utils, preprocessing, source_recon, visualize

__all__ = ["utils", "preprocessing", "source_recon", "visualize", "HAS_OSLE", "HAS_OSLD"]

# --- Warn about missing environments ---
if not HAS_OSLE:
    warnings.warn(
        "osl-ephys not found. EEG preprocessing/wrappers, source_recon, and "
        "osle_expansion are unavailable. "
        "Install from: https://github.com/OHBA-analysis/osl-ephys",
        UserWarning,
        stacklevel=2,
    )

# --- Mode summary ---
_mode = "osl-ephys + core" if HAS_OSLE else "core only"
print(f"semp v2.0 loaded [{_mode}]")
