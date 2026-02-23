"""semp v2.0: Simultaneous EEG-fMRI Preprocessing Toolbox

Available functionality depends on the installed environment:
  - Core (always): semp.utils, semp.eeg (helpers/metrics/extensions),
                   semp.visualize (array_ops, statistics)
  - OSL-Ephys env: semp.eeg (prep/src wrappers), semp.utils.osle_expansion
  - OSL-Dynamics env: semp.visualize.visualize (power maps, connectivity, PSD)

osl-ephys and osl-dynamics are not designed to coexist in the same conda
environment. Install semp into whichever environment you need.
"""

import warnings
import importlib.util

# --- Detect installed optional environments ---
HAS_OSLE = importlib.util.find_spec("osl_ephys") is not None
HAS_OSLD = importlib.util.find_spec("osl_dynamics") is not None

# --- Always import subpackages (each handles its own missing deps internally) ---
from semp import utils, eeg, visualize

__all__ = ["utils", "eeg", "visualize", "HAS_OSLE", "HAS_OSLD"]

# --- Warn about missing environments ---
if not HAS_OSLE:
    warnings.warn(
        "osl-ephys not found. EEG preprocessing/wrappers and "
        "osle_expansion are unavailable. "
        "Install from: https://github.com/OHBA-analysis/osl-ephys",
        UserWarning,
        stacklevel=2,
    )

if not HAS_OSLD:
    warnings.warn(
        "osl-dynamics not found. semp.visualize is unavailable. "
        "Install from: https://github.com/OHBA-analysis/osl-dynamics",
        UserWarning,
        stacklevel=2,
    )

# --- Mode summary ---
if HAS_OSLE and HAS_OSLD:
    _mode = "full (osl-ephys + osl-dynamics)"
elif HAS_OSLE:
    _mode = "osl-ephys only"
elif HAS_OSLD:
    _mode = "osl-dynamics only"
else:
    _mode = "core only"

print(f"semp v2.0 loaded [{_mode}]")