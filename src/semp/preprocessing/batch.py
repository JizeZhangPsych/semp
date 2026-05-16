"""Thin shims around osl-ephys batch runners that auto-inject semp wrappers.

osl-ephys's `find_func` looks up a stage name in this order:
    1. `extra_funcs` (by `__name__`)
    2. `osl_wrappers.run_osl_<name>`
    3. `mne_wrappers.run_mne_<name>`
    4. MNE method on Raw / Epochs

semp wrappers (`crop_TR`, `epoch_aas`, `manual_ica`, ...) live in none of
those layers, so users have to spell each one out in `extra_funcs=[...]`.
Forgetting one yields the cryptic `Function not found! <name>` followed
by a `'NoneType' object is not callable`.

These shims pre-pend `SEMP_EXTRA_FUNCS` (auto-collected from
`semp.preprocessing.wrappers`) so any semp wrapper just works in a config.
User-supplied `extra_funcs=` are appended *after* the semp ones, which
means user functions take priority on name collisions (find_func uses
`np.argmax` on the index list, so the later entry wins).

Usage::

    from semp import run_proc_batch       # instead of osl_ephys's
    run_proc_batch(config, files, ..., extra_funcs=[my_initialize])
"""
from osl_ephys.preprocessing import (
    run_proc_batch as _osle_run_proc_batch,
    run_proc_chain as _osle_run_proc_chain,
)

from . import SEMP_EXTRA_FUNCS


def _merge_extra_funcs(user_extra):
    user_extra = list(user_extra) if user_extra else []
    user_names = {f.__name__ for f in user_extra}
    # Only add semp wrappers the user didn't already pass (avoids duplicate
    # entries; the user's copy takes precedence anyway, but a clean list
    # keeps the in-stage logging tidy).
    semp_only = [f for f in SEMP_EXTRA_FUNCS if f.__name__ not in user_names]
    return semp_only + user_extra


def run_proc_batch(*args, extra_funcs=None, **kwargs):
    return _osle_run_proc_batch(*args, extra_funcs=_merge_extra_funcs(extra_funcs), **kwargs)


def run_proc_chain(*args, extra_funcs=None, **kwargs):
    return _osle_run_proc_chain(*args, extra_funcs=_merge_extra_funcs(extra_funcs), **kwargs)


run_proc_batch.__doc__ = (
    "semp shim around osl_ephys.preprocessing.run_proc_batch: pre-injects "
    "every public semp wrapper into extra_funcs so configs can reference "
    "them by name without an explicit import.\n\n"
    + (_osle_run_proc_batch.__doc__ or "")
)
run_proc_chain.__doc__ = (
    "semp shim around osl_ephys.preprocessing.run_proc_chain: pre-injects "
    "every public semp wrapper into extra_funcs so configs can reference "
    "them by name without an explicit import.\n\n"
    + (_osle_run_proc_chain.__doc__ or "")
)
