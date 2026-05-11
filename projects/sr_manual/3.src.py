#%%
"""Source reconstruction for the sr_manual pipeline.

Inputs are the *after*-ICA fifs written by ``2.ica.py``
(``<subject>_afterica-raw.fif``). All file lookups go through the
sr_manual ``StaresinaRestPathfinder`` — no ``Study`` / ``HeteroStudy``.
"""
import os
import re
import glob
from pathlib import Path

import numpy as np
from osl_ephys import source_recon, utils as osl_utils

from pathfinder import StaresinaRestPathfinder


continue_interrupt = True
skip_error = False

# ── output dir for source recon (matches the 'src' pattern in pathfinder.py) ─
RECON_DIR = Path(
    '/ohba/pi/mwoolrich/datasets/oxford/staresina/eeg_fmri/after_src_sr_manual'
)

pf = StaresinaRestPathfinder()

# sMRI: typically 2 files per run (raw + normalised). PF disallows multiple
# matches per key, so we glob manually and pick the latest (assumed to be the
# normalised one --- matches the original sr/2.src.py convention of
# ``sorted(smri)[-1]``).
_SMRI_GLOB = (
    '/ohba/pi/mwoolrich/raw_datasets/oxford/staresina/eeg_fmri/'
    'sub-{subject:0>3}/ses-0{session}/mri/run-0{run}/*/*t1*.nii'
)


def find_smri(file_id):
    fields = pf.id2dict(file_id)
    return sorted(glob.glob(_SMRI_GLOB.format(**fields)))


# ── pipeline custom step: convert .pom polhemus to RHINO headshape format ──
def polhemus_translation(outdir, subject):
    unused_ch = ['F11', 'F12', 'FT11', 'FT12', 'Cb1', 'Cb2']
    rhino_pth_dict = source_recon.rhino.get_coreg_filenames(outdir, subject)

    polhemus_path = pf[subject]['polhemus']
    with open(polhemus_path, 'r') as f:
        polhemus_content = f.read()
    polhemus_content = re.sub(r"#.*?\n", "\n", polhemus_content)

    loc_match = re.search(
        r"LOCATION_LIST START_LIST([\s\S]*?)LOCATION_LIST END_LIST",
        polhemus_content,
    )
    assert loc_match is not None
    locations = np.asarray(
        [line.split() for line in loc_match.group(1).strip().splitlines()],
        dtype=np.float32,
    )

    rem_match = re.search(
        r"REMARK_LIST START_LIST([\s\S]*?)REMARK_LIST END_LIST",
        polhemus_content,
    )
    assert rem_match is not None
    remarks = rem_match.group(1).strip().splitlines()

    # axis sign flips so the polhemus cloud comes out in RHINO's convention
    sign_x = (locations[remarks.index('C6')][0]
              > locations[remarks.index('C5')][0]) * 2 - 1
    sign_y = (locations[remarks.index('Fpz')][1]
              > locations[remarks.index('Oz')][1]) * 2 - 1
    sign_z = (locations[remarks.index('Cz')][2]
              > np.mean(locations[:, 2])) * 2 - 1
    locations = [[sign_x * x, sign_y * y, sign_z * z]
                 for x, y, z in locations]

    if (locations[remarks.index('Left ear')][0]
        > locations[remarks.index('Right ear')][0]):
        locations[remarks.index('Left ear')][0]  *= -1
        locations[remarks.index('Right ear')][0] *= -1
        locations[remarks.index('Nasion')][0]    *= -1
    if (locations[remarks.index('Nasion')][1]
        < (locations[remarks.index('Left ear')][1]
           + locations[remarks.index('Right ear')][1]) / 2):
        locations[remarks.index('Nasion')][1]    *= -1
        locations[remarks.index('Left ear')][1]  *= -1
        locations[remarks.index('Right ear')][1] *= -1

    headshape_coords = []
    for idx, remark in enumerate(remarks):
        if remark in unused_ch:
            continue
        if remark == 'Left ear':
            with open(rhino_pth_dict['polhemus_lpa_file'], 'w') as f:
                f.write(f"{locations[idx][0]}\n{locations[idx][1]}\n{locations[idx][2]}\n")
        elif remark == 'Right ear':
            with open(rhino_pth_dict['polhemus_rpa_file'], 'w') as f:
                f.write(f"{locations[idx][0]}\n{locations[idx][1]}\n{locations[idx][2]}\n")
        elif remark == 'Nasion':
            with open(rhino_pth_dict['polhemus_nasion_file'], 'w') as f:
                f.write(f"{locations[idx][0]}\n{locations[idx][1]}\n{locations[idx][2]}\n")
        else:
            headshape_coords.append(locations[idx])

    headshape_coords = np.array(headshape_coords).T
    with open(rhino_pth_dict['polhemus_headshape_file'], 'w') as f:
        for row in headshape_coords:
            row = [f"{coord:.6f}" for coord in row]
            f.write(' '.join(row) + '\n')


config = """
    source_recon:
        - polhemus_translation: {}
        - compute_surfaces:
            include_nose: true
        - coregister:
            use_nose: true
            use_headshape: true
            allow_smri_scaling: true
        - forward_model:
            model: Triple Layer
            eeg: true
        - beamform_and_parcellate:
            freq_range: [1, 45]
            chantypes: eeg
            rank: {eeg: 45}
            parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
            method: spatial_basis
            orthogonalisation: symmetric
            reg: 0.05
"""


#%%
if __name__ == "__main__":
    # NB: subjects without a usable polhemus file are auto-dropped by the
    # pathfinder (`polhemus` is in StaresinaRestPathfinder.REQUIRED_KEYS),
    # so the historical `no_polhemus_list` (17111, 17112, 1121, 2121) is
    # redundant --- those file_ids never appear in `pf` to begin with.

    osl_utils.logger.set_up(level="INFO")
    source_recon.setup_fsl(
        os.environ.get('FSLDIR', '/opt/ohba/software/software/fsl/6.0.7.9')
    )

    subject_list, afterica_files, smri_files = [], [], []
    for file_id in sorted(pf):
        paths = pf[file_id]
        if 'afterica' not in paths:
            print(f"WARNING: {file_id} has no after-ICA fif (run 2.ica.py?), skipping")
            continue
        if 'polhemus' not in paths:
            print(f"WARNING: {file_id} has no polhemus on disk, skipping")
            continue

        smri = find_smri(file_id)
        if len(smri) not in (1, 2):
            print(f"WARNING: {file_id} has {len(smri)} smri files, skipping")
            continue

        smri_files.append(smri[-1])             # normed assumed second
        afterica_files.append(str(paths['afterica']))
        subject_list.append(file_id)

    if continue_interrupt:
        finished = {p.split('/')[-3] for p in
                    glob.glob(f'{RECON_DIR}/*/parc/lcmv-parc-raw.fif')}
        errored  = {os.path.basename(p).split('_')[0] for p in
                    glob.glob(f'{RECON_DIR}/logs/*.error.log')}

        kept_s, kept_a, kept_m = [], [], []
        for s, a, m in zip(subject_list, afterica_files, smri_files):
            if s in finished:
                print(f"WARNING: {s} already finished, skipping")
            elif skip_error and s in errored:
                print(f"WARNING: {s} previously errored, skipping")
            else:
                kept_s.append(s); kept_a.append(a); kept_m.append(m)
        subject_list, afterica_files, smri_files = kept_s, kept_a, kept_m

    print(f"[3.src] running source recon on {len(subject_list)} subject(s)")
    source_recon.run_src_batch(
        config,
        outdir=str(RECON_DIR),
        subjects=subject_list,
        preproc_files=afterica_files,
        smri_files=smri_files,
        extra_funcs=[polhemus_translation],
        gen_report=False,   # osl-ephys src_report.gen_html_data crashes on
                            # parcel-only raws (plot_freqbands skips writing
                            # freqbands.png because parcel misc-chans have no
                            # montage, then gen_html_data tries to copy it)
        # dask_client=True,
    )

# %%
