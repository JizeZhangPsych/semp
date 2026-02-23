import re
from pathlib import Path
from typing import Dict, Optional

from semp.utils import BasePathfinder

class StaresinaRestPathfinder(BasePathfinder):
    """
    Concrete Pathfinder for the Staresina resting EEG-fMRI dataset.

    File ID convention: stripped subject (no leading zeros) + single-digit session
    + single-digit run + single-digit block.

    Examples
    --------
    sub-001 ses-02 run-01 block-03  →  file_id "1213"
    sub-011 ses-01 run-02 block-03  →  file_id "11123"

    Reverse (id2dict) returns subject zero-padded to 2 digits (to match sub-0{subject}
    in path patterns) and session/run/block as single digits (ses-0{session} etc.):

    "1213"   →  subject="01", session="2", run="1", block="3"
    "11213"  →  subject="11", session="2", run="1", block="3"
    """
    
    DEFAULT_FILE_PATTERNS = {
        'rest': "/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/edfs/sub-0{subject}_ses-0{session}_run-0{run}_block-0{block}_task-resting_convert.cdt.edf",
        'dpo': "/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/sub-0{subject}/ses-0{session}/eeg/sub-0{subject}_ses-0{session}_run-0{run}_block-0{block}_task-resting.cdt.dpo",
        'polhemus': "/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/sub-0{subject}/ses-0{session}/polhemus/sub-0{subject}_ses-0{session}_run-0{run}_{foo}.pom",
        'preproc': "/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/after_prep_sts/{subject}{session}{run}{block}/{subject}{session}{run}{block}_preproc-raw.fif",
        'src': "/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/after_src_sts/{subject}{session}{run}{block}/parc/lcmv-parc-raw.fif",
    }
    DEFAULT_ANCHOR = "rest"
    REQUIRED_KEYS = {"rest", "dpo", "polhemus"}

    def __init__(self, file_patterns: Optional[Dict[str, str]] = None, anchor: Optional[str] = None):
        patterns = file_patterns if file_patterns is not None else self.DEFAULT_FILE_PATTERNS
        chosen_anchor = anchor if anchor is not None else self.DEFAULT_ANCHOR
        # pass anchor explicitly to BasePathfinder so it can validate it
        super().__init__(file_patterns=patterns, anchor=chosen_anchor, required=self.REQUIRED_KEYS)

    def dict2id(self, fields: Dict[str, Optional[str]]) -> str:
        """Convert fields dict to canonical file_id.

        subject and session are stripped of leading zeros (since they are
        already stored without the fixed '0' prefix from the path pattern).
        run and block are used as single digits.

        Examples
        --------
        {'subject': '01', 'session': '2', 'run': '1', 'block': '3'}  →  "1213"
        {'subject': '11', 'session': '1', 'run': '2', 'block': '3'} →  "11123"
        """
        subj = fields.get('subject')
        ses = fields.get('session', "1")
        run = fields.get('run', "1")
        block = fields.get('block', "1")

        subj_num = subj.lstrip("0") if subj else "1"

        return f"{subj_num}{ses}{run}{block}"

    def id2dict(self, file_id: str) -> Dict[str, str]:
        """Convert canonical file_id back to a fields dict.

        The last three characters are always single-digit session, run, and block.
        Everything before is the subject number. subject is
        zero-padded to 2 digits to match the path patterns (sub-0{subject}).

        Examples
        --------
        "1213"   →  {'subject': '01', 'session': '2', 'run': '1', 'block': '3'}
        "11213"  →  {'subject': '11', 'session': '2', 'run': '1', 'block': '3'}
        """
        if len(file_id) < 4:
            raise ValueError(f"Invalid Staresina file_id: '{file_id}' (must be at least 4 digits)")

        subj_num = file_id[:-3]
        ses_num  = file_id[-3]
        run_num  = file_id[-2]
        block_num = file_id[-1]

        return {
            "subject": subj_num.zfill(2),
            "session": ses_num,
            "run":     run_num,
            "block":   block_num,
        }
