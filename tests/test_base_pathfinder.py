# tests/test_base_pathfinder.py
import os, sys
import pytest
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'semp', 'utils')))
from pathfinder import BasePathfinder  # <<-- REPLACE this with the real import


class TmpPathfinder(BasePathfinder):
    """
    Minimal test Pathfinder.

    dict2id: canonical id is subject+session+run (concatenated strings).
    id2dict: reverse that simple encoding (subject=first, session=second, run=third char).
    This keeps subject/session/run as single characters in tests so parsing is simple.
    """
    def dict2id(self, fields):
        # fields will be like {'subject': '1', 'session': '1', 'run': '1', ...}
        subj = fields.get('subject') or "0"
        ses = fields.get('session') or "0"
        run = fields.get('run') or "0"
        return f"{subj}{ses}{run}"

    def id2dict(self, file_id: str):
        if len(file_id) != 3:
            raise ValueError("test id2dict expects 3-char ids")
        return {"subject": file_id[0], "session": file_id[1], "run": file_id[2]}


def write_file(path: Path):
    """Utility: create parent dirs and touch file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")  # small file


def test_build_paths_from_anchor(tmp_path):
    """
    Anchor 'raw' should be used to build canonical file_ids and then other kinds resolved.
    """
    # patterns: include subject/session/run as single-digit tokens
    raw_pattern = str(tmp_path / "{subject}" / "{session}" / "{subject}_{session}_{run}_raw.fif")
    preproc_pattern = str(tmp_path / "{subject}" / "{session}" / "{subject}_{session}_{run}_preproc.fif")

    # create files for two subjects
    files = []
    for subject in ("1", "2"):
        f_raw = Path(raw_pattern.format(subject=subject, session="1", run="1"))
        f_pre = Path(preproc_pattern.format(subject=subject, session="1", run="1"))
        write_file(f_raw)
        # create preproc only for subject '1' to test missing-kind handling
        if subject == "1":
            write_file(f_pre)
        files.append((subject, f_raw, f_pre))

    pf = TmpPathfinder(file_patterns={"raw": raw_pattern, "preproc": preproc_pattern}, anchor="raw")

    # expected file ids: '111' and '211'
    ids = set(pf.get_file_ids())
    assert ids == {"111", "211"}

    # check raw paths present and correct
    p111 = pf["111"]["raw"]
    assert Path(p111).exists()
    assert Path(p111) == files[0][1]  # subject '1'

    p211 = pf["211"]["raw"]
    assert Path(p211).exists()
    assert Path(p211) == files[1][1]  # subject '2'

    # preproc exists for 111 but not for 211
    assert pf.get("111").get("preproc") == files[0][2]
    assert pf.get("211").get("preproc") is None


def test_anchor_missing_raises(tmp_path):
    """Providing an anchor name that is not a key of file_patterns should raise KeyError."""
    raw_pattern = str(tmp_path / "{subject}" / "{session}" / "{subject}_{session}_{run}_raw.fif")
    with pytest.raises(KeyError):
        TmpPathfinder(file_patterns={"raw": raw_pattern}, anchor="does_not_exist")


def test_duplicate_file_id_raises(tmp_path):
    """
    If dict2id collapses different files into the same canonical id, _glob_pattern should detect
    duplicate file_id and raise ValueError during refresh().
    We arrange two files with the same subject but different session so that a custom dict2id
    that uses only subject will cause duplicate.
    """
    # anchor pattern includes subject and session (so two files exist),
    # but we will use a small helper subclass that collapses id to subject only.
    class CollapsingPF(TmpPathfinder):
        def dict2id(self, fields):
            # use only subject -> two files with same subject but different session collapse
            subj = fields.get("subject") or "0"
            return subj  # single-char id

        def id2dict(self, file_id: str):
            # not needed for this test, but implement minimal
            return {"subject": file_id, "session": None, "run": None}

    raw_pattern = str(tmp_path / "{subject}" / "{session}" / "{subject}_{session}_{run}_raw.fif")

    # create two files with same subject '1' but different session '1' and '2'
    p1 = Path(raw_pattern.format(subject="1", session="1", run="1"))
    p2 = Path(raw_pattern.format(subject="1", session="2", run="1"))
    write_file(p1)
    write_file(p2)

    with pytest.raises(ValueError):
        CollapsingPF(file_patterns={"raw": raw_pattern}, anchor="raw")


def test_get_returns_empty_dict_for_missing(tmp_path):
    """pf.get(missing) should return an empty dict (fresh instance) and not None."""
    raw_pattern = str(tmp_path / "{subject}" / "{session}" / "{subject}_{session}_{run}_raw.fif")
    write_file(Path(raw_pattern.format(subject="1", session="1", run="1")))
    pf = TmpPathfinder(file_patterns={"raw": raw_pattern}, anchor="raw")

    missing = pf.get("999")
    assert isinstance(missing, dict)
    assert missing == {}
    # ensure it's a fresh dict: mutation does not affect future calls
    missing["x"] = 1
    assert pf.get("999") == {}