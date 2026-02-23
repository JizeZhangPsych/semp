from __future__ import annotations
import glob
from pathlib import Path
from string import Formatter
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import KeysView, Iterator, Iterable
from dataclasses import dataclass, field
from typing import Dict, Optional, Set
import parse
from .logger import log_or_print

@dataclass(frozen=True)
class BasePathfinder(ABC, Mapping):
    """
    Immutable double-level dictionary of file_id + kind → Path,
    with field-based parsing and reverse lookup.

    Placeholders starting with 'foo' are reserved and ignored
    for file_id computation. Only the remaining core fields
    are used in dict2id / id2dict.

    Attributes
    ----------
    file_patterns : Dict[str, str]
        Mapping from file kinds (e.g., 'raw', 'preproc') to filename patterns.
        Patterns can contain `{field}` placeholders, e.g.
        '/path/to/your/dataset/{subject}/{session}/{subject}_{task}_raw.fif'.
    fieldnames : Set[str]
        Set of all core fieldnames found in patterns (excluding foo* placeholders).
    paths : Dict[str, Dict[str, Path]]
        Double-level mapping of file_id → kind → Path.
    
    Usage
    -----
    1. Define a subclass implementing dict2id / id2dict:
    
        class EEGPathfinder(BasePathfinder):
            def dict2id(self, fields):
                # Example: 'subject_session_task' → '010102'
                # run defaults to '01' if is None. this could happens if some files are shared across runs.
                # if run-01 is not always present in your dataset, you can also iterate over all possible runs to find an available one.
                
                runs = fields['run'] if fields.get('run') is not None else '01'
                return f"{fields['subject']}{fields['session']}{runs}"
                
            def id2dict(self, file_id):
                # Reverse mapping from '010102' → dict
                return {'subject': file_id[:2], 'session': file_id[2:4], 'run': file_id[4:]}

    2. Instantiate with file patterns:

        file_patterns = {
            'raw': '/path/to/your/dataset/{subject}/{session}/{subject}_{run}_{foo1}raw{foo2}.fif',
            'preproc': '/path/to/your/dataset/{subject}/{session}/{subject}_{run}_preproc.fif',
            'polhemus': '/path/to/your/dataset/{subject}/{session}/{subject}_polhemus.txt', 
        }
        pf = DatasetPathfinder(file_patterns)
        
        This example assumes different runs shares the same polhemus file, so with both 010101 and 010102,
        pf['010101']['polhemus'] and pf['010102']['polhemus'] will point to the same file.
        
        Any placeholders starting with 'foo' are ignored and not included in fieldnames or file_id.

    3. Access paths by file_id:

        file_id = '010101'  # subject 01, session 01, run 01
        raw_path = pf[file_id]['raw']
        preproc_path = pf[file_id]['preproc']

    4. Refresh from disk if paths change:

        pf.refresh()

    Notes
    -----
    - The class automatically parses filenames based on `file_patterns` and
      builds the `paths` dictionary upon initialization.
    - Missing fields in filenames (e.g., files shared across runs) are set to None.
    - Duplicate file_ids will raise a ValueError.
    # - Use `pf.keys()` to iterate over all file_ids.
    """

    file_patterns: Dict[str, str]
    # anchor is a single kind name (str) used as the canonical anchor, should be unambiguous, and the pattern must contain all placeholders.
    anchor: str = "raw"
    ignored_prefix: str = "foo"
    # Kinds listed here are required: if a file_id has no match on disk for a required kind,
    # it is removed from paths and a warning is issued. Non-listed kinds are skipped silently.
    required: tuple[str, ...] = ()
    fieldnames: Set[str] = field(init=False)
    paths: Dict[str, Dict[str, Path]] = field(init=False)

    def __post_init__(self):
        # Extract all placeholders from patterns
        all_placeholders: Set[str] = set()
        for pattern in self.file_patterns.values():
            all_placeholders.update(
                fname for _, fname, _, _ in Formatter().parse(pattern) if fname
            )

        # Ignore any placeholder starting with 'foo'
        core_fields = {f for f in all_placeholders if not f.startswith(self.ignored_prefix)}
        object.__setattr__(self, "fieldnames", core_fields)
        
        if self.anchor not in self.file_patterns:
            raise KeyError(f"anchor {self.anchor!r} not found in file_patterns")
        # ensure anchor pattern is unambiguous
        anchor_pat = self.file_patterns[self.anchor]
        if "}{" in anchor_pat:
            raise ValueError("Anchor pattern must be unambiguous (no '}{').")
        # ensure anchor contains all core fields (so anchor can define canonical ids)
        anchor_placeholders = {fname for _, fname, _, _ in Formatter().parse(anchor_pat) if fname}
        anchor_core = {f for f in anchor_placeholders if not f.startswith(self.ignored_prefix)}
        missing_in_anchor = core_fields - anchor_core
        if missing_in_anchor:
            raise ValueError(
                "Anchor pattern does not contain all core placeholders. "
                f"Missing: {sorted(missing_in_anchor)}"
            )

        # Populate paths dict by scanning disk
        self.refresh()
        
    def refresh(self):
        """
        Two-step refresh to handle ambiguous patterns with consecutive placeholders.

        1. Scan unambiguous patterns (no '}{') to build canonical file_ids.
        2. Use these file_ids to resolve ambiguous patterns safely.
        """
        paths: dict[str, dict[str, Path]] = {}

        # Step 1: use the single canonical anchor to build canonical file_ids
        matched = self._glob_pattern(self.anchor)
        if not matched:
            # if anchor exists but matches nothing that is likely an environmental issue
            raise ValueError(f"Anchor '{self.anchor}' matched no files on disk. Please check your file pattern definition or IO environment.")

        for file_id, fname in matched.items():
            paths[file_id] = {self.anchor: Path(fname)}

        # Step 2: possibly ambiguous patterns
        for kind, pattern in self.file_patterns.items():
            if kind == self.anchor:
                continue
            to_remove = set()
            for file_id in paths:
                fields = self.id2dict(file_id)
                # Fill foo placeholders with '*' to allow globbing
                all_placeholders = {fname for _, fname, _, _ in Formatter().parse(pattern) if fname}
                placeholders = {p: "*" for p in all_placeholders if p.startswith(self.ignored_prefix)}
                fmt_dict = {**fields, **placeholders}
                try:
                    expected_fname = pattern.format(**fmt_dict)
                except KeyError:
                    if kind in self.required:
                        log_or_print(f"file_id '{file_id}' removed: kind '{kind}' pattern has unresolvable fields.", warning=True)
                        to_remove.add(file_id)
                    continue

                candidates = glob.glob(expected_fname)
                if not candidates:
                    if kind in self.required:
                        log_or_print(f"file_id '{file_id}' removed: kind '{kind}' not found on disk (expected: {expected_fname}).", warning=True)
                        to_remove.add(file_id)
                    continue
                # assign first matching file
                if len(candidates) > 1:
                    raise ValueError(
                        f"Multiple ambiguous files found for file_id '{file_id}', kind '{kind}': {candidates}"
                    )
                paths[file_id][kind] = Path(candidates[0])
            for file_id in to_remove:
                del paths[file_id]

        object.__setattr__(self, "paths", paths)

    def __getitem__(self, file_id: str) -> dict[str, Path]:
        """Return dict of {kind: Path} for a file_id."""
        return self.paths[file_id]
    
    def get(self, file_id: str, default: Optional[Dict[str, Path]] = None) -> Dict[str, Path]:
        """Safe get for a file_id. Returns mapping kind->Path or a new empty dict.

        If `default` is provided (non-None) it is returned when file_id is missing.
        When `default` is None, an empty dict is returned (fresh instance).
        """
        if default is None:
            default = {}
        return self.paths.get(file_id, default)
    
    def __contains__(self, file_id: str) -> bool:
        """Check if a file_id exists in the paths dictionary."""
        return file_id in self.paths

    # Mapping protocol ---------------------------------------------------
    def __iter__(self) -> Iterator[str]:
        """Iterate over file_id keys (live view)."""
        return iter(self.paths)

    def __len__(self) -> int:
        """Number of file_ids currently discovered."""
        return len(self.paths)

    # `keys()`, `items()`, `values()` are provided by Mapping automatically,
    # but help with typing hints if callers rely on KeysView.
    def keys(self) -> KeysView[str]:
        """Return a live view of discovered file IDs (dict_keys view)."""
        return self.paths.keys()

    # Convenience: explicit items/values typing (optional)
    def items(self) -> Iterable[tuple[str, dict[str, Path]]]:
        return self.paths.items()

    def values(self) -> Iterable[dict[str, Path]]:
        return self.paths.values()

    # Small convenience & debugging helpers
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(anchor={self.anchor!r}, n_files={len(self)})"

    def filename2id(self, filename: str, kind: str) -> str:
        """
        Convert a filename into a file_id using the core fields
        of the pattern for the given kind.
        """
        return self.dict2id(self.filename2dict(filename, kind))

    def filename2dict(self, filename: str, kind: str) -> dict[str, str | None]:
        """
        Convert a filename into a dictionary of core fields → parsed values.
        Fields corresponding to placeholders starting with 'foo' are ignored.
        Missing core fields are set to None.

        Raises a warning if any core field is missing in the filename.
        """
        if kind not in self.file_patterns:
            raise KeyError(f"Kind '{kind}' not found in file_patterns.")

        pattern = self.file_patterns[kind]
        parsed = parse.parse(pattern, str(filename))
        if parsed is None:
            raise ValueError(
                f"Failed to parse filename '{filename}' with pattern of kind '{kind}'"
            )

        parsed_dict = parsed.named
        core_dict = {}
        for field in self.fieldnames:
            value = parsed_dict.get(field, None)
            if value is None:
                log_or_print(f"Warning: field '{field}' missing in file '{filename}'")
            core_dict[field] = value
        return core_dict

    def _glob_pattern(self, kind: str) -> dict[str, str]:
        """
        Scan filesystem for all files matching self.file_patterns[kind].
        Returns a dict mapping file_id → filename for unique matches.
        """
        pattern = self.file_patterns[kind]

        # Build a glob pattern by replacing each placeholder with '*' while preserving literal parts.
        glob_parts = []
        for literal_text, field_name, _, _ in Formatter().parse(pattern):
            glob_parts.append(literal_text)
            if field_name:
                glob_parts.append("*")
        glob_pattern = "".join(glob_parts)

        # Use glob.glob (recursive if user used ** in patterns).
        candidate_files = sorted(glob.glob(glob_pattern))

        matched_files: dict[str, str] = {}
        for fname in candidate_files:
            if parse.parse(pattern, fname) is None:
                continue

            fields = self.filename2dict(fname, kind)
            file_id = self.dict2id(fields)

            if file_id in matched_files:
                raise ValueError(
                    f"Duplicate file_id '{file_id}' for multiple files: "
                    f"{matched_files[file_id]} and {fname}"
                )

            matched_files[file_id] = fname

        return matched_files

    @abstractmethod
    def dict2id(self, fields: Dict[str, str | None]) -> str:
        """Convert a dictionary of core fields into a canonical file_id string."""

    @abstractmethod
    def id2dict(self, file_id: str) -> Dict[str, str | None]:
        """Convert a canonical file_id string back into a dictionary of core fields."""