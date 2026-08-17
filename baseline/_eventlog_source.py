"""
_eventlog_source.py — shared helpers for baseline discovery.

Purpose:
- centralise the event-log input resolution so every discovery script
  can be pointed at the held-out training log (s6_train.csv) produced by
  validation/01_train_test_split.py instead of the full event log;
- keep the paramset naming consistent so results computed on the full
  log and on the train split do not overwrite each other on disk.

Resolution order used by resolve_event_csv():
1. explicit path passed as argument (typically an argparse --input);
2. TRUCKSIM_EVENTLOG environment variable (absolute path);
3. data/processed/CTB/s6_train.csv (held-out training log — new default);
4. data/processed/CTB/s6_eventlog_target_rank_features.csv (full log).

Rationale:
- The ProSiT reference implementation (Vinci et al., 2026) performs an
  ad-hoc in-memory 80/20 split every time the demo is run
  (README > "Full workflow with evaluation"). That leaves no persistent
  hold-out artefact and no reproducible boundary between discovery
  training data and validation data. This helper guarantees that every
  discovery script in this project consumes the SAME persisted training
  split produced by validation/01_train_test_split.py.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable


# Path stubs are relative to the repository root (one level above this file:
# baseline/_eventlog_source.py -> trucksimulation/).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TRAIN = _REPO_ROOT / "data" / "processed" / "CTB" / "s6_train.csv"
_DEFAULT_FULL = _REPO_ROOT / "data" / "processed" / "CTB" / "s6_eventlog_target_rank_features.csv"
_LEGACY_FULL_V1 = _REPO_ROOT / "data" / "processed" / "CTB" / "s6_eventlog_target_rank_features_v1.csv"


def _candidate_paths(explicit: str | os.PathLike | None) -> Iterable[Path]:
    if explicit:
        yield Path(explicit)
    env_override = os.environ.get("TRUCKSIM_EVENTLOG")
    if env_override:
        yield Path(env_override)
    yield _DEFAULT_TRAIN
    yield _DEFAULT_FULL
    yield _LEGACY_FULL_V1


def resolve_event_csv(explicit: str | os.PathLike | None = None, *, verbose: bool = True) -> str:
    """Return the first existing candidate path for the event log.

    Prints a banner identifying which log was chosen when verbose=True so
    downstream artefacts can be traced back to their input.
    """
    tried = []
    for candidate in _candidate_paths(explicit):
        candidate = Path(candidate).expanduser().resolve()
        tried.append(str(candidate))
        if candidate.exists():
            if verbose:
                label = _describe_source(candidate)
                print(f"[eventlog] Using {label}: {candidate}")
            return str(candidate)
    raise FileNotFoundError(
        "No event log candidate exists. Tried:\n  - "
        + "\n  - ".join(tried)
        + "\nRun validation/01_train_test_split.py to produce s6_train.csv."
    )


def _describe_source(path: Path) -> str:
    name = path.name.lower()
    if name == _DEFAULT_TRAIN.name.lower():
        return "held-out TRAIN log"
    if name == _DEFAULT_FULL.name.lower() or name == _LEGACY_FULL_V1.name.lower():
        return "FULL event log (no hold-out — use only for legacy comparisons)"
    return "custom event log"


def is_train_log(path: str | os.PathLike) -> bool:
    """Return True if path points to the s6_train.csv training split."""
    try:
        return Path(path).resolve().name.lower() == _DEFAULT_TRAIN.name.lower()
    except OSError:
        return False


def paramset_suffix_for(path: str | os.PathLike) -> str:
    """Suffix to append to a discovery_params/<paramset>/ folder based on the input."""
    return "train80" if is_train_log(path) else "full"


def add_input_arg(parser: argparse.ArgumentParser) -> None:
    """Register the standard --input flag on a script's ArgumentParser."""
    parser.add_argument(
        "--input",
        dest="input_csv",
        default=None,
        help="Path to the event log CSV. Defaults to s6_train.csv when it exists.",
    )


def parse_input_arg() -> str | None:
    """Convenience wrapper for scripts that don't otherwise use argparse.

    Only recognises --input=<path> so that unknown flags do not blow up.
    Returns None when the flag is not provided.
    """
    import sys

    argv = sys.argv[1:]
    result = None
    for i, token in enumerate(argv):
        if token == "--input" and i + 1 < len(argv):
            result = argv[i + 1]
        elif token.startswith("--input="):
            result = token.split("=", 1)[1]
    return result


def resolve_paramset_dir(script_dir: str | os.PathLike, suffix: str,
                         *, create: bool = True) -> str:
    """Resolve or create the discovery_params/<paramset>/ folder for the current run.

    Existing folders whose name ends in the given suffix are reused so that
    successive discovery steps write into the same paramset. When no matching
    folder exists a new one named ``params_<utc-timestamp>_<suffix>`` is
    created (when ``create=True``).
    """
    root = Path(script_dir) / "discovery_params"
    root.mkdir(parents=True, exist_ok=True)
    suffix_tag = f"_{suffix}" if suffix else ""
    matches = [p for p in root.iterdir() if p.is_dir() and (not suffix or p.name.endswith(suffix_tag))]
    if matches:
        return str(sorted(matches, key=lambda p: p.name)[-1])
    if not create:
        raise FileNotFoundError(
            f"No discovery paramset folder ending in '{suffix_tag}' under {root}. "
            "Re-run with create=True or supply --paramset."
        )
    from datetime import datetime as _dt

    name = "params_" + _dt.utcnow().strftime("%Y%m%d_%H%M%S") + suffix_tag
    new_dir = root / name
    new_dir.mkdir(parents=True, exist_ok=True)
    return str(new_dir)
