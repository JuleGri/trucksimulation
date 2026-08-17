"""
01_train_test_split.py

Purpose:
- perform a case-level, temporal 80/20 hold-out split of the CTB event log;
- provide a reproducible training portion for parameter discovery and a
  held-out testing portion for distributional simulation validation;
- record the split cutoff and per-activity coverage in a manifest for the
  thesis appendix.

Inputs:
- data/processed/CTB/s6_eventlog_target_rank_features.csv (default; override
  with --input).

Outputs:
- data/processed/CTB/s6_train.csv
- data/processed/CTB/s6_test.csv
- validation/results/split_manifest.json

Methodological note:
- cases are ordered by their earliest event timestamp (min of
  enabled/start/time columns). The 80th quantile of that ordering is the
  cutoff. All events belonging to a case go entirely to train or test to
  avoid leakage of context features (utilization, demand) that are
  aggregated across the case.
- the split is deterministic; --seed only controls the tie-breaking of
  cases sharing the exact same arrival timestamp.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data") / "processed" / "CTB" / "s6_eventlog_target_rank_features.csv"
DEFAULT_TRAIN = Path("data") / "processed" / "CTB" / "s6_train.csv"
DEFAULT_TEST = Path("data") / "processed" / "CTB" / "s6_test.csv"
DEFAULT_MANIFEST = Path("validation") / "results" / "split_manifest.json"

CASE_COL = "case:concept:name"
ACT_COL = "concept:name"
TS_COLS = ["enabled:timestamp", "start:timestamp", "time:timestamp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test-out", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--test-size", type=float, default=0.20,
                        help="Fraction of cases (chronologically last) held out for validation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Only affects tie-breaking for identical arrival timestamps.")
    return parser.parse_args()


def load_eventlog(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Event log not found at {path.resolve()}")
    print(f"[split] Reading {path}")
    df = pd.read_csv(path)
    for col in TS_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if CASE_COL not in df.columns:
        raise KeyError(f"Event log is missing required column {CASE_COL!r}")
    return df


def compute_case_arrival(df: pd.DataFrame) -> pd.Series:
    """Return the earliest event timestamp per case (used as arrival)."""
    available = [c for c in TS_COLS if c in df.columns]
    if not available:
        raise KeyError("Event log has none of the expected timestamp columns.")
    stacked = df[[CASE_COL] + available].copy()
    stacked["_min_ts"] = stacked[available].min(axis=1)
    return stacked.groupby(CASE_COL)["_min_ts"].min()


def split_case_ids(case_arrival: pd.Series, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("--test-size must be strictly between 0 and 1.")
    df = case_arrival.reset_index().rename(columns={"_min_ts": "arrival_ts"})
    df = df.dropna(subset=["arrival_ts"])
    # Deterministic tie-breaker so cases with identical arrival timestamps
    # do not oscillate between train and test across runs.
    rng = np.random.default_rng(seed)
    df["_tiebreak"] = rng.random(len(df))
    df = df.sort_values(["arrival_ts", "_tiebreak"]).reset_index(drop=True)
    cutoff_index = int(np.floor(len(df) * (1.0 - test_size)))
    cutoff_index = max(1, min(len(df) - 1, cutoff_index))
    train_ids = df.iloc[:cutoff_index][CASE_COL].to_numpy()
    test_ids = df.iloc[cutoff_index:][CASE_COL].to_numpy()
    cutoff_ts = df.iloc[cutoff_index]["arrival_ts"]
    return train_ids, test_ids, cutoff_ts


def per_activity_counts(df: pd.DataFrame) -> dict[str, int]:
    if ACT_COL not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[ACT_COL].value_counts().to_dict().items()}


def summarize(df: pd.DataFrame, label: str) -> dict:
    available_ts = [c for c in TS_COLS if c in df.columns]
    ts_min = min((df[c].min() for c in available_ts), default=pd.NaT)
    ts_max = max((df[c].max() for c in available_ts), default=pd.NaT)
    return {
        "label": label,
        "n_events": int(len(df)),
        "n_cases": int(df[CASE_COL].nunique()),
        "timestamp_min": None if pd.isna(ts_min) else ts_min.isoformat(),
        "timestamp_max": None if pd.isna(ts_max) else ts_max.isoformat(),
        "activity_counts": per_activity_counts(df),
    }


def main() -> int:
    args = parse_args()

    df = load_eventlog(args.input)
    case_arrival = compute_case_arrival(df)
    train_ids, test_ids, cutoff_ts = split_case_ids(case_arrival, args.test_size, args.seed)

    train_ids_set = set(train_ids.tolist())
    test_ids_set = set(test_ids.tolist())
    train_df = df[df[CASE_COL].isin(train_ids_set)].copy()
    test_df = df[df[CASE_COL].isin(test_ids_set)].copy()

    # Sanity: no overlap and full coverage
    overlap = train_ids_set & test_ids_set
    if overlap:
        raise RuntimeError(f"Case-id overlap between train and test ({len(overlap)} cases).")
    covered = len(train_df) + len(test_df)
    dropped = len(df) - covered
    if dropped < 0:
        raise RuntimeError("Row accounting inconsistency: covered exceeds total.")

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.test_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    print(f"[split] Writing train -> {args.train_out} ({len(train_df):,} events / {len(train_ids):,} cases)")
    train_df.to_csv(args.train_out, index=False)
    print(f"[split] Writing test  -> {args.test_out} ({len(test_df):,} events / {len(test_ids):,} cases)")
    test_df.to_csv(args.test_out, index=False)

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "input_path": str(args.input),
        "train_path": str(args.train_out),
        "test_path": str(args.test_out),
        "test_size": args.test_size,
        "seed": args.seed,
        "cutoff_arrival_ts": cutoff_ts.isoformat() if pd.notna(cutoff_ts) else None,
        "n_cases_total": int(df[CASE_COL].nunique()),
        "n_events_total": int(len(df)),
        "n_events_dropped_missing_ts": int(dropped),
        "train": summarize(train_df, "train"),
        "test": summarize(test_df, "test"),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2))
    print(f"[split] Manifest -> {args.manifest}")

    # Console recap
    print("\n[split] --- Summary ---")
    print(f"  Cutoff arrival timestamp : {manifest['cutoff_arrival_ts']}")
    print(f"  Train cases              : {manifest['train']['n_cases']:>7,}   events: {manifest['train']['n_events']:>8,}")
    print(f"  Test  cases              : {manifest['test']['n_cases']:>7,}   events: {manifest['test']['n_events']:>8,}")
    if dropped:
        print(f"  Events dropped (no ts)   : {dropped:>7,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
