#!/usr/bin/env python3
"""Compare simulated and held-out CTB control flow with 3-gram distance.

The primary metric follows Chapela-Campa et al.'s N-Gram Distribution
Distance (NGD): events are ordered by completion timestamp, each case is
padded at both ends with ``n - 1`` dummy activities, and the sum of absolute
differences between n-gram counts is divided by the combined number of
n-grams in both logs. Lower values are better and the range is [0, 1].

The CTB log also contains an explicit case order because three held-out cases
have small timestamp inversions. Therefore, the published completion-order
metric is accompanied by an explicit-order sensitivity value. Both use the
same padding and normalization.

By default, the four standardized feature-source configurations are compared
over identical seeds 42--51: visit-only, static-only, native-only, and both.
Alternative frozen bundles can be supplied explicitly on the command line.

The script performs no discovery and never mutates the input pickles.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from _eventlog_contract import ACT_COL, CASE_COL, ORDER_COL, TS_COMPLETE  # noqa: E402
from _prosit_ctb_calibration import simulate_ctb  # noqa: E402


DEFAULT_REAL = REPO_ROOT / "data" / "processed" / "CTB" / "s6_test.csv"
DEFAULT_MANIFEST = REPO_ROOT / "validation" / "results" / "split_manifest.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "validation"
    / "results"
    / "feature_source_factorial_standardized_ngd_20260829"
)
DEFAULT_N = 3
DUMMY_ACTIVITY = "<BOUNDARY>"


@dataclass(frozen=True)
class Configuration:
    name: str
    params: Path


DEFAULT_CONFIGURATIONS = (
    Configuration(
        "visit_only",
        REPO_ROOT
        / "baseline/discovery_params/params_20260816_214403_train80"
        / "prosit_discovery_feature_visit_only_common_20260829"
        / "prosit_params.pkl",
    ),
    Configuration(
        "static_only",
        REPO_ROOT
        / "baseline/discovery_params/params_20260816_214403_train80"
        / "prosit_discovery_feature_static_only_20260829"
        / "prosit_params.pkl",
    ),
    Configuration(
        "native_only",
        REPO_ROOT
        / "baseline/discovery_params/params_20260816_214403_train80"
        / "prosit_discovery_feature_native_only_20260829"
        / "prosit_params.pkl",
    ),
    Configuration(
        "both",
        REPO_ROOT
        / "baseline/discovery_params/params_20260816_214403_train80"
        / "prosit_discovery_feature_both_common_20260829"
        / "prosit_params.pkl",
    ),
)

PAIRED_CONTRASTS = (
    ("static_only", "visit_only"),
    ("native_only", "visit_only"),
    ("both", "static_only"),
    ("both", "native_only"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=tuple(configuration.name for configuration in DEFAULT_CONFIGURATIONS),
        default=[configuration.name for configuration in DEFAULT_CONFIGURATIONS],
        help="Frozen configurations to run (default: all four).",
    )
    parser.add_argument(
        "--configuration",
        action="append",
        default=[],
        metavar="NAME=PARAMS_PKL",
        help=(
            "Custom configuration. Repeat for several bundles. When supplied, "
            "these replace --configurations and permit factorial experiments."
        ),
    )
    parser.add_argument(
        "--contrast",
        action="append",
        default=[],
        metavar="LEFT:RIGHT",
        help="Custom paired contrast. Repeat as needed.",
    )
    parser.add_argument("--n-traces", type=int, default=None)
    parser.add_argument("--t-start", default=None)
    parser.add_argument(
        "--timestamp-resolution",
        choices=("minute", "native"),
        default="minute",
    )
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    if args.n_seeds < 1:
        parser.error("--n-seeds must be positive")
    return args


def parse_custom_configurations(values: list[str]) -> list[Configuration]:
    configurations = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --configuration {value!r}; expected NAME=PATH")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("Custom configuration name cannot be empty")
        configurations.append(Configuration(name, Path(raw_path).expanduser()))
    names = [configuration.name for configuration in configurations]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate custom configuration names: {names}")
    return configurations


def parse_custom_contrasts(values: list[str]) -> list[tuple[str, str]]:
    contrasts = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"Invalid --contrast {value!r}; expected LEFT:RIGHT")
        left, right = (part.strip() for part in value.split(":", 1))
        if not left or not right:
            raise ValueError(f"Invalid --contrast {value!r}; names cannot be empty")
        contrasts.append((left, right))
    return contrasts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Parameter pickle not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def resolve_window(
    manifest_path: Path,
    real: pd.DataFrame,
    n_traces_override: int | None,
    t_start_override: str | None,
) -> tuple[int, datetime | None, dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    n_traces = n_traces_override
    if n_traces is None:
        n_traces = int(manifest.get("test", {}).get("n_cases", real[CASE_COL].nunique()))
    cutoff = t_start_override or manifest.get("cutoff_arrival_ts")
    t_start = (
        datetime.fromisoformat(str(cutoff).replace("Z", "+00:00"))
        if cutoff
        else None
    )
    return int(n_traces), t_start, manifest


def load_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Held-out log not found: {path}")
    frame = pd.read_csv(path)
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    missing = {CASE_COL, ACT_COL, TS_COMPLETE}.difference(frame.columns)
    if missing:
        raise KeyError(f"Held-out log is missing columns: {sorted(missing)}")
    return frame


def _ordered_activities(frame: pd.DataFrame, mode: str) -> list[list[str]]:
    """Return one activity sequence per case using a deterministic tie break."""

    if mode not in {"completion", "explicit"}:
        raise ValueError(f"Unsupported order mode: {mode}")
    missing = {CASE_COL, ACT_COL}.difference(frame.columns)
    if missing:
        raise KeyError(f"Event log is missing columns: {sorted(missing)}")

    selected_columns = [CASE_COL, ACT_COL]
    if ORDER_COL in frame.columns:
        selected_columns.append(ORDER_COL)
    if mode == "completion":
        if TS_COMPLETE not in frame.columns:
            raise KeyError(f"Completion-order NGD requires {TS_COMPLETE}")
        selected_columns.append(TS_COMPLETE)
    # Simulation output contains many wide case-attribute columns. NGD needs
    # only identifiers, activity, order and completion time; copying/sorting
    # the full frame made this step unnecessarily expensive.
    ordered = frame.loc[:, selected_columns].copy()
    ordered["_ngd_input_row"] = np.arange(len(ordered), dtype=np.int64)
    ordered["_ngd_case_rank"] = pd.factorize(ordered[CASE_COL], sort=False)[0]
    if ORDER_COL in ordered.columns:
        explicit = pd.to_numeric(ordered[ORDER_COL], errors="coerce")
        if explicit.isna().any():
            raise ValueError(f"{ORDER_COL} contains missing or non-numeric values")
        ordered["_ngd_explicit_order"] = explicit.astype(np.int64)
    else:
        ordered["_ngd_explicit_order"] = ordered.groupby(
            CASE_COL, sort=False
        ).cumcount()

    sort_columns = ["_ngd_case_rank"]
    if mode == "completion":
        ordered["_ngd_completion"] = pd.to_datetime(
            ordered[TS_COMPLETE], errors="coerce"
        )
        if ordered["_ngd_completion"].isna().any():
            raise ValueError(f"{TS_COMPLETE} contains missing or invalid timestamps")
        sort_columns.append("_ngd_completion")
    sort_columns.extend(["_ngd_explicit_order", "_ngd_input_row"])
    ordered = ordered.sort_values(sort_columns, kind="stable")

    activities = ordered[ACT_COL].astype(str).to_numpy()
    case_ranks = ordered["_ngd_case_rank"].to_numpy()
    if len(activities) == 0:
        return []
    boundaries = np.flatnonzero(case_ranks[1:] != case_ranks[:-1]) + 1
    return [part.tolist() for part in np.split(activities, boundaries)]


def ngram_profile(
    frame: pd.DataFrame,
    n: int,
    mode: str,
) -> Counter[tuple[str, ...]]:
    profile: Counter[tuple[str, ...]] = Counter()
    boundary = [DUMMY_ACTIVITY] * (n - 1)
    for activities in _ordered_activities(frame, mode):
        padded = [*boundary, *activities, *boundary]
        profile.update(
            tuple(padded[index : index + n])
            for index in range(len(padded) - n + 1)
        )
    return profile


def ngram_distance(
    first: Counter[tuple[str, ...]],
    second: Counter[tuple[str, ...]],
) -> float:
    denominator = sum(first.values()) + sum(second.values())
    if denominator <= 0:
        return 0.0
    keys = set(first) | set(second)
    return float(sum(abs(first[key] - second[key]) for key in keys) / denominator)


def profile_comparison_rows(
    real: Counter[tuple[str, ...]],
    simulated: Counter[tuple[str, ...]],
    configuration: str,
    seed: int,
    order_mode: str,
) -> Iterable[dict]:
    denominator = sum(real.values()) + sum(simulated.values())
    for gram in sorted(set(real) | set(simulated)):
        real_count = int(real[gram])
        sim_count = int(simulated[gram])
        yield {
            "configuration": configuration,
            "seed": int(seed),
            "order_mode": order_mode,
            **{f"activity_{index + 1}": gram[index] for index in range(len(gram))},
            "real_count": real_count,
            "sim_count": sim_count,
            "signed_contribution": (sim_count - real_count) / denominator,
            "absolute_contribution": abs(sim_count - real_count) / denominator,
        }


def mean_ci(values: pd.Series) -> tuple[int, float, float, float, float]:
    sample = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(sample) == 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(sample.mean())
    sd = float(sample.std(ddof=1)) if len(sample) > 1 else 0.0
    half_width = (
        float(stats.t.ppf(0.975, len(sample) - 1) * sd / np.sqrt(len(sample)))
        if len(sample) > 1
        else 0.0
    )
    return len(sample), mean, sd, mean - half_width, mean + half_width


def self_test() -> None:
    """Reproduce the 0.4 two-gram example from the published definition."""

    left: Counter[tuple[str, ...]] = Counter()
    right: Counter[tuple[str, ...]] = Counter()
    for _ in range(3):
        for profile, sequence in ((left, list("ABCD")), (right, list("ABED"))):
            padded = [DUMMY_ACTIVITY, *sequence, DUMMY_ACTIVITY]
            profile.update(tuple(padded[i : i + 2]) for i in range(len(padded) - 1))
    measured = ngram_distance(left, right)
    if not np.isclose(measured, 0.4):
        raise AssertionError(f"NGD self-test failed: expected 0.4, got {measured}")


def main() -> None:
    args = parse_args()
    self_test()
    args.output.mkdir(parents=True, exist_ok=True)

    real = load_log(args.real)
    n_traces, t_start, manifest = resolve_window(
        args.manifest, real, args.n_traces, args.t_start
    )
    real_profiles = {
        mode: ngram_profile(real, args.n, mode)
        for mode in ("completion", "explicit")
    }
    changed_cases = sum(
        completion != explicit
        for completion, explicit in zip(
            _ordered_activities(real, "completion"),
            _ordered_activities(real, "explicit"),
        )
    )
    selected_configurations = (
        parse_custom_configurations(args.configuration)
        if args.configuration
        else [
            configuration
            for configuration in DEFAULT_CONFIGURATIONS
            if configuration.name in args.configurations
        ]
    )
    paired_contrasts = (
        parse_custom_contrasts(args.contrast)
        if args.contrast
        else list(PAIRED_CONTRASTS)
    )

    replication_rows: list[dict] = []
    contribution_rows: list[dict] = []
    for configuration in selected_configurations:
        print(f"[ngd] Loading {configuration.name}: {configuration.params}")
        params = load_pickle(configuration.params)
        for offset in range(args.n_seeds):
            seed = args.base_seed + offset
            print(f"[ngd]   {configuration.name}, seed={seed} ...", end=" ", flush=True)
            simulation_started = time.perf_counter()
            simulated = simulate_ctb(
                params,
                n_traces=n_traces,
                t_start=t_start,
                seed=seed,
                timestamp_resolution=(
                    "min" if args.timestamp_resolution == "minute" else None
                ),
            )
            simulated = simulated.loc[:, ~simulated.columns.duplicated()].copy()
            simulation_seconds = time.perf_counter() - simulation_started
            print(f"simulation={simulation_seconds:.1f}s;", end=" ", flush=True)
            ngram_started = time.perf_counter()
            sim_profiles = {}
            for mode in ("completion", "explicit"):
                mode_started = time.perf_counter()
                sim_profiles[mode] = ngram_profile(simulated, args.n, mode)
                print(
                    f"{mode}={time.perf_counter() - mode_started:.1f}s;",
                    end=" ",
                    flush=True,
                )
            row = {
                "configuration": configuration.name,
                "seed": int(seed),
                "n": int(args.n),
                "sim_cases": int(simulated[CASE_COL].nunique()),
                "sim_events": int(len(simulated)),
                "simulation_seconds": float(simulation_seconds),
                "ngram_seconds": float(time.perf_counter() - ngram_started),
                "ngd_completion_order": ngram_distance(
                    real_profiles["completion"], sim_profiles["completion"]
                ),
                "ngd_explicit_order": ngram_distance(
                    real_profiles["explicit"], sim_profiles["explicit"]
                ),
                "real_unique_ngrams_completion": int(
                    len(real_profiles["completion"])
                ),
                "sim_unique_ngrams_completion": int(
                    len(sim_profiles["completion"])
                ),
            }
            replication_rows.append(row)
            for mode in ("completion", "explicit"):
                contribution_rows.extend(
                    profile_comparison_rows(
                        real_profiles[mode],
                        sim_profiles[mode],
                        configuration.name,
                        seed,
                        mode,
                    )
                )
            print(
                f"NGD={row['ngd_completion_order']:.6f}; "
                f"explicit={row['ngd_explicit_order']:.6f}; "
                f"ngrams={row['ngram_seconds']:.1f}s"
            )
            # Keep every completed seed recoverable if a later simulation is
            # interrupted. The final write below replaces this incrementally
            # complete table with the same schema.
            pd.DataFrame(replication_rows).to_csv(
                args.output / "ngd_replications.csv", index=False
            )

    replications = pd.DataFrame(replication_rows)
    replications.to_csv(args.output / "ngd_replications.csv", index=False)

    summary_rows = []
    for configuration, frame in replications.groupby("configuration", sort=False):
        for metric in ("ngd_completion_order", "ngd_explicit_order"):
            count, mean, sd, lo, hi = mean_ci(frame[metric])
            summary_rows.append(
                {
                    "configuration": configuration,
                    "metric": metric,
                    "n_replications": count,
                    "mean": mean,
                    "sd": sd,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "ngd_summary.csv", index=False)

    indexed = {
        name: frame.set_index("seed")
        for name, frame in replications.groupby("configuration", sort=False)
    }
    contrast_rows = []
    for left, right in paired_contrasts:
        if left not in indexed or right not in indexed:
            continue
        common = indexed[left].index.intersection(indexed[right].index)
        for metric in ("ngd_completion_order", "ngd_explicit_order"):
            difference = indexed[left].loc[common, metric] - indexed[right].loc[
                common, metric
            ]
            count, mean, sd, lo, hi = mean_ci(difference)
            contrast_rows.append(
                {
                    "contrast": f"{left}_minus_{right}",
                    "metric": metric,
                    "n_paired_seeds": count,
                    "mean_difference": mean,
                    "sd_difference": sd,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "interval_excludes_zero": bool(lo > 0 or hi < 0),
                }
            )
    pd.DataFrame(contrast_rows).to_csv(
        args.output / "ngd_paired_contrasts.csv", index=False
    )

    contributions = pd.DataFrame(contribution_rows)
    contributions.to_csv(args.output / "ngram_contributions_by_seed.csv", index=False)
    gram_columns = [f"activity_{index + 1}" for index in range(args.n)]
    contribution_summary = (
        contributions.groupby(
            ["configuration", "order_mode", *gram_columns],
            as_index=False,
            dropna=False,
        )[["real_count", "sim_count", "signed_contribution", "absolute_contribution"]]
        .mean()
        .rename(
            columns={
                "real_count": "real_count_mean",
                "sim_count": "sim_count_mean",
                "signed_contribution": "signed_contribution_mean",
                "absolute_contribution": "absolute_contribution_mean",
            }
        )
        .sort_values(
            ["configuration", "order_mode", "absolute_contribution_mean"],
            ascending=[True, True, False],
        )
    )
    contribution_summary.to_csv(
        args.output / "ngram_contribution_summary.csv", index=False
    )

    real_rows = []
    for mode, profile in real_profiles.items():
        for gram, count in sorted(profile.items()):
            real_rows.append(
                {
                    "order_mode": mode,
                    **{f"activity_{index + 1}": gram[index] for index in range(args.n)},
                    "count": int(count),
                }
            )
    pd.DataFrame(real_rows).to_csv(
        args.output / "heldout_ngram_profile.csv", index=False
    )

    manifest_payload = {
        "schema": "ctb-ngram-control-flow-1",
        "metric": {
            "name": "N-Gram Distribution Distance",
            "n": int(args.n),
            "primary_order": "completion timestamp, stable explicit/input-order tie break",
            "sensitivity_order": "explicit case:event:order when present; input order otherwise",
            "padding": f"n-1 copies of {DUMMY_ACTIVITY!r} at both trace boundaries",
            "normalization": "sum absolute count differences / combined n-gram count",
            "better": "lower; 0 means identical n-gram histograms",
            "reference": "Chapela-Campa et al., A framework for measuring the quality of business process simulation models, Information Systems 127 (2025) 102447",
        },
        "design": {
            "seeds": list(range(args.base_seed, args.base_seed + args.n_seeds)),
            "cases_per_run": int(n_traces),
            "start_time": t_start.isoformat() if t_start else None,
            "timestamp_resolution": args.timestamp_resolution,
            "real_cases": int(real[CASE_COL].nunique()),
            "real_events": int(len(real)),
            "real_cases_changed_by_completion_sort": int(changed_cases),
            "split_cutoff": manifest.get("cutoff_arrival_ts"),
        },
        "inputs": {
            "real_log": {
                "path": str(args.real.resolve()),
                "sha256": sha256(args.real),
            },
            "split_manifest": {
                "path": str(args.manifest.resolve()),
                "sha256": sha256(args.manifest),
            },
            "simulation_runner": {
                "path": str((REPO_ROOT / "_prosit_ctb_calibration.py").resolve()),
                "sha256": sha256(REPO_ROOT / "_prosit_ctb_calibration.py"),
            },
            "experiment_driver": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "configurations": {
                configuration.name: {
                    "path": str(configuration.params.resolve()),
                    "sha256": sha256(configuration.params),
                }
                for configuration in selected_configurations
            },
        },
        "privacy_note": (
            "heldout_ngram_profile.csv contains aggregate activity-sequence counts only; "
            "it contains no case identifiers, resources, attributes, or timestamps."
        ),
    }
    with (args.output / "ngd_run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    print("\n[ngd] === Summary ===")
    print(summary.to_string(index=False))
    print(f"[ngd] Results written to {args.output}")


if __name__ == "__main__":
    main()
