"""Paired comparison of the discovered and physically capped RMG models.

The scenario driver writes one replication table for the frozen discovered
resource capacities and one for the optional domain-constrained sensitivity
baseline.  This script joins both tables by seed and scenario and reports:

1. the direct effect of the RMG capacity cap for each scenario; and
2. the interaction between the cap and each operational intervention
   (difference of paired scenario effects).

The comparison is valid only when both runs use the same frozen source model,
seeds, trace count, start time, timestamp resolution and demand intervention.
Those conditions and the contract status are hard preconditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = SCRIPT_DIR / "results"
DEFAULT_REFERENCE = (
    RESULTS_ROOT / "final_deterministic_20260829_rules_workload_scenarios_uncapped_ci"
)
DEFAULT_SENSITIVITY = (
    RESULTS_ROOT / "final_deterministic_20260829_rules_workload_scenarios_cap3_ci"
)

SCENARIO_BASELINE = "baseline"
INTERVENTIONS = ("t22_closed", "demand_plus_20pct")
SCENARIOS = (SCENARIO_BASELINE, *INTERVENTIONS)

METRICS = (
    "mean_turnaround_min",
    "median_turnaround_min",
    "p90_turnaround_min",
    "mean_rmg_service_min",
    "median_rmg_service_min",
    "p90_rmg_service_min",
    "mean_rmg_pre_service_min",
    "median_rmg_pre_service_min",
    "p90_rmg_pre_service_min",
    "rmg_events",
    "mean_inter_arrival_min",
    "zero_inter_arrival_share",
    "arrival_rate_per_elapsed_hour",
)

HEADLINE_METRICS = (
    "mean_turnaround_min",
    "p90_turnaround_min",
    "mean_rmg_service_min",
    "mean_rmg_pre_service_min",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--sensitivity-dir", type=Path, default=DEFAULT_SENSITIVITY)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to --sensitivity-dir.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_run_pair(reference: dict, sensitivity: dict) -> None:
    for name, run in (("reference", reference), ("sensitivity", sensitivity)):
        if run.get("status") != "completed":
            raise ValueError(f"{name} run is not completed: {run.get('status')!r}")
        if not run.get("all_contracts_passed"):
            raise ValueError(f"{name} run did not pass all case contracts.")
        if not run.get("all_duration_tail_counts_zero"):
            raise ValueError(f"{name} run contains a duration-tail violation.")

    comparable_fields = (
        "source_baseline_sha256",
        "n_traces",
        "t_start",
        "seeds",
        "timestamp_resolution",
        "demand_increase_pct",
    )
    # The original run predates the explicit source/effective distinction, so
    # its baseline hash is also its source hash.
    reference_source_hash = reference.get(
        "source_baseline_sha256", reference.get("baseline_sha256")
    )
    sensitivity_source_hash = sensitivity.get(
        "source_baseline_sha256", sensitivity.get("baseline_sha256")
    )
    if reference_source_hash != sensitivity_source_hash:
        raise ValueError("Runs do not originate from the same frozen source model.")

    for field in comparable_fields[1:]:
        if reference.get(field) != sensitivity.get(field):
            raise ValueError(
                f"Run metadata differs for {field}: "
                f"{reference.get(field)!r} != {sensitivity.get(field)!r}"
            )
    if reference.get("rmg_max_concurrency_cap") is not None:
        raise ValueError("Reference run must not already contain an RMG cap.")
    if sensitivity.get("rmg_max_concurrency_cap") is None:
        raise ValueError("Sensitivity run does not declare an RMG cap.")


def _load_replications(run_dir: Path) -> tuple[pd.DataFrame, Path]:
    path = run_dir / "scenario_replications.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"seed", "scenario", *METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{path} lacks columns: {sorted(missing)}")
    if frame.duplicated(["seed", "scenario"]).any():
        raise ValueError(f"{path} contains duplicate seed/scenario rows.")
    observed = set(frame["scenario"])
    if observed != set(SCENARIOS):
        raise ValueError(f"Unexpected scenarios in {path}: {sorted(observed)}")
    return frame, path


def _ci(values: pd.Series) -> tuple[int, float, float, float, float]:
    samples = (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    n = int(len(samples))
    if n < 2:
        raise ValueError("At least two paired replications are required.")
    mean = float(samples.mean())
    std = float(samples.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n))
    return n, mean, std, mean - half, mean + half


def _paired_cap_effects(reference: pd.DataFrame, sensitivity: pd.DataFrame) -> pd.DataFrame:
    merged = reference.merge(
        sensitivity,
        on=["seed", "scenario"],
        suffixes=("_reference", "_sensitivity"),
        validate="one_to_one",
    )
    if len(merged) != len(reference) or len(merged) != len(sensitivity):
        raise ValueError("The replication tables do not have identical seed/scenario keys.")

    rows = []
    for row in merged.itertuples(index=False):
        for metric in METRICS:
            reference_value = float(getattr(row, f"{metric}_reference"))
            sensitivity_value = float(getattr(row, f"{metric}_sensitivity"))
            delta = sensitivity_value - reference_value
            rows.append(
                {
                    "seed": int(row.seed),
                    "scenario": row.scenario,
                    "metric": metric,
                    "reference_value": reference_value,
                    "sensitivity_value": sensitivity_value,
                    "delta": delta,
                    "delta_pct": (
                        100.0 * delta / reference_value
                        if reference_value != 0.0
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _summarize_effects(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        for metric in METRICS:
            group = effects[
                effects["scenario"].eq(scenario) & effects["metric"].eq(metric)
            ]
            n, mean_delta, std_delta, lo, hi = _ci(group["delta"])
            _, mean_pct, std_pct, pct_lo, pct_hi = _ci(group["delta_pct"])
            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "n": n,
                    "mean_reference": float(group["reference_value"].mean()),
                    "mean_sensitivity": float(group["sensitivity_value"].mean()),
                    "mean_delta": mean_delta,
                    "std_delta": std_delta,
                    "ci95_delta_lo": lo,
                    "ci95_delta_hi": hi,
                    "mean_delta_pct": mean_pct,
                    "std_delta_pct": std_pct,
                    "ci95_delta_pct_lo": pct_lo,
                    "ci95_delta_pct_hi": pct_hi,
                    "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
                }
            )
    return pd.DataFrame(rows)


def _interactions(reference: pd.DataFrame, sensitivity: pd.DataFrame) -> pd.DataFrame:
    ref = reference.set_index(["seed", "scenario"])
    cap = sensitivity.set_index(["seed", "scenario"])
    rows = []
    for seed in sorted(reference["seed"].unique()):
        for scenario in INTERVENTIONS:
            for metric in METRICS:
                reference_effect = float(
                    ref.loc[(seed, scenario), metric]
                    - ref.loc[(seed, SCENARIO_BASELINE), metric]
                )
                sensitivity_effect = float(
                    cap.loc[(seed, scenario), metric]
                    - cap.loc[(seed, SCENARIO_BASELINE), metric]
                )
                rows.append(
                    {
                        "seed": int(seed),
                        "scenario": scenario,
                        "metric": metric,
                        "reference_scenario_effect": reference_effect,
                        "sensitivity_scenario_effect": sensitivity_effect,
                        "interaction_delta": sensitivity_effect - reference_effect,
                    }
                )
    return pd.DataFrame(rows)


def _summarize_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in INTERVENTIONS:
        for metric in METRICS:
            group = interactions[
                interactions["scenario"].eq(scenario)
                & interactions["metric"].eq(metric)
            ]
            n, mean, std, lo, hi = _ci(group["interaction_delta"])
            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "n": n,
                    "mean_reference_scenario_effect": float(
                        group["reference_scenario_effect"].mean()
                    ),
                    "mean_sensitivity_scenario_effect": float(
                        group["sensitivity_scenario_effect"].mean()
                    ),
                    "mean_interaction_delta": mean,
                    "std_interaction_delta": std,
                    "ci95_interaction_lo": lo,
                    "ci95_interaction_hi": hi,
                    "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    reference_dir = args.reference_dir.resolve()
    sensitivity_dir = args.sensitivity_dir.resolve()
    out_dir = (args.out_dir or sensitivity_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_summary_path = reference_dir / "scenario_run_summary.json"
    sensitivity_summary_path = sensitivity_dir / "scenario_run_summary.json"
    reference_summary = _load_json(reference_summary_path)
    sensitivity_summary = _load_json(sensitivity_summary_path)
    _validate_run_pair(reference_summary, sensitivity_summary)

    reference, reference_replications_path = _load_replications(reference_dir)
    sensitivity, sensitivity_replications_path = _load_replications(sensitivity_dir)
    effects = _paired_cap_effects(reference, sensitivity)
    effect_summary = _summarize_effects(effects)
    interactions = _interactions(reference, sensitivity)
    interaction_summary = _summarize_interactions(interactions)

    effects_path = out_dir / "capacity_sensitivity_paired_deltas.csv"
    effect_summary_path = out_dir / "capacity_sensitivity_paired_delta_summary.csv"
    interactions_path = out_dir / "capacity_sensitivity_interactions.csv"
    interaction_summary_path = out_dir / "capacity_sensitivity_interaction_summary.csv"
    effects.to_csv(effects_path, index=False)
    effect_summary.to_csv(effect_summary_path, index=False)
    interactions.to_csv(interactions_path, index=False)
    interaction_summary.to_csv(interaction_summary_path, index=False)

    headline_effects = effect_summary[
        effect_summary["metric"].isin(HEADLINE_METRICS)
    ]
    headline_interactions = interaction_summary[
        interaction_summary["metric"].isin(HEADLINE_METRICS)
    ]
    comparison = {
        "status": "completed",
        "comparison_semantics": "sensitivity minus reference, paired by seed",
        "interaction_semantics": (
            "(scenario minus baseline under sensitivity) minus "
            "(scenario minus baseline under reference), paired by seed"
        ),
        "reference_run_summary": str(reference_summary_path),
        "reference_run_summary_sha256": _sha256(reference_summary_path),
        "sensitivity_run_summary": str(sensitivity_summary_path),
        "sensitivity_run_summary_sha256": _sha256(sensitivity_summary_path),
        "reference_replications": str(reference_replications_path),
        "reference_replications_sha256": _sha256(reference_replications_path),
        "sensitivity_replications": str(sensitivity_replications_path),
        "sensitivity_replications_sha256": _sha256(sensitivity_replications_path),
        "source_baseline_sha256": sensitivity_summary["source_baseline_sha256"],
        "seeds": sensitivity_summary["seeds"],
        "n_traces": sensitivity_summary["n_traces"],
        "rmg_concurrency_before": sensitivity_summary["rmg_concurrency_before"],
        "rmg_concurrency_after": sensitivity_summary["rmg_concurrency_after"],
        "headline_cap_effects": headline_effects.to_dict(orient="records"),
        "headline_interactions": headline_interactions.to_dict(orient="records"),
        "artifacts": {
            "paired_deltas": str(effects_path),
            "paired_delta_summary": str(effect_summary_path),
            "interactions": str(interactions_path),
            "interaction_summary": str(interaction_summary_path),
        },
    }
    comparison_path = out_dir / "capacity_sensitivity_comparison.json"
    with comparison_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)

    columns = [
        "scenario",
        "metric",
        "mean_reference",
        "mean_sensitivity",
        "mean_delta",
        "ci95_delta_lo",
        "ci95_delta_hi",
        "ci_excludes_zero",
    ]
    print("[capacity sensitivity] Direct paired cap effects")
    print(headline_effects[columns].to_string(index=False))
    interaction_columns = [
        "scenario",
        "metric",
        "mean_reference_scenario_effect",
        "mean_sensitivity_scenario_effect",
        "mean_interaction_delta",
        "ci95_interaction_lo",
        "ci95_interaction_hi",
        "ci_excludes_zero",
    ]
    print("\n[capacity sensitivity] Cap x scenario interactions")
    print(headline_interactions[interaction_columns].to_string(index=False))
    print(f"\n[capacity sensitivity] COMPLETE -> {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
