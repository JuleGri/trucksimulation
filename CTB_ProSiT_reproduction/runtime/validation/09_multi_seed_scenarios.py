"""Matched-seed validation of the final calibrated CTB what-if scenarios.

The authoritative, held-out-validated parameter bundle is frozen as the
baseline.  Two transparent white-box interventions are derived from it:

1. ``t22_closed`` removes T22 from every activity/resource mapping and makes
   its calendar unavailable.
2. ``demand_plus_20pct`` increases the arrival intensity by 20 percent by
   dividing every empirical inter-arrival sample by 1.20.  Its zero mass and
   empirical distribution shape are retained.

All other discovered and CTB-calibrated parameters remain unchanged.  Every
resource eligible for an RMG activity is capped at maximum concurrency three
before deriving all three scenario templates.  This is the final domain
constraint: CTB has three RMG cranes and three truck waiting lanes per block,
whereas timestamp overlap can overestimate physical simultaneous capacity.
The freely discovered source bundle is retained unchanged for diagnosis.  For
each seed, the effective baseline and both interventions are
simulated from the same start time with the same trace count and random seed.
Deltas are therefore paired by seed.  Every generated log must satisfy the
sequential CTB case contract.

Outputs under ``validation/results/<label>/``:

* ``scenario_replications.csv``: one row per seed and scenario;
* ``scenario_kpi_summary.csv``: scenario-level means and 95% t intervals;
* ``scenario_paired_deltas.csv``: intervention minus baseline per seed;
* ``scenario_paired_delta_summary.csv``: paired effects and 95% t intervals;
* ``scenario_contracts.csv``: contract audit for all 30 generated logs;
* ``scenario_parameter_changes.json``: explicit white-box parameter diff;
* ``params_*.pkl``: frozen derived parameter bundles, including the effective
  baseline when a concurrency cap is requested;
* ``t22_receive_delivery_summary.csv``: focused T22 effects by operation type;
* ``figures/scenario_paired_deltas_ci.png``: paired-effect plot;
* ``figures/t22_receive_delivery_deltas_ci.png``: focused T22 plot; and
* ``scenario_run_summary.json``: reproducibility metadata.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import pickle
from pathlib import Path
import sys
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from _eventlog_contract import (  # noqa: E402
    ACT_COL,
    CASE_COL,
    GATE_IN,
    ORDER_COL,
    eventlog_contract_report,
)
from _prosit_ctb_calibration import simulate_ctb  # noqa: E402


DEFAULT_PARAMS = (
    REPO_ROOT
    / "baseline"
    / "discovery_params"
    / "params_20260816_214403_train80"
    / "prosit_discovery_workload_sequential_calibrated"
    / "prosit_params.pkl"
)
DEFAULT_MANIFEST = REPO_ROOT / "validation" / "results" / "split_manifest.json"
DEFAULT_OUT_ROOT = REPO_ROOT / "validation" / "results"
DEFAULT_LABEL = "prosit_sequential_calibrated_scenarios_rmg_cap3_ci"

RMG_ACTIVITIES = ("RMG_receive", "RMG_delivery", "RMG_mixed")
DEFAULT_BLOCKED_BLOCKS = ("T22",)
DEFAULT_DEMAND_INCREASE_PCT = 20.0
DEFAULT_RMG_MAX_CONCURRENCY = 3

SCENARIO_BASELINE = "baseline"
SCENARIO_T22 = "t22_closed"
SCENARIO_DEMAND = "demand_plus_20pct"
SCENARIO_ORDER = (SCENARIO_BASELINE, SCENARIO_T22, SCENARIO_DEMAND)

CONTRACT_KEYS = (
    "gate_only_cases",
    "wrong_case_boundary_cases",
    "within_case_overlap_cases",
    "decreasing_completion_cases",
    "gate_out_before_final_yard_cases",
)
HARD_CONTRACT_KEYS = tuple(key for key in CONTRACT_KEYS if key != "gate_only_cases")

KPI_COLUMNS = (
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
    "rmg_receive_events",
    "mean_rmg_receive_service_min",
    "median_rmg_receive_service_min",
    "p90_rmg_receive_service_min",
    "mean_rmg_receive_pre_service_min",
    "median_rmg_receive_pre_service_min",
    "p90_rmg_receive_pre_service_min",
    "rmg_delivery_events",
    "mean_rmg_delivery_service_min",
    "median_rmg_delivery_service_min",
    "p90_rmg_delivery_service_min",
    "mean_rmg_delivery_pre_service_min",
    "median_rmg_delivery_pre_service_min",
    "p90_rmg_delivery_pre_service_min",
    "mean_inter_arrival_min",
    "zero_inter_arrival_share",
    "arrival_rate_per_elapsed_hour",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--n-traces", type=int, default=None)
    parser.add_argument(
        "--t-start",
        default=None,
        help="ISO start timestamp; defaults to cutoff_arrival_ts in the manifest.",
    )
    parser.add_argument(
        "--blocked-blocks",
        nargs="+",
        default=list(DEFAULT_BLOCKED_BLOCKS),
    )
    parser.add_argument(
        "--demand-increase-pct",
        type=float,
        default=DEFAULT_DEMAND_INCREASE_PCT,
        help=(
            "Demand increase relative to baseline. The default 20 divides "
            "all inter-arrival samples by 1.20."
        ),
    )
    parser.add_argument(
        "--rmg-max-concurrency",
        type=int,
        default=DEFAULT_RMG_MAX_CONCURRENCY,
        help=(
            "Hard domain cap applied to max_concurrency for every RMG resource "
            "before all three templates are derived (default: 3). The freely "
            "discovered source pickle is never modified."
        ),
    )
    parser.add_argument(
        "--no-rmg-cap",
        action="store_true",
        help="Diagnostic only: reproduce the freely discovered capacities without the cap.",
    )
    parser.add_argument(
        "--timestamp-resolution",
        choices=("minute", "native"),
        default="minute",
    )
    parser.add_argument(
        "--save-seed-logs",
        action="store_true",
        help="Also persist all 30 event logs (disabled by default due to size).",
    )
    args = parser.parse_args()
    if args.no_rmg_cap:
        args.rmg_max_concurrency = None
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_params(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Parameter pickle not found: {path}")
    with path.open("rb") as handle:
        params = pickle.load(handle)
    calibration = getattr(params, "ctb_calibration", None)
    if not calibration or not calibration.get("holdout_used") is False:
        raise ValueError(
            "Scenario baseline must be the training-only CTB-calibrated parameter bundle."
        )
    return params


def _resolve_window(
    manifest_path: Path,
    n_traces_override: int | None,
    t_start_override: str | None,
) -> tuple[int, datetime | None, dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    n_traces = n_traces_override
    if n_traces is None:
        n_traces = int(manifest["test"]["n_cases"])
    cutoff = t_start_override or manifest.get("cutoff_arrival_ts")
    t_start = None
    if cutoff:
        t_start = datetime.fromisoformat(str(cutoff).replace("Z", "+00:00"))
    return int(n_traces), t_start, manifest


def _closed_calendar() -> dict[int, dict[int, bool]]:
    return {day: {hour: False for hour in range(24)} for day in range(7)}


def _resource_mapping(params, activity: str) -> list[str]:
    if activity not in params.act_to_resources:
        raise KeyError(f"Activity {activity!r} has no resource mapping.")
    return list(params.act_to_resources[activity])


def _rmg_resource_names(params) -> list[str]:
    """Return the sorted union of resources eligible for an RMG activity."""

    resources = {
        resource
        for activity in RMG_ACTIVITIES
        for resource in _resource_mapping(params, activity)
    }
    if not resources:
        raise ValueError("The model has no resources eligible for RMG activities.")
    return sorted(resources)


def _rmg_concurrency_summary(params) -> dict:
    """Return a JSON-serializable summary of the effective RMG capacities."""

    resources = _rmg_resource_names(params)
    missing = [
        resource
        for resource in resources
        if resource not in getattr(params, "max_concurrency", {})
    ]
    if missing:
        raise KeyError(f"RMG resources lack max_concurrency values: {missing}")
    per_resource = {
        resource: int(params.max_concurrency[resource])
        for resource in resources
    }
    if any(value < 1 for value in per_resource.values()):
        raise ValueError(f"RMG maximum concurrency must be positive: {per_resource}")
    return {
        "n_resources": len(resources),
        "per_resource": per_resource,
        "minimum": min(per_resource.values()),
        "maximum": max(per_resource.values()),
        "aggregate": sum(per_resource.values()),
    }


def apply_rmg_concurrency_cap(params, cap: int):
    """Return a deep-copied bundle with RMG maximum concurrency clamped.

    The intervention changes only ``max_concurrency`` entries of resources
    reachable from an RMG activity. Existing values below the requested cap
    are retained. Resource mappings, weights, calendars and all other model
    families remain byte-for-byte represented by the deep copy.
    """

    if cap < 1:
        raise ValueError("The RMG maximum-concurrency cap must be at least one.")
    scenario = deepcopy(params)
    before = _rmg_concurrency_summary(scenario)
    for resource, discovered in before["per_resource"].items():
        scenario.max_concurrency[resource] = min(int(discovered), int(cap))
    after = _rmg_concurrency_summary(scenario)
    if after["maximum"] > cap:
        raise AssertionError("RMG maximum-concurrency cap was not applied completely.")
    return scenario


def apply_t22_closure(params, blocked_blocks: Iterable[str]):
    """Return a deep-copied bundle with only the declared block closure."""

    scenario = deepcopy(params)
    blocked = set(blocked_blocks)
    if not blocked:
        raise ValueError("At least one block must be declared unavailable.")

    seen = {resource for values in scenario.act_to_resources.values() for resource in values}
    unknown = blocked.difference(seen).difference(scenario.calendars)
    if unknown:
        raise ValueError(f"Blocked resources are unknown to the model: {sorted(unknown)}")

    for activity, resources in list(scenario.act_to_resources.items()):
        filtered = [resource for resource in resources if resource not in blocked]
        if resources and not filtered:
            raise ValueError(
                f"Closing {sorted(blocked)} leaves activity {activity!r} without a resource."
            )
        scenario.act_to_resources[activity] = filtered

    for resource in blocked:
        if resource in scenario.calendars:
            scenario.calendars[resource] = _closed_calendar()
    return scenario


def _arrival_model_summary(params) -> dict:
    """Return auditable statistics of the unconditional empirical arrival model."""

    model = getattr(params, "arrival_time_distribution", None)
    rules = getattr(model, "rules", None)
    if not isinstance(rules, dict) or set(rules) != {0}:
        raise ValueError(
            "The demand intervention requires the calibrated unconditional "
            "arrival rule with key 0."
        )
    samples = np.asarray(rules[0].get("sampled", []), dtype=float)
    if samples.size == 0 or not np.isfinite(samples).all() or (samples < 0.0).any():
        raise ValueError("The empirical arrival sample is empty, invalid, or negative.")
    mean = float(samples.mean())
    return {
        "n_samples": int(samples.size),
        "mean_working_inter_arrival_min": mean,
        "implied_arrivals_per_working_hour": (
            float(60.0 / mean) if mean > 0.0 else float("inf")
        ),
        "zero_probability": float(np.mean(samples == 0.0)),
        "minimum_working_inter_arrival_min": float(samples.min()),
        "maximum_working_inter_arrival_min": float(samples.max()),
    }


def apply_demand_increase(params, demand_increase_pct: float):
    """Return a deep-copied bundle with a higher arrival intensity only.

    If the arrival intensity is multiplied by ``m``, inter-arrival times must
    be divided by ``m``.  Scaling the empirical samples (including exact
    zeros) preserves their shape and simultaneous-arrival probability.  The
    gate admission calendar remains unchanged.
    """

    if not np.isfinite(demand_increase_pct) or demand_increase_pct <= 0.0:
        raise ValueError("The demand increase must be a positive finite percentage.")
    demand_multiplier = 1.0 + float(demand_increase_pct) / 100.0
    inter_arrival_scale = 1.0 / demand_multiplier

    scenario = deepcopy(params)
    before = _arrival_model_summary(scenario)
    rule = scenario.arrival_time_distribution.rules[0]
    scaled = [float(value) * inter_arrival_scale for value in rule["sampled"]]
    scaled_mean = float(np.mean(scaled))
    rule["sampled"] = scaled
    rule["value"] = scaled_mean
    # ``sampled`` is ProSiT's runtime source. Keep its portable fallback
    # internally consistent for serialization and independent inspection.
    rule["dist"] = (
        "fixed",
        (scaled_mean,),
        float(min(scaled)),
        float(max(scaled)),
    )
    after = _arrival_model_summary(scenario)
    if not np.isclose(
        after["implied_arrivals_per_working_hour"]
        / before["implied_arrivals_per_working_hour"],
        demand_multiplier,
    ):
        raise AssertionError("Demand scaling did not produce the requested arrival rate.")
    return scenario


def _mapping_diff(baseline, scenario) -> dict[str, dict[str, list[str]]]:
    changes: dict[str, dict[str, list[str]]] = {}
    activities = sorted(set(baseline.act_to_resources) | set(scenario.act_to_resources))
    for activity in activities:
        before = list(baseline.act_to_resources.get(activity, []))
        after = list(scenario.act_to_resources.get(activity, []))
        if before != after:
            changes[activity] = {
                "before": before,
                "after": after,
                "removed": [resource for resource in before if resource not in after],
                "added": [resource for resource in after if resource not in before],
            }
    return changes


def _calendar_active_slots(params, resource: str) -> int | None:
    calendar = params.calendars.get(resource)
    if calendar is None:
        return None
    return int(sum(bool(active) for hours in calendar.values() for active in hours.values()))


def _parameter_change_report(
    source_baseline,
    baseline,
    scenario_a,
    scenario_b,
    *,
    blocked_blocks: list[str],
    demand_increase_pct: float,
    rmg_max_concurrency: int | None,
) -> dict:
    before_arrivals = _arrival_model_summary(baseline)
    after_arrivals = _arrival_model_summary(scenario_b)
    demand_multiplier = 1.0 + demand_increase_pct / 100.0
    source_rmg = _rmg_concurrency_summary(source_baseline)
    effective_rmg = _rmg_concurrency_summary(baseline)
    return {
        "principle": (
            "The effective baseline is derived by deep copy from the frozen calibrated "
            "source bundle. The default RMG cap of three is a final domain constraint, "
            "while --no-rmg-cap retains the discovered overlap capacities only for "
            "diagnosis. Both operational "
            "scenarios are then independently derived from that effective baseline. "
            "Scenario A changes only the listed resource mappings and blocked-resource "
            "calendar. Scenario B changes only the empirical inter-arrival samples and "
            "their serialized fallback statistics."
        ),
        "rmg_capacity_constraint": {
            "enabled": rmg_max_concurrency is not None,
            "requested_max_concurrency": rmg_max_concurrency,
            "before": source_rmg,
            "after": effective_rmg,
            "changed_resources": {
                resource: {
                    "before": source_rmg["per_resource"][resource],
                    "after": effective_rmg["per_resource"][resource],
                }
                for resource in source_rmg["per_resource"]
                if source_rmg["per_resource"][resource]
                != effective_rmg["per_resource"][resource]
            },
            "unchanged_parameter_families": [
                "control-flow Petri net",
                "arrival model and gate admission calendar",
                "activity-duration and waiting-time models",
                "routing decision models",
                "activity/resource mappings",
                "resource calendars, selection models, and weights",
                "case attributes and feature schema",
                "CTB minute timestamp observation model",
            ],
        },
        "blocked_blocks": blocked_blocks,
        "t22_closed": {
            "interpretation": (
                "Resource-pool reallocation experiment: T22 is removed from the "
                "eligible RMG pools. Receive and delivery effects are reported "
                "separately; no physical container relocation is simulated."
            ),
            "act_to_resources_changes": _mapping_diff(baseline, scenario_a),
            "calendar_active_slots_before": {
                block: _calendar_active_slots(baseline, block)
                for block in blocked_blocks
            },
            "calendar_active_slots_after": {
                block: _calendar_active_slots(scenario_a, block)
                for block in blocked_blocks
            },
        },
        "demand_plus_20pct": {
            "requested_demand_increase_pct": float(demand_increase_pct),
            "arrival_intensity_multiplier": float(demand_multiplier),
            "inter_arrival_time_multiplier": float(1.0 / demand_multiplier),
            "arrival_model_before": before_arrivals,
            "arrival_model_after": after_arrivals,
            "act_to_resources_changes": _mapping_diff(baseline, scenario_b),
            "arrival_calendar_changed": baseline.arrival_calendar != scenario_b.arrival_calendar,
            "resource_calendars_changed": baseline.calendars != scenario_b.calendars,
        },
        "scenario_specific_unchanged_parameter_families": {
            "t22_closed": [
                "control-flow Petri net",
                "arrival model and gate admission calendar",
                "activity-duration models",
                "routing decision models",
                "resource-selection models and weights",
                "case attributes and feature schema",
                "CTB minute timestamp observation model",
            ],
            "demand_plus_20pct": [
                "control-flow Petri net",
                "gate admission calendar",
                "activity-duration models",
                "routing decision models",
                "activity/resource mappings",
                "resource calendars, selection models, and weights",
                "case attributes and feature schema",
                "CTB minute timestamp observation model",
            ],
        },
    }


def _prepare_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[:, ~df.columns.duplicated()].copy()
    for column in ("enabled:timestamp", "start:timestamp", "time:timestamp"):
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


def _summarize_simulation(df: pd.DataFrame, *, seed: int, scenario: str) -> dict:
    prepared = _prepare_timestamps(df)
    if not {CASE_COL, ACT_COL, "start:timestamp", "time:timestamp"}.issubset(prepared.columns):
        raise ValueError("Simulated log is missing case, activity, or lifecycle timestamps.")

    prepared["service_time_min"] = (
        prepared["time:timestamp"] - prepared["start:timestamp"]
    ).dt.total_seconds() / 60.0
    if "enabled:timestamp" in prepared.columns:
        prepared["pre_service_time_min"] = (
            prepared["start:timestamp"] - prepared["enabled:timestamp"]
        ).dt.total_seconds() / 60.0
        prepared["pre_service_time_min"] = prepared["pre_service_time_min"].clip(lower=0.0)
    else:
        prepared["pre_service_time_min"] = np.nan

    grouped = prepared.groupby(CASE_COL, sort=False)
    turnaround = (
        grouped["time:timestamp"].max() - grouped["start:timestamp"].min()
    ).dt.total_seconds() / 60.0
    rmg = prepared[prepared[ACT_COL].isin(RMG_ACTIVITIES)]
    rmg_receive = prepared[prepared[ACT_COL].eq("RMG_receive")]
    rmg_delivery = prepared[prepared[ACT_COL].eq("RMG_delivery")]
    gate_in_starts = (
        prepared.loc[prepared[ACT_COL].eq(GATE_IN), "start:timestamp"]
        .dropna()
        .sort_values()
    )
    observed_inter_arrivals = (
        gate_in_starts.diff().dt.total_seconds().div(60.0).dropna()
    )
    arrival_span_min = (
        float((gate_in_starts.iloc[-1] - gate_in_starts.iloc[0]).total_seconds() / 60.0)
        if len(gate_in_starts) > 1
        else float("nan")
    )
    arrival_rate_per_elapsed_hour = (
        float(60.0 * (len(gate_in_starts) - 1) / arrival_span_min)
        if arrival_span_min > 0.0
        else float("nan")
    )

    def _stat(series: pd.Series, kind: str) -> float:
        values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            return float("nan")
        if kind == "mean":
            return float(values.mean())
        if kind == "median":
            return float(values.median())
        if kind == "p90":
            return float(values.quantile(0.9))
        raise KeyError(kind)

    return {
        "seed": int(seed),
        "scenario": scenario,
        "sim_cases": int(prepared[CASE_COL].nunique()),
        "sim_events": int(len(prepared)),
        "rmg_events": int(len(rmg)),
        "mean_turnaround_min": _stat(turnaround, "mean"),
        "median_turnaround_min": _stat(turnaround, "median"),
        "p90_turnaround_min": _stat(turnaround, "p90"),
        "max_turnaround_min": float(turnaround.max()),
        "mean_rmg_service_min": _stat(rmg["service_time_min"], "mean"),
        "median_rmg_service_min": _stat(rmg["service_time_min"], "median"),
        "p90_rmg_service_min": _stat(rmg["service_time_min"], "p90"),
        "mean_rmg_pre_service_min": _stat(rmg["pre_service_time_min"], "mean"),
        "median_rmg_pre_service_min": _stat(rmg["pre_service_time_min"], "median"),
        "p90_rmg_pre_service_min": _stat(rmg["pre_service_time_min"], "p90"),
        "rmg_receive_events": int(len(rmg_receive)),
        "mean_rmg_receive_service_min": _stat(rmg_receive["service_time_min"], "mean"),
        "median_rmg_receive_service_min": _stat(rmg_receive["service_time_min"], "median"),
        "p90_rmg_receive_service_min": _stat(rmg_receive["service_time_min"], "p90"),
        "mean_rmg_receive_pre_service_min": _stat(rmg_receive["pre_service_time_min"], "mean"),
        "median_rmg_receive_pre_service_min": _stat(rmg_receive["pre_service_time_min"], "median"),
        "p90_rmg_receive_pre_service_min": _stat(rmg_receive["pre_service_time_min"], "p90"),
        "rmg_delivery_events": int(len(rmg_delivery)),
        "mean_rmg_delivery_service_min": _stat(rmg_delivery["service_time_min"], "mean"),
        "median_rmg_delivery_service_min": _stat(rmg_delivery["service_time_min"], "median"),
        "p90_rmg_delivery_service_min": _stat(rmg_delivery["service_time_min"], "p90"),
        "mean_rmg_delivery_pre_service_min": _stat(rmg_delivery["pre_service_time_min"], "mean"),
        "median_rmg_delivery_pre_service_min": _stat(rmg_delivery["pre_service_time_min"], "median"),
        "p90_rmg_delivery_pre_service_min": _stat(rmg_delivery["pre_service_time_min"], "p90"),
        "mean_inter_arrival_min": _stat(observed_inter_arrivals, "mean"),
        "zero_inter_arrival_share": float(
            np.mean(observed_inter_arrivals.to_numpy() == 0.0)
        ),
        "arrival_span_min": arrival_span_min,
        "arrival_rate_per_elapsed_hour": arrival_rate_per_elapsed_hour,
        "service_events_above_24h": int((prepared["service_time_min"] > 1440.0).sum()),
        "turnaround_cases_above_24h": int((turnaround > 1440.0).sum()),
    }


def _contract_audit(
    df: pd.DataFrame,
    *,
    seed: int,
    scenario: str,
    n_traces: int,
    blocked_blocks: set[str],
) -> dict:
    prepared = _prepare_timestamps(df)
    prepared[ORDER_COL] = prepared.groupby(CASE_COL, sort=False).cumcount()
    report = eventlog_contract_report(prepared, _already_ordered=True)
    result = {"seed": int(seed), "scenario": scenario}
    result.update({key: int(report.get(key, 0)) for key in CONTRACT_KEYS})

    result["wrong_case_count"] = abs(int(prepared[CASE_COL].nunique()) - int(n_traces))
    resource = prepared.get("org:resource", pd.Series("", index=prepared.index)).astype(str)
    t22_assignments = int(resource.isin(blocked_blocks).sum())
    result["t22_resource_assignments"] = t22_assignments
    result["prohibited_resource_assignments"] = (
        t22_assignments if scenario == SCENARIO_T22 else 0
    )

    return result


def _assert_audit(audit: dict, metrics: dict) -> None:
    violations = sum(int(audit[key]) for key in HARD_CONTRACT_KEYS)
    violations += int(audit["wrong_case_count"])
    violations += int(audit["prohibited_resource_assignments"])
    violations += int(metrics["service_events_above_24h"])
    violations += int(metrics["turnaround_cases_above_24h"])
    if violations:
        raise RuntimeError(
            f"Seed {audit['seed']} scenario {audit['scenario']} failed validation: "
            f"audit={audit}, duration_tail={{'service': "
            f"{metrics['service_events_above_24h']}, 'turnaround': "
            f"{metrics['turnaround_cases_above_24h']}}}"
        )


def _ci(values: pd.Series) -> tuple[int, float, float, float, float]:
    samples = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(samples))
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(samples.mean())
    if n == 1:
        return n, mean, float("nan"), mean, mean
    std = float(samples.std(ddof=1))
    half = float(stats.t.ppf(0.975, df=n - 1) * std / np.sqrt(n))
    return n, mean, std, mean - half, mean + half


def _scenario_kpi_summary(replications: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIO_ORDER:
        group = replications[replications["scenario"].eq(scenario)]
        for metric in KPI_COLUMNS:
            n, mean, std, lo, hi = _ci(group[metric])
            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "half_width": (hi - lo) / 2.0,
                }
            )
    return pd.DataFrame(rows)


def _paired_deltas(replications: pd.DataFrame) -> pd.DataFrame:
    indexed = replications.set_index(["seed", "scenario"])
    rows = []
    for seed in sorted(replications["seed"].unique()):
        baseline = indexed.loc[(seed, SCENARIO_BASELINE)]
        for scenario in (SCENARIO_T22, SCENARIO_DEMAND):
            intervention = indexed.loc[(seed, scenario)]
            row = {"seed": int(seed), "scenario": scenario}
            for metric in KPI_COLUMNS:
                base_value = float(baseline[metric])
                scenario_value = float(intervention[metric])
                row[f"baseline_{metric}"] = base_value
                row[f"scenario_{metric}"] = scenario_value
                row[f"delta_{metric}"] = scenario_value - base_value
                row[f"delta_pct_{metric}"] = (
                    100.0 * (scenario_value - base_value) / base_value
                    if base_value != 0.0 else float("nan")
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _paired_delta_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in (SCENARIO_T22, SCENARIO_DEMAND):
        group = deltas[deltas["scenario"].eq(scenario)]
        for metric in KPI_COLUMNS:
            n, mean, std, lo, hi = _ci(group[f"delta_{metric}"])
            _, pct_mean, pct_std, pct_lo, pct_hi = _ci(group[f"delta_pct_{metric}"])
            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "n": n,
                    "mean_delta": mean,
                    "std_delta": std,
                    "ci95_delta_lo": lo,
                    "ci95_delta_hi": hi,
                    "mean_delta_pct": pct_mean,
                    "std_delta_pct": pct_std,
                    "ci95_delta_pct_lo": pct_lo,
                    "ci95_delta_pct_hi": pct_hi,
                    "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
                }
            )
    return pd.DataFrame(rows)


def _plot_paired_deltas(summary: pd.DataFrame, out_path: Path) -> None:
    metrics = (
        ("mean_turnaround_min", "Mean turnaround delta (min)"),
        ("mean_rmg_service_min", "Mean RMG service delta (min)"),
        ("mean_rmg_pre_service_min", "Mean RMG pre-service delta (min)"),
        ("arrival_rate_per_elapsed_hour", "Arrival-rate delta (trucks/hour)"),
    )
    labels = {
        SCENARIO_T22: "T22 closed",
        SCENARIO_DEMAND: "+20% demand",
    }
    colors = {SCENARIO_T22: "#1f77b4", SCENARIO_DEMAND: "#d62728"}

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, metrics):
        subset = summary[summary["metric"].eq(metric)].set_index("scenario")
        scenarios = [SCENARIO_T22, SCENARIO_DEMAND]
        means = np.array([float(subset.loc[s, "mean_delta"]) for s in scenarios])
        lo = np.array([float(subset.loc[s, "ci95_delta_lo"]) for s in scenarios])
        hi = np.array([float(subset.loc[s, "ci95_delta_hi"]) for s in scenarios])
        x = np.arange(len(scenarios))
        ax.bar(
            x,
            means,
            yerr=np.vstack([means - lo, hi - means]),
            capsize=6,
            color=[colors[s] for s in scenarios],
            alpha=0.85,
            edgecolor="black",
        )
        ax.axhline(0.0, color="black", linewidth=0.9)
        for xi, mean, lower, upper in zip(x, means, lo, hi):
            anchor = upper if mean >= 0 else lower
            ax.text(xi, anchor, f"{mean:+.2f}\n[{lower:+.2f}, {upper:+.2f}]", ha="center", va="bottom" if mean >= 0 else "top", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[s] for s in scenarios], rotation=12, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    fig.suptitle("Matched-seed CTB scenario effects (n=10; 95% paired t intervals)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_t22_receive_delivery(summary: pd.DataFrame, out_path: Path) -> None:
    """Plot the matched T22 effect separately for receive and delivery work."""

    metrics = (
        ("mean_rmg_receive_service_min", "Receive service-time delta (min)"),
        ("mean_rmg_delivery_service_min", "Delivery service-time delta (min)"),
        ("mean_rmg_receive_pre_service_min", "Receive pre-service delta (min)"),
        ("mean_rmg_delivery_pre_service_min", "Delivery pre-service delta (min)"),
    )
    focused = summary[summary["scenario"].eq(SCENARIO_T22)].set_index("metric")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        row = focused.loc[metric]
        mean = float(row["mean_delta"])
        lower = float(row["ci95_delta_lo"])
        upper = float(row["ci95_delta_hi"])
        ax.bar(
            [0],
            [mean],
            yerr=np.array([[mean - lower], [upper - mean]]),
            capsize=7,
            color="#1f77b4",
            alpha=0.85,
            edgecolor="black",
            width=0.55,
        )
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.text(
            0.5,
            0.96,
            f"delta {mean:+.3f} min\n95% CI [{lower:+.3f}, {upper:+.3f}]",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
        ax.set_xticks([])
        ax.margins(y=0.28)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    fig.suptitle(
        "T22 resource-pool reallocation: operation-specific effects\n"
        "(n=10 matched seeds; 95% paired t intervals)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_pickle(params, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(params, handle)


def main() -> int:
    args = parse_args()
    if args.n_seeds < 2:
        raise SystemExit("--n-seeds must be at least 2 for a confidence interval.")

    params_path = args.params.resolve()
    manifest_path = args.manifest.resolve()
    out_dir = (args.out_root / args.label).resolve()
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if args.save_seed_logs:
        (out_dir / "logs").mkdir(exist_ok=True)

    n_traces, t_start, manifest = _resolve_window(
        manifest_path,
        args.n_traces,
        args.t_start,
    )
    blocked_blocks = list(dict.fromkeys(args.blocked_blocks))

    print(f"[scenario] Loading frozen baseline: {params_path}")
    source_baseline = _load_params(params_path)
    baseline = (
        apply_rmg_concurrency_cap(source_baseline, args.rmg_max_concurrency)
        if args.rmg_max_concurrency is not None
        else source_baseline
    )
    scenario_a = apply_t22_closure(baseline, blocked_blocks)
    scenario_b = apply_demand_increase(baseline, args.demand_increase_pct)
    templates = {
        SCENARIO_BASELINE: baseline,
        SCENARIO_T22: scenario_a,
        SCENARIO_DEMAND: scenario_b,
    }

    baseline_path = params_path
    if args.rmg_max_concurrency is not None:
        baseline_path = (
            out_dir
            / f"params_baseline_rmg_max_concurrency_{args.rmg_max_concurrency}.pkl"
        )
        _write_pickle(baseline, baseline_path)
    scenario_a_path = out_dir / "params_t22_closed.pkl"
    scenario_b_path = out_dir / "params_demand_plus_20pct.pkl"
    _write_pickle(scenario_a, scenario_a_path)
    _write_pickle(scenario_b, scenario_b_path)

    parameter_report = _parameter_change_report(
        source_baseline,
        baseline,
        scenario_a,
        scenario_b,
        blocked_blocks=blocked_blocks,
        demand_increase_pct=args.demand_increase_pct,
        rmg_max_concurrency=args.rmg_max_concurrency,
    )
    parameter_report.update(
        {
            "source_baseline_pickle": str(params_path),
            "source_baseline_sha256": _sha256(params_path),
            "baseline_pickle": str(baseline_path),
            "baseline_sha256": _sha256(baseline_path),
            "scenario_pickles": {
                SCENARIO_T22: str(scenario_a_path),
                SCENARIO_DEMAND: str(scenario_b_path),
            },
            "scenario_sha256": {
                SCENARIO_T22: _sha256(scenario_a_path),
                SCENARIO_DEMAND: _sha256(scenario_b_path),
            },
        }
    )
    with (out_dir / "scenario_parameter_changes.json").open("w", encoding="utf-8") as handle:
        json.dump(parameter_report, handle, indent=2)

    run_summary_path = out_dir / "scenario_run_summary.json"
    run_summary = {
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "source_baseline_pickle": str(params_path),
        "source_baseline_sha256": parameter_report["source_baseline_sha256"],
        "baseline_pickle": str(baseline_path),
        "baseline_sha256": parameter_report["baseline_sha256"],
        "training_only_calibration": True,
        "holdout_used_for_parameter_changes": False,
        "manifest": str(manifest_path),
        "manifest_cutoff": manifest.get("cutoff_arrival_ts"),
        "n_traces": n_traces,
        "t_start": None if t_start is None else t_start.isoformat(),
        "seeds": list(range(args.base_seed, args.base_seed + args.n_seeds)),
        "timestamp_resolution": args.timestamp_resolution,
        "paired_common_random_numbers": True,
        "demand_increase_pct": float(args.demand_increase_pct),
        "rmg_max_concurrency_cap": args.rmg_max_concurrency,
        "rmg_concurrency_before": parameter_report["rmg_capacity_constraint"]["before"],
        "rmg_concurrency_after": parameter_report["rmg_capacity_constraint"]["after"],
        "saved_event_logs": bool(args.save_seed_logs),
    }
    with run_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)

    print(
        f"[scenario] n_seeds={args.n_seeds} n_traces={n_traces:,} "
        f"t_start={t_start} rmg_cap={args.rmg_max_concurrency} "
        f"scenarios={list(templates)}"
    )
    rows: list[dict] = []
    audits: list[dict] = []
    for seed in range(args.base_seed, args.base_seed + args.n_seeds):
        print(f"[scenario] seed={seed}")
        for scenario_name in SCENARIO_ORDER:
            print(f"  {scenario_name} ...", end=" ", flush=True)
            simulated = simulate_ctb(
                deepcopy(templates[scenario_name]),
                n_traces=n_traces,
                t_start=t_start,
                seed=seed,
                timestamp_resolution=(
                    "min" if args.timestamp_resolution == "minute" else None
                ),
            )
            metrics = _summarize_simulation(
                simulated,
                seed=seed,
                scenario=scenario_name,
            )
            audit = _contract_audit(
                simulated,
                seed=seed,
                scenario=scenario_name,
                n_traces=n_traces,
                blocked_blocks=set(blocked_blocks),
            )
            _assert_audit(audit, metrics)
            rows.append(metrics)
            audits.append(audit)
            if args.save_seed_logs:
                simulated.to_csv(
                    out_dir / "logs" / f"seed_{seed}_{scenario_name}.csv",
                    index=False,
                )
            print(
                f"turnaround={metrics['mean_turnaround_min']:.3f} min "
                f"RMG-service={metrics['mean_rmg_service_min']:.3f} min "
                f"contract=PASS"
            )

    replications = pd.DataFrame(rows)
    contracts = pd.DataFrame(audits)
    deltas = _paired_deltas(replications)
    kpi_summary = _scenario_kpi_summary(replications)
    delta_summary = _paired_delta_summary(deltas)

    replications.to_csv(out_dir / "scenario_replications.csv", index=False)
    contracts.to_csv(out_dir / "scenario_contracts.csv", index=False)
    deltas.to_csv(out_dir / "scenario_paired_deltas.csv", index=False)
    kpi_summary.to_csv(out_dir / "scenario_kpi_summary.csv", index=False)
    delta_summary.to_csv(out_dir / "scenario_paired_delta_summary.csv", index=False)
    t22_metrics = [
        "mean_rmg_receive_service_min",
        "mean_rmg_delivery_service_min",
        "mean_rmg_receive_pre_service_min",
        "mean_rmg_delivery_pre_service_min",
    ]
    t22_focus = delta_summary[
        delta_summary["scenario"].eq(SCENARIO_T22)
        & delta_summary["metric"].isin(t22_metrics)
    ].copy()
    t22_focus.to_csv(out_dir / "t22_receive_delivery_summary.csv", index=False)
    _plot_paired_deltas(
        delta_summary,
        figures_dir / "scenario_paired_deltas_ci.png",
    )
    _plot_t22_receive_delivery(
        delta_summary,
        figures_dir / "t22_receive_delivery_deltas_ci.png",
    )

    headline = delta_summary[
        delta_summary["metric"].isin(
            [
                "mean_turnaround_min",
                "mean_rmg_service_min",
                "mean_rmg_pre_service_min",
                "arrival_rate_per_elapsed_hour",
            ]
        )
    ].copy()
    run_summary.update(
        {
            "status": "completed",
            "finished_at": datetime.now().astimezone().isoformat(),
            "all_contracts_passed": True,
            "all_duration_tail_counts_zero": True,
            "headline_paired_effects": headline.to_dict(orient="records"),
            "t22_receive_delivery_effects": t22_focus.to_dict(orient="records"),
            "artifacts": {
                "replications": str(out_dir / "scenario_replications.csv"),
                "contracts": str(out_dir / "scenario_contracts.csv"),
                "paired_deltas": str(out_dir / "scenario_paired_deltas.csv"),
                "kpi_summary": str(out_dir / "scenario_kpi_summary.csv"),
                "paired_delta_summary": str(out_dir / "scenario_paired_delta_summary.csv"),
                "figure": str(figures_dir / "scenario_paired_deltas_ci.png"),
                "t22_receive_delivery_summary": str(
                    out_dir / "t22_receive_delivery_summary.csv"
                ),
                "t22_receive_delivery_figure": str(
                    figures_dir / "t22_receive_delivery_deltas_ci.png"
                ),
                "parameter_changes": str(out_dir / "scenario_parameter_changes.json"),
            },
        }
    )
    with run_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)

    print("\n[scenario] === Paired headline effects ===")
    print(
        headline[
            [
                "scenario",
                "metric",
                "mean_delta",
                "ci95_delta_lo",
                "ci95_delta_hi",
                "mean_delta_pct",
                "ci_excludes_zero",
            ]
        ].to_string(index=False)
    )
    print("\n[scenario] === T22 receive/delivery effects ===")
    print(
        t22_focus[
            [
                "metric",
                "mean_delta",
                "ci95_delta_lo",
                "ci95_delta_hi",
                "mean_delta_pct",
                "ci_excludes_zero",
            ]
        ].to_string(index=False)
    )
    print(f"[scenario] COMPLETE -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
