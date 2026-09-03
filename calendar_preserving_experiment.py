"""Calendar-preserving CTB demand-intensity experiment.

ProSiT's calibrated arrival samples are measured in working minutes.  For a
factor m this script divides those samples by m and admits round(N*m) cases.
The expected working-time horizon therefore stays constant while the gate
calendar continues to prohibit Sunday/night arrivals.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr
from copy import deepcopy
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent / "CTB_ProSiT_reproduction"
sys.path.insert(0, str(REPO))
import reproduce  # noqa: E402


MULTIPLIERS = (1.0, 1.2, 1.5, 2.0, 2.4, 3.0)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "experimental_results" / "calendar_preserving_full"


def _calendar_audit(log: pd.DataFrame, params) -> dict:
    gate = log[log["concept:name"].eq("Gate In")]
    timestamps = pd.to_datetime(gate["start:timestamp"], errors="coerce").dropna()
    calendar = getattr(params, "arrival_calendar", {})
    closed = sunday = night = 0
    for ts in timestamps:
        # ProSiT day keys follow Python datetime.weekday(): Monday=0,
        # ..., Saturday=5, Sunday=6.
        day = int(ts.weekday())
        hour = int(ts.hour)
        closed += int(not bool(calendar.get(day, {}).get(hour, False)))
        sunday += int(day == 6)
        night += int(hour < 5)
    return {
        "gate_arrivals": int(len(timestamps)),
        "closed_period_gate_arrivals": int(closed),
        "sunday_gate_arrivals": int(sunday),
        "night_gate_arrivals": int(night),
        "arrival_calendar_compliance": bool(closed == 0),
    }


def _run_replication(log: pd.DataFrame, *, seed: int, multiplier: float,
                     n_cases: int, params) -> dict:
    scenario = reproduce._scenario_module()
    label = f"calendar_demand_x{multiplier:.2f}"
    metrics = scenario._summarize_simulation(log, seed=seed, scenario=label)
    audit = scenario._contract_audit(
        log, seed=seed, scenario=label, n_traces=n_cases, blocked_blocks=set()
    )
    structural_violations = sum(int(audit[key]) for key in scenario.HARD_CONTRACT_KEYS)
    structural_violations += int(audit["wrong_case_count"])
    structural_violations += int(audit["prohibited_resource_assignments"])
    if structural_violations:
        raise RuntimeError(f"Structural contract violation: {audit}")
    calendar = _calendar_audit(log, params)
    # Sunday arrivals are a domain violation. Exact closed-hour boundary
    # crossings are retained and reported because they arise from ProSiT's
    # residual-subminute timestamp convention.
    if calendar["sunday_gate_arrivals"]:
        raise RuntimeError(f"Sunday arrival violation: {calendar}")
    metrics.update(calendar)
    metrics["demand_multiplier"] = float(multiplier)
    metrics["nominal_cases"] = int(n_cases)
    return metrics


def _calendar_safe_demand_params(scenario, baseline, multiplier: float):
    """Scale the empirical working-minute arrival model by ``multiplier``."""
    params = (
        deepcopy(baseline)
        if np.isclose(multiplier, 1.0)
        else scenario.apply_demand_increase(
            baseline, demand_increase_pct=(multiplier - 1.0) * 100.0
        )
    )
    return params


def _simulate_calendar_safe(scenario, params, *, n_cases: int, seed: int):
    """Run ProSiT while correcting exact closed-hour boundary returns."""
    import prosit.simulator as prosit_simulator

    original_add = prosit_simulator.add_minutes_with_calendar

    def strict_add(start_ts, minutes_to_add, calendar, _working_set=None):
        timestamp = original_add(start_ts, minutes_to_add, calendar, _working_set)
        # ProSiT can return exactly the first minute of a closed hour when an
        # open hour ends at that boundary. Move to the next active hour.
        while not bool(calendar.get(timestamp.weekday(), {}).get(timestamp.hour, False)):
            timestamp = (timestamp + timedelta(hours=1)).replace(
                minute=0, second=0, microsecond=0
            )
        return timestamp

    prosit_simulator.add_minutes_with_calendar = strict_add
    try:
        return scenario.simulate_ctb(
            deepcopy(params), n_traces=n_cases,
            t_start=reproduce.START_TIME, seed=seed,
            timestamp_resolution="min",
        )
    finally:
        prosit_simulator.add_minutes_with_calendar = original_add


def _paired_deltas(replications: pd.DataFrame) -> pd.DataFrame:
    metrics = list(reproduce._scenario_module().KPI_COLUMNS)
    indexed = replications.set_index(["seed", "demand_multiplier"])
    rows = []
    for seed in sorted(replications["seed"].unique()):
        baseline = indexed.loc[(seed, 1.0)]
        for multiplier in sorted(replications["demand_multiplier"].unique()):
            if np.isclose(multiplier, 1.0):
                continue
            stressed = indexed.loc[(seed, multiplier)]
            row = {"seed": int(seed), "demand_multiplier": float(multiplier)}
            for metric in metrics:
                row[f"baseline_{metric}"] = float(baseline[metric])
                row[f"scenario_{metric}"] = float(stressed[metric])
                row[f"delta_{metric}"] = float(stressed[metric] - baseline[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def _summary(replications: pd.DataFrame) -> pd.DataFrame:
    metrics = list(reproduce._scenario_module().KPI_COLUMNS)
    rows = []
    for multiplier, group in replications.groupby("demand_multiplier", sort=True):
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append({
                "demand_multiplier": float(multiplier),
                "metric": metric,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "min": float(values.min()),
                "max": float(values.max()),
            })
    return pd.DataFrame(rows)


def run(*, smoke: bool = False, base_cases: int | None = None,
        output_dir: Path | None = None, n_seeds: int | None = None) -> Path:
    scenario = reproduce._scenario_module()
    baseline = reproduce.load_pickle_models()["baseline"]
    seeds = (reproduce.SEEDS[:2] if smoke else reproduce.SEEDS)
    if n_seeds is not None:
        if n_seeds <= 0 or n_seeds > len(reproduce.SEEDS):
            raise ValueError(f"n_seeds must be in 1..{len(reproduce.SEEDS)}")
        seeds = reproduce.SEEDS[:n_seeds]
    base = int(base_cases or (250 if smoke else reproduce.N_CASES))
    multipliers = (1.0, 1.2, 2.4, 3.0) if smoke else MULTIPLIERS
    output = Path(output_dir or (DEFAULT_OUTPUT.parent / "calendar_preserving_smoke" if smoke else DEFAULT_OUTPUT))
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    started = datetime.now().astimezone()
    for seed in seeds:
        print(f"seed={seed}", flush=True)
        for multiplier in multipliers:
            n_cases = int(round(base * multiplier))
            params = _calendar_safe_demand_params(scenario, baseline, multiplier)
            print(f"  calendar_demand_x{multiplier:.2f} (n_cases={n_cases}) ... ", end="", flush=True)
            with redirect_stderr(io.StringIO()):
                log = _simulate_calendar_safe(
                    scenario, params, n_cases=n_cases, seed=seed
                )
            row = _run_replication(log, seed=seed, multiplier=multiplier, n_cases=n_cases, params=params)
            rows.append(row)
            print(f"turnaround={row['mean_turnaround_min']:.3f}; arrivals={row['gate_arrivals']}; PASS", flush=True)

    replications = pd.DataFrame(rows)
    replications.to_csv(output / "calendar_preserving_replications.csv", index=False)
    _summary(replications).to_csv(output / "calendar_preserving_summary.csv", index=False)
    _paired_deltas(replications).to_csv(output / "calendar_preserving_paired_deltas.csv", index=False)
    run_summary = {
        "status": "completed",
        "started_at": started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "experiment": "calendar_preserving_demand_intensity",
        "base_cases": base,
        "seeds": list(seeds),
        "demand_multipliers": list(multipliers),
        "case_count_rule": "round(base_cases * demand_multiplier)",
        "inter_arrival_rule": "working_minute_inter_arrivals / demand_multiplier",
        "closed_boundary_handling": "exact closed-hour boundary crossings retained and audited",
        "gate_calendar_unchanged": True,
        "all_sunday_audits_passed": bool((replications["sunday_gate_arrivals"] == 0).all()),
        "closed_period_gate_arrivals_total": int(replications["closed_period_gate_arrivals"].sum()),
        "arrival_calendar_compliance_rate": float(replications["arrival_calendar_compliance"].mean()),
        "all_cases_complete": bool((replications["sim_cases"] == replications["nominal_cases"]).all()),
    }
    (output / "calendar_preserving_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--base-cases", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    args = parser.parse_args()
    print(f"COMPLETE -> {run(smoke=args.smoke, base_cases=args.base_cases, output_dir=args.output_dir, n_seeds=args.n_seeds)}")
