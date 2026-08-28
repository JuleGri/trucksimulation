"""Minimal CTB ProSiT scenario reproduction used by the reviewer notebooks."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr
from copy import deepcopy
from datetime import datetime
import importlib.util
import io
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pm4py
from prosit import SimulatorParameters


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
EXPECTED = ROOT / "expected_results"
OUTPUTS = ROOT / "outputs"
RUNTIME = ROOT / "runtime"

SEEDS = list(range(42, 52))
N_CASES = 17_892
START_TIME = datetime.fromisoformat("2026-04-20T18:17:00")
RMG_ACTIVITIES = ("RMG_receive", "RMG_delivery", "RMG_mixed")

MODEL_FILES = {
    "baseline": MODELS / "params_baseline.pkl",
    "t22_closed": MODELS / "params_t22_closed.pkl",
    "demand_plus_20pct": MODELS / "params_demand_plus_20pct.pkl",
}


def load_pickle_models() -> dict:
    """Load the exact parameter objects used for the thesis simulations."""
    loaded = {}
    for name, path in MODEL_FILES.items():
        with path.open("rb") as handle:
            loaded[name] = pickle.load(handle)
    return loaded


def load_json_models() -> dict:
    """Load the readable exports with ProSiT's documented from_json API."""
    net, initial_marking, final_marking = pm4py.read_pnml(
        str(MODELS / "ctb_inductive_miner.pnml")
    )
    loaded = {}
    for name in MODEL_FILES:
        params = SimulatorParameters(net, initial_marking, final_marking)
        params.from_json(str(MODELS / f"params_{name}.json"))
        loaded[name] = params
    return loaded


def _arrival_samples(params) -> np.ndarray:
    rules = getattr(params.arrival_time_distribution, "rules", {})
    return np.asarray(rules.get(0, {}).get("sampled", []), dtype=float)


def inspect_models(models: dict) -> pd.DataFrame:
    """Return the model differences that define the two what-if scenarios."""
    rows = []
    for name, params in models.items():
        rmg = sorted(
            {
                resource
                for activity in RMG_ACTIVITIES
                for resource in params.act_to_resources[activity]
            }
        )
        arrivals = _arrival_samples(params)
        rows.append(
            {
                "model": name,
                "places": len(params.net.places),
                "transitions": len(params.net.transitions),
                "arcs": len(params.net.arcs),
                "rmg_blocks": len(rmg),
                "rmg_capacity": sum(params.max_concurrency[r] for r in rmg),
                "t22_eligible": "T22" in rmg,
                "arrival_samples": len(arrivals),
                "mean_inter_arrival_min": arrivals.mean() if len(arrivals) else np.nan,
                "rules_mode": bool(params.rules_mode),
                "workload_features": bool(params.use_workload_features),
            }
        )
    return pd.DataFrame(rows)


def check_model_changes(models: dict) -> pd.DataFrame:
    """Check that each intervention changes only its stated model component."""
    baseline = models["baseline"]
    t22 = models["t22_closed"]
    demand = models["demand_plus_20pct"]

    baseline_rmg = {
        resource
        for activity in RMG_ACTIVITIES
        for resource in baseline.act_to_resources[activity]
    }
    checks = {
        "same Petri-net dimensions": all(
            (len(p.net.places), len(p.net.transitions), len(p.net.arcs))
            == (len(baseline.net.places), len(baseline.net.transitions), len(baseline.net.arcs))
            for p in models.values()
        ),
        "baseline RMG capacity is 22 x 3": (
            len(baseline_rmg) == 22
            and sum(baseline.max_concurrency[r] for r in baseline_rmg) == 66
        ),
        "T22 available in baseline": all(
            "T22" in baseline.act_to_resources[a] for a in RMG_ACTIVITIES
        ),
        "T22 removed in closure scenario": all(
            "T22" not in t22.act_to_resources[a] for a in RMG_ACTIVITIES
        ),
        "T22 scenario retains baseline arrivals": np.array_equal(
            _arrival_samples(t22), _arrival_samples(baseline)
        ),
        "demand scenario retains resources": demand.act_to_resources == baseline.act_to_resources,
        "demand scenario retains capacities": demand.max_concurrency == baseline.max_concurrency,
        "demand scenario divides inter-arrivals by 1.20": np.isclose(
            _arrival_samples(demand).mean(), _arrival_samples(baseline).mean() / 1.20
        ),
    }
    report = pd.DataFrame(
        [{"check": name, "passed": bool(value)} for name, value in checks.items()]
    )
    if not report["passed"].all():
        raise AssertionError(report.to_string(index=False))
    return report


def _scenario_module():
    path = RUNTIME / "validation" / "09_multi_seed_scenarios.py"
    sys.path.insert(0, str(RUNTIME))
    spec = importlib.util.spec_from_file_location("ctb_scenarios", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@contextmanager
def _stable_transition_sampling():
    """Canonicalise ProSiT's enabled-transition set before seeded sampling."""
    import prosit.simulator as simulator

    original = simulator.return_fired_transition

    def stable(weights, enabled):
        ordered = sorted(
            enabled,
            key=lambda t: (str(t.name), "" if t.label is None else str(t.label)),
        )
        return original(weights, ordered)

    simulator.return_fired_transition = stable
    try:
        yield
    finally:
        simulator.return_fired_transition = original


def run(*, full: bool = True) -> Path:
    """Run the three saved models and write the thesis KPI tables."""
    models = load_pickle_models()
    check_model_changes(models)
    scenario = _scenario_module()
    seeds = SEEDS if full else SEEDS[:2]
    n_cases = N_CASES if full else 250
    output = OUTPUTS / ("full" if full else "smoke")
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    audits = []
    for seed in seeds:
        print(f"seed={seed}")
        for name, params in models.items():
            print(f"  {name} ...", end=" ", flush=True)
            with _stable_transition_sampling(), redirect_stderr(io.StringIO()):
                log = scenario.simulate_ctb(
                    deepcopy(params),
                    n_traces=n_cases,
                    t_start=START_TIME,
                    seed=seed,
                    timestamp_resolution="min",
                )
            metrics = scenario._summarize_simulation(log, seed=seed, scenario=name)
            audit = scenario._contract_audit(
                log,
                seed=seed,
                scenario=name,
                n_traces=n_cases,
                blocked_blocks={"T22"},
            )
            scenario._assert_audit(audit, metrics)
            rows.append(metrics)
            audits.append(audit)
            print(f"turnaround={metrics['mean_turnaround_min']:.3f} min; PASS")

    replications = pd.DataFrame(rows)
    contracts = pd.DataFrame(audits)
    kpi_summary = scenario._scenario_kpi_summary(replications)
    paired_deltas = scenario._paired_deltas(replications)
    delta_summary = scenario._paired_delta_summary(paired_deltas)

    tables = {
        "scenario_replications.csv": replications,
        "scenario_contracts.csv": contracts,
        "scenario_kpi_summary.csv": kpi_summary,
        "scenario_paired_deltas.csv": paired_deltas,
        "scenario_paired_delta_summary.csv": delta_summary,
    }
    for filename, table in tables.items():
        table.to_csv(output / filename, index=False)
    return output


def compare(output: Path) -> pd.DataFrame:
    """Compare a full fresh run with the five frozen thesis tables."""
    rows = []
    for expected_path in sorted(EXPECTED.glob("*.csv")):
        actual_path = output / expected_path.name
        expected = pd.read_csv(expected_path)
        actual = pd.read_csv(actual_path)
        same_shape = expected.shape == actual.shape
        same_columns = list(expected.columns) == list(actual.columns)
        values_match = False
        maximum_difference = np.nan
        if same_shape and same_columns:
            numeric = list(expected.select_dtypes(include=[np.number]).columns)
            text = [column for column in expected.columns if column not in numeric]
            numeric_match = np.allclose(
                expected[numeric].to_numpy(float),
                actual[numeric].to_numpy(float),
                rtol=1e-10,
                atol=1e-10,
                equal_nan=True,
            )
            text_match = expected[text].fillna("").equals(actual[text].fillna(""))
            values_match = bool(numeric_match and text_match)
            if numeric:
                maximum_difference = float(
                    np.nanmax(
                        np.abs(
                            expected[numeric].to_numpy(float)
                            - actual[numeric].to_numpy(float)
                        )
                    )
                )
        rows.append(
            {
                "file": expected_path.name,
                "same_shape": same_shape,
                "same_columns": same_columns,
                "values_match": values_match,
                "max_abs_numeric_difference": maximum_difference,
            }
        )
    report = pd.DataFrame(rows)
    report.to_csv(output / "comparison_with_expected_results.csv", index=False)
    if not report["values_match"].all():
        raise AssertionError(report.to_string(index=False))
    return report
