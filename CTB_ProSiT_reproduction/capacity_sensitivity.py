"""Reproduce the final uncapped-versus-cap-three sensitivity."""

from __future__ import annotations

from contextlib import redirect_stderr
from copy import deepcopy
import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import reproduce


ROOT = Path(__file__).resolve().parent
EXPECTED = ROOT / "expected_results" / "capacity_sensitivity"
OUTPUTS = ROOT / "outputs"
UNCAPPED_MODEL = ROOT / "models" / "params_uncapped.pkl"
SCENARIOS = ("baseline", "t22_closed", "demand_plus_20pct")
INTERVENTIONS = SCENARIOS[1:]
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


def load_uncapped_model():
    with UNCAPPED_MODEL.open("rb") as handle:
        return pickle.load(handle)


def _ci(values: pd.Series) -> tuple[int, float, float, float, float]:
    sample = pd.to_numeric(values, errors="coerce").dropna()
    n = int(len(sample))
    mean = float(sample.mean())
    if n == 1:
        return n, mean, float("nan"), mean, mean
    std = float(sample.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n))
    return n, mean, std, mean - half, mean + half


def _cap_effects(uncapped: pd.DataFrame, capped: pd.DataFrame) -> pd.DataFrame:
    merged = uncapped.merge(
        capped,
        on=["seed", "scenario"],
        suffixes=("_uncapped", "_cap3"),
        validate="one_to_one",
    )
    rows = []
    for row in merged.itertuples(index=False):
        for metric in METRICS:
            reference = float(getattr(row, f"{metric}_uncapped"))
            capped_value = float(getattr(row, f"{metric}_cap3"))
            delta = capped_value - reference
            rows.append(
                {
                    "seed": int(row.seed),
                    "scenario": row.scenario,
                    "metric": metric,
                    "reference_value": reference,
                    "sensitivity_value": capped_value,
                    "delta": delta,
                    "delta_pct": 100.0 * delta / reference if reference else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _summarise_cap_effects(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario_name in SCENARIOS:
        for metric in METRICS:
            group = effects[
                effects["scenario"].eq(scenario_name)
                & effects["metric"].eq(metric)
            ]
            n, mean, std, lo, hi = _ci(group["delta"])
            _, pct, pct_std, pct_lo, pct_hi = _ci(group["delta_pct"])
            rows.append(
                {
                    "scenario": scenario_name,
                    "metric": metric,
                    "n": n,
                    "mean_reference": float(group["reference_value"].mean()),
                    "mean_sensitivity": float(group["sensitivity_value"].mean()),
                    "mean_delta": mean,
                    "std_delta": std,
                    "ci95_delta_lo": lo,
                    "ci95_delta_hi": hi,
                    "mean_delta_pct": pct,
                    "std_delta_pct": pct_std,
                    "ci95_delta_pct_lo": pct_lo,
                    "ci95_delta_pct_hi": pct_hi,
                    "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
                }
            )
    return pd.DataFrame(rows)


def _interaction_summary(uncapped: pd.DataFrame, capped: pd.DataFrame) -> pd.DataFrame:
    reference = uncapped.set_index(["seed", "scenario"])
    sensitivity = capped.set_index(["seed", "scenario"])
    rows = []
    for scenario_name in INTERVENTIONS:
        for metric in METRICS:
            samples = []
            reference_effects = []
            sensitivity_effects = []
            for seed in sorted(uncapped["seed"].unique()):
                ref_effect = float(
                    reference.loc[(seed, scenario_name), metric]
                    - reference.loc[(seed, "baseline"), metric]
                )
                cap_effect = float(
                    sensitivity.loc[(seed, scenario_name), metric]
                    - sensitivity.loc[(seed, "baseline"), metric]
                )
                reference_effects.append(ref_effect)
                sensitivity_effects.append(cap_effect)
                samples.append(cap_effect - ref_effect)
            n, mean, std, lo, hi = _ci(pd.Series(samples))
            rows.append(
                {
                    "scenario": scenario_name,
                    "metric": metric,
                    "n": n,
                    "mean_reference_scenario_effect": float(np.mean(reference_effects)),
                    "mean_sensitivity_scenario_effect": float(np.mean(sensitivity_effects)),
                    "mean_interaction_delta": mean,
                    "std_interaction_delta": std,
                    "ci95_interaction_lo": lo,
                    "ci95_interaction_hi": hi,
                    "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
                }
            )
    return pd.DataFrame(rows)


def run(*, full: bool = True, capped_output: Path | None = None) -> Path:
    scenario = reproduce._scenario_module()
    uncapped = load_uncapped_model()
    templates = {
        "baseline": deepcopy(uncapped),
        "t22_closed": scenario.apply_t22_closure(uncapped, {"T22"}),
        "demand_plus_20pct": scenario.apply_demand_increase(
            uncapped, demand_increase_pct=20.0
        ),
    }
    seeds = reproduce.SEEDS if full else reproduce.SEEDS[:2]
    n_cases = reproduce.N_CASES if full else 250
    output = OUTPUTS / ("capacity_full" if full else "capacity_smoke")
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    audits = []
    for seed in seeds:
        print(f"seed={seed}")
        for name, params in templates.items():
            print(f"  uncapped {name} ...", end=" ", flush=True)
            with redirect_stderr(io.StringIO()):
                log = scenario.simulate_ctb(
                    deepcopy(params),
                    n_traces=n_cases,
                    t_start=reproduce.START_TIME,
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

    uncapped_replications = pd.DataFrame(rows)
    pd.DataFrame(audits).to_csv(output / "uncapped_contracts.csv", index=False)
    uncapped_replications.to_csv(output / "uncapped_replications.csv", index=False)

    if capped_output is None:
        capped_output = OUTPUTS / ("full" if full else "smoke")
    capped_path = Path(capped_output) / "scenario_replications.csv"
    if not capped_path.exists():
        raise FileNotFoundError(
            f"Run reproduce.run(full={full}) first; missing {capped_path}"
        )
    capped_replications = pd.read_csv(capped_path)
    effects = _cap_effects(uncapped_replications, capped_replications)
    effect_summary = _summarise_cap_effects(effects)
    interaction_summary = _interaction_summary(
        uncapped_replications, capped_replications
    )
    effect_summary.to_csv(
        output / "capacity_sensitivity_paired_delta_summary.csv", index=False
    )
    interaction_summary.to_csv(
        output / "capacity_sensitivity_interaction_summary.csv", index=False
    )
    return output


def compare(output: Path) -> pd.DataFrame:
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
    if report.empty or not report["values_match"].all():
        raise AssertionError(report.to_string(index=False))
    return report
