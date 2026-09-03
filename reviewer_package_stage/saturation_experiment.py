"""Paired demand-saturation stress test for the calibrated CTB baseline."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr
from copy import deepcopy
from datetime import datetime
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import reproduce


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
EXPECTED = ROOT / "expected_results" / "saturation"
MULTIPLIERS = (1.0, 1.2, 1.5, 2.0, 2.4, 3.0)
HEADLINE_METRICS = (
    "mean_turnaround_min",
    "p90_turnaround_min",
    "mean_rmg_service_min",
    "mean_rmg_pre_service_min",
    "arrival_rate_per_elapsed_hour",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _ci(values: pd.Series) -> tuple[int, float, float, float, float]:
    sample = pd.to_numeric(values, errors="coerce").dropna()
    n = int(len(sample))
    mean = float(sample.mean())
    if n == 1:
        return n, mean, float("nan"), mean, mean
    std = float(sample.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n))
    return n, mean, std, mean - half, mean + half


def _check_contract(scenario, audit: dict) -> None:
    violations = sum(int(audit[key]) for key in scenario.HARD_CONTRACT_KEYS)
    violations += int(audit["wrong_case_count"])
    violations += int(audit["prohibited_resource_assignments"])
    if violations:
        raise RuntimeError(f"Saturation run failed its structural contract: {audit}")


def _summarise(replications: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = list(reproduce._scenario_module().KPI_COLUMNS)
    for multiplier, group in replications.groupby("demand_multiplier", sort=True):
        for metric in metrics:
            n, mean, std, lo, hi = _ci(group[metric])
            rows.append(
                {
                    "demand_multiplier": float(multiplier),
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                }
            )
    return pd.DataFrame(rows)


def _paired_deltas(replications: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    deltas = pd.DataFrame(rows)

    summary_rows = []
    for multiplier, group in deltas.groupby("demand_multiplier", sort=True):
        for metric in metrics:
            n, mean, std, lo, hi = _ci(group[f"delta_{metric}"])
            summary_rows.append(
                {
                    "demand_multiplier": float(multiplier),
                    "metric": metric,
                    "n": n,
                    "mean_delta": mean,
                    "std_delta": std,
                    "ci95_delta_lo": lo,
                    "ci95_delta_hi": hi,
                    "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
                }
            )
    return deltas, pd.DataFrame(summary_rows)


def _plot(summary: pd.DataFrame, out_path: Path) -> None:
    panels = (
        ("mean_turnaround_min", "Mean turnaround", "min"),
        ("p90_turnaround_min", "P90 turnaround", "min"),
        ("mean_rmg_pre_service_min", "Mean RMG pre-service", "min"),
        ("mean_rmg_service_min", "Mean RMG service", "min"),
    )
    arrival = (
        summary[summary["metric"].eq("arrival_rate_per_elapsed_hour")]
        .set_index("demand_multiplier")["mean"]
    )
    realised_ratio = arrival / float(arrival.loc[1.0])
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    for ax, (metric, title, unit) in zip(axes.ravel(), panels):
        data = summary[summary["metric"].eq(metric)].sort_values("demand_multiplier")
        x = data["demand_multiplier"].map(realised_ratio).to_numpy(float)
        y = data["mean"].to_numpy(float)
        lo = data["ci95_lo"].to_numpy(float)
        hi = data["ci95_hi"].to_numpy(float)
        ax.plot(x, y, marker="o", color="#243b53", linewidth=1.8)
        ax.fill_between(x, lo, hi, color="#9fb3c8", alpha=0.45)
        ax.axvline(2.37, color="#b31b34", linestyle="--", linewidth=1.1)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit)
        ax.grid(True, linestyle=":", alpha=0.45)
    for ax in axes[-1]:
        ax.set_xlabel("Realised elapsed arrival-rate ratio")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run(*, full: bool = True) -> Path:
    scenario = reproduce._scenario_module()
    baseline = reproduce.load_pickle_models()["baseline"]
    seeds = reproduce.SEEDS if full else reproduce.SEEDS[:2]
    n_cases = reproduce.N_CASES if full else 250
    multipliers = MULTIPLIERS if full else (1.0, 1.2, 2.4, 3.0)
    output = OUTPUTS / ("saturation_full" if full else "saturation_smoke")
    output.mkdir(parents=True, exist_ok=True)

    templates = {}
    for multiplier in multipliers:
        templates[multiplier] = (
            deepcopy(baseline)
            if np.isclose(multiplier, 1.0)
            else scenario.apply_demand_increase(
                baseline, demand_increase_pct=(multiplier - 1.0) * 100.0
            )
        )

    rows = []
    audits = []
    started = datetime.now().astimezone()
    for seed in seeds:
        print(f"seed={seed}", flush=True)
        for multiplier in multipliers:
            label = f"demand_x{multiplier:.2f}"
            print(f"  {label} ...", end=" ", flush=True)
            with reproduce._stable_transition_sampling(), redirect_stderr(io.StringIO()):
                log = scenario.simulate_ctb(
                    deepcopy(templates[multiplier]),
                    n_traces=n_cases,
                    t_start=reproduce.START_TIME,
                    seed=seed,
                    timestamp_resolution="min",
                )
            metrics = scenario._summarize_simulation(log, seed=seed, scenario=label)
            audit = scenario._contract_audit(
                log,
                seed=seed,
                scenario=label,
                n_traces=n_cases,
                blocked_blocks=set(),
            )
            _check_contract(scenario, audit)
            metrics["demand_multiplier"] = float(multiplier)
            audit["demand_multiplier"] = float(multiplier)
            rows.append(metrics)
            audits.append(audit)
            print(
                f"turnaround={metrics['mean_turnaround_min']:.3f}; "
                f"p90={metrics['p90_turnaround_min']:.3f}; PASS",
                flush=True,
            )

    replications = pd.DataFrame(rows)
    contracts = pd.DataFrame(audits)
    summary = _summarise(replications)
    deltas, delta_summary = _paired_deltas(replications)

    replications.to_csv(output / "saturation_replications.csv", index=False)
    contracts.to_csv(output / "saturation_contracts.csv", index=False)
    summary.to_csv(output / "saturation_summary.csv", index=False)
    deltas.to_csv(output / "saturation_paired_deltas.csv", index=False)
    delta_summary.to_csv(output / "saturation_paired_delta_summary.csv", index=False)
    _plot(summary, output / "saturation_response_curve.png")

    headline = delta_summary[delta_summary["metric"].isin(HEADLINE_METRICS)]
    run_summary = {
        "status": "completed",
        "started_at": started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "model": "calibrated rules+workload cap-three diagnostic baseline",
        "n_cases_per_run": int(n_cases),
        "seeds": list(seeds),
        "demand_multipliers": list(multipliers),
        "paired_common_random_numbers": True,
        "all_structural_contracts_passed": True,
        "headline_paired_effects": headline.to_dict(orient="records"),
    }
    (output / "saturation_run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )
    return output


def compare(output: Path) -> pd.DataFrame:
    """Compare a fresh full run with the frozen saturation tables."""
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


if __name__ == "__main__":
    args = parse_args()
    result = run(full=not args.smoke)
    print(f"COMPLETE -> {result}")
