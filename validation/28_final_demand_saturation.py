"""Demand-saturation ladder for the final selected CTB parameter bundle."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr
from copy import deepcopy
from datetime import datetime
import hashlib
from importlib import util
import io
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DEFAULT_PARAMS = (
    REPO
    / "validation/results/tvt_final_visit_only_scenarios_cap3_20260831"
    / "params_baseline_rmg_max_concurrency_3.pkl"
)
DEFAULT_OUTPUT = (
    REPO / "validation/results/tvt_final_visit_only_saturation_cap3_20260831"
)
DEFAULT_MANIFEST = REPO / "validation/results/split_manifest.json"
MULTIPLIERS = (1.0, 1.2, 1.5, 2.0, 2.4, 3.0)


def load_scenario_module():
    path = REPO / "validation/09_multi_seed_scenarios.py"
    spec = util.spec_from_file_location("ctb_scenarios", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--n-cases", type=int, default=17892)
    parser.add_argument(
        "--rmg-max-concurrency",
        type=int,
        default=3,
        help="Clamp every RMG resource to this concurrency before the ladder.",
    )
    parser.add_argument(
        "--model-label",
        default="final selected visit-only expert-repaired cap-three bundle",
        help="Auditable description written to the run summary.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interval(values: pd.Series) -> tuple[int, float, float, float, float]:
    sample = pd.to_numeric(values, errors="coerce").dropna()
    n = len(sample)
    mean = float(sample.mean())
    sd = float(sample.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return n, mean, sd, mean - half, mean + half


def summarise(replications: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for multiplier, group in replications.groupby("demand_multiplier", sort=True):
        for metric in metrics:
            n, mean, sd, lo, hi = interval(group[metric])
            rows.append({"demand_multiplier": multiplier, "metric": metric,
                         "n": n, "mean": mean, "std": sd,
                         "ci95_lo": lo, "ci95_hi": hi})
    return pd.DataFrame(rows)


def paired(replications: pd.DataFrame, metrics: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = replications.set_index(["seed", "demand_multiplier"])
    rows = []
    for seed in sorted(replications.seed.unique()):
        baseline = indexed.loc[(seed, 1.0)]
        for multiplier in sorted(replications.demand_multiplier.unique()):
            if np.isclose(multiplier, 1.0):
                continue
            scenario = indexed.loc[(seed, multiplier)]
            row = {"seed": int(seed), "demand_multiplier": float(multiplier)}
            for metric in metrics:
                row[f"baseline_{metric}"] = float(baseline[metric])
                row[f"scenario_{metric}"] = float(scenario[metric])
                row[f"delta_{metric}"] = float(scenario[metric] - baseline[metric])
            rows.append(row)
    deltas = pd.DataFrame(rows)
    summaries = []
    for multiplier, group in deltas.groupby("demand_multiplier", sort=True):
        for metric in metrics:
            n, mean, sd, lo, hi = interval(group[f"delta_{metric}"])
            summaries.append({"demand_multiplier": multiplier, "metric": metric,
                              "n": n, "mean_delta": mean, "std_delta": sd,
                              "ci95_delta_lo": lo, "ci95_delta_hi": hi,
                              "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    return deltas, pd.DataFrame(summaries)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scenario = load_scenario_module()
    with args.params.open("rb") as handle:
        source_baseline = pickle.load(handle)
    source_rmg_concurrency = scenario._rmg_concurrency_summary(source_baseline)
    baseline = scenario.apply_rmg_concurrency_cap(
        source_baseline, args.rmg_max_concurrency
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    start = pd.Timestamp(manifest["cutoff_arrival_ts"]).to_pydatetime()
    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))
    templates = {
        multiplier: (
            deepcopy(baseline)
            if np.isclose(multiplier, 1.0)
            else scenario.apply_demand_increase(
                baseline, demand_increase_pct=(multiplier - 1.0) * 100.0
            )
        )
        for multiplier in MULTIPLIERS
    }

    rows, audits = [], []
    started = datetime.now().astimezone()
    for seed in seeds:
        print(f"seed={seed}", flush=True)
        for multiplier in MULTIPLIERS:
            label = f"demand_x{multiplier:.2f}"
            print(f"  {label} ...", end=" ", flush=True)
            with redirect_stderr(io.StringIO()):
                log = scenario.simulate_ctb(
                    deepcopy(templates[multiplier]), n_traces=args.n_cases,
                    t_start=start, seed=seed, timestamp_resolution="min"
                )
            metrics = scenario._summarize_simulation(log, seed=seed, scenario=label)
            audit = scenario._contract_audit(
                log, seed=seed, scenario=label, n_traces=args.n_cases,
                blocked_blocks=set()
            )
            violations = sum(int(audit[key]) for key in scenario.HARD_CONTRACT_KEYS)
            violations += int(audit["wrong_case_count"])
            violations += int(audit["prohibited_resource_assignments"])
            if violations:
                raise RuntimeError(f"Structural contract failed: {audit}")
            metrics["demand_multiplier"] = multiplier
            audit["demand_multiplier"] = multiplier
            rows.append(metrics)
            audits.append(audit)
            print(f"turnaround={metrics['mean_turnaround_min']:.3f}; PASS", flush=True)

    replications = pd.DataFrame(rows)
    contracts = pd.DataFrame(audits)
    metrics = tuple(scenario.KPI_COLUMNS)
    summary = summarise(replications, metrics)
    deltas, delta_summary = paired(replications, metrics)
    replications.to_csv(args.output / "saturation_replications.csv", index=False)
    contracts.to_csv(args.output / "saturation_contracts.csv", index=False)
    summary.to_csv(args.output / "saturation_summary.csv", index=False)
    deltas.to_csv(args.output / "saturation_paired_deltas.csv", index=False)
    delta_summary.to_csv(args.output / "saturation_paired_delta_summary.csv", index=False)
    run_summary = {
        "status": "completed",
        "started_at": started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "source_params": str(args.params),
        "source_params_sha256": sha256(args.params),
        "model": args.model_label,
        "rules_mode": bool(baseline.rules_mode),
        "use_workload_features": bool(baseline.use_workload_features),
        "requested_rmg_max_concurrency": int(args.rmg_max_concurrency),
        "rmg_concurrency_before": source_rmg_concurrency,
        "rmg_concurrency_after": scenario._rmg_concurrency_summary(baseline),
        "n_cases_per_run": args.n_cases,
        "seeds": seeds,
        "demand_multipliers": list(MULTIPLIERS),
        "paired_common_random_numbers": True,
        "all_structural_contracts_passed": True,
    }
    (args.output / "saturation_run_summary.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8"
    )
    (args.output / "PIPELINE_COMPLETE.txt").write_text("complete\n", encoding="utf-8")
    print(f"COMPLETE -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
