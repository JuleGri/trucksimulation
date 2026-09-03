#!/usr/bin/env python3
"""Run the pre-specified CTB 64/16/20 model-selection pipeline."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DATA = REPO / "data/processed/CTB"
RESULTS = REPO / "validation/results"
SPLIT_DIR = DATA / "split64_16_20"
FULL = DATA / "s6_eventlog_target_rank_features.csv"
TRAIN64 = SPLIT_DIR / "s6_train.csv"
VALID16 = SPLIT_DIR / "s6_validation.csv"
TEST20 = DATA / "s6_test.csv"
TRAIN80 = DATA / "s6_train.csv"
VAL_MANIFEST = RESULTS / "split_manifest_train64_validation16.json"
TVT_MANIFEST = RESULTS / "split_manifest_64_16_20.json"
FINAL_MANIFEST = RESULTS / "split_manifest.json"
OUT = RESULTS / "train_validation_test_selection_20260831"

CONFIGS = {
    "visit_only": ["--no-use-workload-features", "--workload-blind-attributes"],
    "static_only": ["--no-use-workload-features"],
    "native_only": ["--use-workload-features", "--workload-blind-attributes"],
    "both": ["--use-workload-features", "--no-workload-blind-attributes"],
}
PRIMARY = (
    "case_turnaround_emd_min",
    "yard_service_time_emd_frequency_weighted_min",
    "yard_activity_rate_l1_error",
    "ngd_completion_order",
)


def run(name: str, command: list[str], expected: Path) -> None:
    if expected.exists():
        print(f"[tvt] SKIP {name}: {expected}", flush=True)
        return
    print(f"[tvt] START {name}\n  {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO, check=True)
    if not expected.exists():
        raise FileNotFoundError(f"{name} did not create {expected}")
    print(f"[tvt] DONE {name}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split() -> None:
    if TVT_MANIFEST.exists() and TRAIN64.exists() and VALID16.exists():
        print("[tvt] SKIP split: persisted artifacts exist", flush=True)
        return
    from _eventlog_contract import canonicalize_case_order, validate_eventlog_contract

    frame = pd.read_csv(FULL)
    for column in ("start:timestamp", "time:timestamp"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame = canonicalize_case_order(frame)
    arrivals = (
        frame.assign(_arrival=frame[["start:timestamp", "time:timestamp"]].min(axis=1))
        .groupby("case:concept:name")["_arrival"].min().reset_index()
    )
    rng = np.random.default_rng(42)
    arrivals["_tie"] = rng.random(len(arrivals))
    arrivals = arrivals.sort_values(["_arrival", "_tie"], kind="stable").reset_index(drop=True)
    n = len(arrivals)
    cut64, cut80 = int(np.floor(n * .64)), int(np.floor(n * .80))
    ids = {
        "train": set(arrivals.iloc[:cut64]["case:concept:name"]),
        "validation": set(arrivals.iloc[cut64:cut80]["case:concept:name"]),
        "test": set(arrivals.iloc[cut80:]["case:concept:name"]),
    }
    parts = {name: canonicalize_case_order(frame[frame["case:concept:name"].isin(values)].copy()) for name, values in ids.items()}
    for name, part in parts.items():
        validate_eventlog_contract(part, label=name)
    persisted_test = pd.read_csv(TEST20, usecols=["case:concept:name"])
    persisted_train = pd.read_csv(TRAIN80, usecols=["case:concept:name"])
    if set(persisted_test["case:concept:name"]) != ids["test"]:
        raise ValueError("New 20% test cases do not equal the persisted untouched test set")
    if set(persisted_train["case:concept:name"]) != ids["train"] | ids["validation"]:
        raise ValueError("New 64%+16% cases do not equal the persisted 80% training set")
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN64.parent.mkdir(parents=True, exist_ok=True)
    parts["train"].to_csv(TRAIN64, index=False)
    parts["validation"].to_csv(VALID16, index=False)
    def desc(part: pd.DataFrame) -> dict:
        return {"n_cases": int(part["case:concept:name"].nunique()), "n_events": int(len(part))}
    cutoff64 = arrivals.iloc[cut64]["_arrival"]
    cutoff80 = arrivals.iloc[cut80]["_arrival"]
    val_manifest = {
        "design": "64% discovery training and subsequent 16% temporal validation",
        "cutoff_arrival_ts": cutoff64.isoformat(),
        "train": desc(parts["train"]),
        "test": desc(parts["validation"]),
    }
    VAL_MANIFEST.write_text(json.dumps(val_manifest, indent=2), encoding="utf-8")
    payload = {
        "schema": "ctb-temporal-train-validation-test-1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "case_order": "earliest observed event timestamp; seeded tie break 42",
        "fractions": {"train": .64, "validation": .16, "test": .20},
        "cutoff_train_validation": cutoff64.isoformat(),
        "cutoff_validation_test": cutoff80.isoformat(),
        "train": desc(parts["train"]), "validation": desc(parts["validation"]), "test": desc(parts["test"]),
        "persisted_80_20_equivalence": True,
    }
    TVT_MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[tvt] DONE split: {payload}", flush=True)


def latest_train80_root() -> Path:
    roots = sorted((REPO / "baseline/discovery_params").glob("params_*_train80"))
    if not roots:
        raise FileNotFoundError("No train80 parameter root exists")
    return roots[-1]


def discover_four() -> dict[str, Path]:
    root = latest_train80_root()
    models = {}
    for name, flags in CONFIGS.items():
        suffix = f"_tvt_train64_{name}_20260831"
        folder = root / f"prosit_discovery{suffix}"
        command = [sys.executable, "baseline/07_run_prosit_discovery.py", "--input", str(TRAIN64), "--test", str(VALID16), "--manifest", str(VAL_MANIFEST), "--control-flow-policy", "expert_sequential_contract", *flags, "--out-suffix", suffix, "--skip-simulation", "--skip-figures"]
        run(f"discover {name} on 64%", command, folder / "prosit_params.pkl")
        models[name] = folder / "prosit_params.pkl"
    return models


def validation(models: dict[str, Path]) -> tuple[dict[str, Path], Path]:
    folders = {}
    for name, model in models.items():
        label = f"tvt_validation16_{name}_20260831"
        folder = RESULTS / label
        run(f"validate {name} on 16%", [sys.executable, "validation/06_multi_seed_ci.py", "--params", str(model), "--real", str(VALID16), "--manifest", str(VAL_MANIFEST), "--n-seeds", "10", "--base-seed", "42", "--label", label], folder / "mc_replications.csv")
        folders[name] = folder
    ngd = OUT / "validation_ngd"
    command = [sys.executable, "validation/19_ngram_control_flow_distance.py", "--real", str(VALID16), "--manifest", str(VAL_MANIFEST), "--output", str(ngd), "--n-seeds", "10", "--base-seed", "42"]
    for name, model in models.items():
        command.extend(["--configuration", f"{name}={model}"])
    for name in ("static_only", "native_only", "both"):
        command.extend(["--contrast", f"{name}:visit_only"])
    run("validation 3-gram NGD", command, ngd / "ngd_replications.csv")
    return folders, ngd


def interval(values: pd.Series) -> tuple[float, float, float]:
    x = values.to_numpy(float)
    mean = float(x.mean())
    half = float(stats.t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x)))
    return mean, mean-half, mean+half


def select(folders: dict[str, Path], ngd_dir: Path) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {name: pd.read_csv(folder / "mc_replications.csv").set_index("seed") for name, folder in folders.items()}
    ngd = pd.read_csv(ngd_dir / "ngd_replications.csv")
    for name in frames:
        extra = ngd[ngd["configuration"].eq(name)].set_index("seed")
        frames[name]["ngd_completion_order"] = extra.loc[frames[name].index, "ngd_completion_order"]
    rows = []
    qualified = []
    for name in ("static_only", "native_only", "both"):
        improvements, worsenings = 0, 0
        for metric in PRIMARY:
            mean, lo, hi = interval(frames[name][metric] - frames["visit_only"][metric])
            improved, worsened = hi < 0, lo > 0
            improvements += int(improved); worsenings += int(worsened)
            rows.append({"candidate": name, "metric": metric, "mean_candidate_minus_visit_only": mean, "ci95_lo": lo, "ci95_hi": hi, "resolved_improvement": improved, "resolved_worsening": worsened})
        if improvements >= 1 and worsenings == 0:
            qualified.append(name)
    complexity = {"visit_only": 0, "static_only": 1, "native_only": 1, "both": 2}
    selected = min(qualified, key=lambda name: (complexity[name], frames[name]["ngd_completion_order"].mean())) if qualified else "visit_only"
    pd.DataFrame(rows).to_csv(OUT / "predefined_selection_contrasts.csv", index=False)
    policy = {
        "policy_fixed_before_validation_results": True,
        "primary_metrics": list(PRIMARY),
        "rule": "Additional state replaces visit-only only with a paired 95% CI showing improvement in at least one primary error metric and no paired 95% CI showing worsening in another; parsimony then breaks ties.",
        "qualified_candidates": qualified,
        "selected_configuration": selected,
        "selection_data": "temporal validation 16%; final test 20% not accessed",
        "inputs": {name: sha256(folder / "mc_replications.csv") for name, folder in folders.items()},
    }
    (OUT / "selection_manifest.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(f"[tvt] SELECTED {selected}", flush=True)
    return selected


def final_discovery(selected: str) -> Path:
    root = latest_train80_root()
    suffix = f"_tvt_selected_{selected}_train80_20260831"
    folder = root / f"prosit_discovery{suffix}"
    flags = CONFIGS[selected]
    run("rediscover selected configuration on 80%", [sys.executable, "baseline/07_run_prosit_discovery.py", "--input", str(TRAIN80), "--test", str(TEST20), "--manifest", str(FINAL_MANIFEST), "--control-flow-policy", "expert_sequential_contract", *flags, "--out-suffix", suffix, "--skip-simulation", "--skip-figures"], folder / "prosit_params.pkl")
    return folder / "prosit_params.pkl"


def final_runs(model: Path, selected: str) -> None:
    scenario_label = f"tvt_final_{selected}_scenarios_cap3_20260831"
    scenario = RESULTS / scenario_label
    run("final cap-three scenarios", [sys.executable, "validation/09_multi_seed_scenarios.py", "--params", str(model), "--manifest", str(FINAL_MANIFEST), "--label", scenario_label, "--n-seeds", "10", "--base-seed", "42", "--rmg-max-concurrency", "3"], scenario / "scenario_run_summary.json")
    cap_model = scenario / "params_baseline_rmg_max_concurrency_3.pkl"
    final_label = f"tvt_final_{selected}_cap3_test20_20260831"
    run("final untouched 20% test", [sys.executable, "validation/06_multi_seed_ci.py", "--params", str(cap_model), "--real", str(TEST20), "--manifest", str(FINAL_MANIFEST), "--n-seeds", "10", "--base-seed", "42", "--label", final_label], RESULTS / final_label / "mc_replications.csv")
    ngd = OUT / "final_test_ngd"
    run("final untouched 20% NGD", [sys.executable, "validation/19_ngram_control_flow_distance.py", "--real", str(TEST20), "--manifest", str(FINAL_MANIFEST), "--output", str(ngd), "--n-seeds", "10", "--base-seed", "42", "--configuration", f"{selected}={cap_model}"], ngd / "ngd_replications.csv")
    uncapped_label = f"tvt_final_{selected}_uncapped_test20_20260831"
    run("matching uncapped final baseline", [sys.executable, "validation/06_multi_seed_ci.py", "--params", str(model), "--real", str(TEST20), "--manifest", str(FINAL_MANIFEST), "--n-seeds", "10", "--base-seed", "42", "--label", uncapped_label], RESULTS / uncapped_label / "mc_replications.csv")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    split()
    models = discover_four()
    folders, ngd = validation(models)
    selected = select(folders, ngd)
    model = final_discovery(selected)
    final_runs(model, selected)
    (OUT / "PIPELINE_COMPLETE.txt").write_text(datetime.now().astimezone().isoformat(), encoding="utf-8")
    print("[tvt] PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
