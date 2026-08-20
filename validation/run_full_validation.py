"""Run the complete held-out validation workflow from one command.

The default workflow is deliberately conservative:

1. Verify the existing discovery split (no case overlap and chronological case
   arrivals) and refresh its stale manifest without replacing the CSVs.
2. Regenerate clean train/test XES files.
3. Enforce the structural/temporal CTB case contract on real and simulated logs.
4. Validate the current simulation against the held-out test log.
5. Generate validation plots.
6. Run the multi-seed Monte-Carlo confidence-interval analysis.

Descriptive transition sensitivity, comparison with another validation run,
and thesis-figure export are optional because they require different evidence
or additional up-to-date runs.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import math
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = REPO_ROOT / "validation"
DEFAULT_TRAIN = REPO_ROOT / "data" / "processed" / "CTB" / "s6_train.csv"
DEFAULT_REAL = REPO_ROOT / "data" / "processed" / "CTB" / "s6_test.csv"
DEFAULT_MANIFEST = VALIDATION_DIR / "results" / "split_manifest.json"
DEFAULT_RUN_SUMMARY = (
    REPO_ROOT
    / "baseline"
    / "discovery_params"
    / "params_20260816_214403_train80"
    / "prosit_discovery_workload_sequential"
    / "prosit_run_summary.json"
)


class PipelineError(RuntimeError):
    """Raised when a validation stage cannot safely continue."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        default="prosit_sequential_workload_vs_holdout",
        help="Output folder name under validation/results/.",
    )
    parser.add_argument(
        "--ci-label",
        default=None,
        help="Multi-seed result label (default: <label>_ci).",
    )
    parser.add_argument("--run-summary", type=Path, default=DEFAULT_RUN_SUMMARY)
    parser.add_argument(
        "--sim",
        type=Path,
        default=None,
        help="Simulation CSV. Defaults to simulation.output_csv in --run-summary.",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="ProSiT parameters pickle. Defaults to prosit_params.pkl beside --run-summary.",
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--max-activities", type=int, default=12)

    parser.add_argument("--skip-split-check", action="store_true")
    parser.add_argument(
        "--verify-against-full-log",
        action="store_true",
        help=(
            "Additionally reproduce step 01 from the current full log and require "
            "an exact match. This is expected to fail if the full log changed after discovery."
        ),
    )
    parser.add_argument("--skip-xes", action="store_true")
    parser.add_argument("--skip-contract-check", action="store_true")
    parser.add_argument("--skip-single-run", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-multi-seed", action="store_true")
    parser.add_argument(
        "--include-sensitivity",
        action="store_true",
        help="Also run descriptive transition-percentile sensitivity (step 04).",
    )
    parser.add_argument(
        "--compare-run-a",
        default=None,
        help="Existing validation label to compare against the current --label (step 05).",
    )
    parser.add_argument(
        "--include-thesis-figures",
        action="store_true",
        help="Run step 07; requires all configured CI and scenario inputs.",
    )
    parser.add_argument(
        "--latex-figures",
        type=Path,
        default=None,
        help="Optional target passed to 07_thesis_figures.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without executing or writing files.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_discovery_paths(args: argparse.Namespace) -> tuple[Path, Path, str | None]:
    summary_path = resolve_path(args.run_summary)
    summary: dict = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    elif args.sim is None or args.params is None:
        raise PipelineError(
            f"Discovery summary not found: {summary_path}. Pass both --sim and --params."
        )

    if args.sim is not None:
        sim_path = resolve_path(args.sim)
    else:
        raw_sim = summary.get("simulation", {}).get("output_csv")
        if not raw_sim:
            raise PipelineError(f"No simulation.output_csv recorded in {summary_path}.")
        sim_path = resolve_path(Path(raw_sim))

    if args.params is not None:
        params_path = resolve_path(args.params)
    else:
        output_folder = summary.get("prosit_output_folder")
        params_path = (
            resolve_path(Path(output_folder)) / "prosit_params.pkl"
            if output_folder
            else summary_path.parent / "prosit_params.pkl"
        )
    discovery_t_start = summary.get("simulation", {}).get("t_start")
    return sim_path, params_path, discovery_t_start


def command_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(item) for item in command])


class Runner:
    def __init__(self, label: str, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.out_dir = VALIDATION_DIR / "results" / label
        self.log_path = self.out_dir / "pipeline_execution.json"
        self.log = {
            "label": label,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.executable,
            "repo_root": str(REPO_ROOT),
            "dry_run": dry_run,
            "steps": [],
            "status": "running",
        }

    def write_log(self) -> None:
        if self.dry_run:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(json.dumps(self.log, indent=2), encoding="utf-8")

    def run(self, name: str, command: Sequence[str]) -> None:
        command = [str(item) for item in command]
        print(f"\n[pipeline] === {name} ===")
        print(f"[pipeline] {command_text(command)}")
        record = {
            "name": name,
            "command": command,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "planned" if self.dry_run else "running",
        }
        self.log["steps"].append(record)
        self.write_log()
        if self.dry_run:
            return
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        record["seconds"] = time.perf_counter() - started
        record["returncode"] = completed.returncode
        record["status"] = "completed" if completed.returncode == 0 else "failed"
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.write_log()
        if completed.returncode != 0:
            raise PipelineError(
                f"Stage {name!r} failed with exit code {completed.returncode}."
            )

    def record_internal(self, name: str, details: dict) -> None:
        print(f"\n[pipeline] === {name} ===")
        for key, value in details.items():
            print(f"[pipeline] {key}: {value}")
        self.log["steps"].append(
            {
                "name": name,
                "status": "planned" if self.dry_run else "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "details": details,
            }
        )
        self.write_log()

    def finish(self, status: str, message: str | None = None) -> None:
        self.log["status"] = status
        self.log["finished_at"] = datetime.now(timezone.utc).isoformat()
        if message:
            self.log["message"] = message
        self.write_log()


def _parse_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    value = raw.strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _scan_split(path: Path) -> dict:
    if not path.exists():
        raise PipelineError(f"Split CSV not found: {path}")
    case_arrivals: dict[str, datetime] = {}
    activities: Counter[str] = Counter()
    timestamp_min: datetime | None = None
    timestamp_max: datetime | None = None
    events = 0
    missing_timestamp_events = 0
    case_structure: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"case:concept:name", "concept:name"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise PipelineError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            events += 1
            case_id = row["case:concept:name"]
            activity = row["concept:name"]
            activities[activity] += 1
            structure = case_structure.setdefault(
                case_id,
                {
                    "gate_in": 0,
                    "gate_out": 0,
                    "yard": 0,
                    "first": activity,
                    "last": activity,
                },
            )
            structure["last"] = activity
            if activity == "Gate In":
                structure["gate_in"] = int(structure["gate_in"]) + 1
            elif activity == "Gate Out":
                structure["gate_out"] = int(structure["gate_out"]) + 1
            else:
                structure["yard"] = int(structure["yard"]) + 1
            timestamps = [
                parsed
                for parsed in (
                    _parse_timestamp(row.get("start:timestamp")),
                    _parse_timestamp(row.get("time:timestamp")),
                )
                if parsed is not None
            ]
            if not timestamps:
                missing_timestamp_events += 1
                continue
            event_min = min(timestamps)
            event_max = max(timestamps)
            previous = case_arrivals.get(case_id)
            if previous is None or event_min < previous:
                case_arrivals[case_id] = event_min
            timestamp_min = event_min if timestamp_min is None else min(timestamp_min, event_min)
            timestamp_max = event_max if timestamp_max is None else max(timestamp_max, event_max)
    structure_report = {
        "gate_only_cases": sum(int(item["yard"]) == 0 for item in case_structure.values()),
        "invalid_gate_in_count_cases": sum(
            int(item["gate_in"]) != 1 for item in case_structure.values()
        ),
        "invalid_gate_out_count_cases": sum(
            int(item["gate_out"]) != 1 for item in case_structure.values()
        ),
        "wrong_case_boundary_cases": sum(
            item["first"] != "Gate In" or item["last"] != "Gate Out"
            for item in case_structure.values()
        ),
        "minimum_yard_events_per_case": min(
            (int(item["yard"]) for item in case_structure.values()), default=0
        ),
    }
    return {
        "path": path,
        "events": events,
        "case_arrivals": case_arrivals,
        "activity_counts": dict(activities),
        "timestamp_min": timestamp_min,
        "timestamp_max": timestamp_max,
        "missing_timestamp_events": missing_timestamp_events,
        "case_structure_contract": structure_report,
    }


def verify_existing_split(
    runner: Runner, train_path: Path, test_path: Path, manifest_path: Path
) -> None:
    """Validate the immutable discovery snapshot and refresh its manifest."""
    if runner.dry_run:
        runner.record_internal(
            "verify existing discovery split",
            {
                "train": str(train_path),
                "test": str(test_path),
                "manifest": str(manifest_path),
                "action": "check case disjointness and temporal ordering; refresh manifest",
            },
        )
        return

    train = _scan_split(train_path)
    test = _scan_split(test_path)
    train_cases = set(train["case_arrivals"])
    test_cases = set(test["case_arrivals"])
    overlap = train_cases.intersection(test_cases)
    if overlap:
        raise PipelineError(f"Train/test leakage: {len(overlap)} case IDs occur in both splits.")
    if not train_cases or not test_cases:
        raise PipelineError("Train and test splits must both contain timestamped cases.")
    max_train_arrival = max(train["case_arrivals"].values())
    min_test_arrival = min(test["case_arrivals"].values())
    if max_train_arrival > min_test_arrival:
        raise PipelineError(
            "Temporal split violation: a training case arrives after the first test case "
            f"({max_train_arrival.isoformat()} > {min_test_arrival.isoformat()})."
        )

    for split_name, scan in (("train", train), ("test", test)):
        failures = {
            key: value
            for key, value in scan["case_structure_contract"].items()
            if key != "minimum_yard_events_per_case" and value
        }
        if failures:
            raise PipelineError(
                f"{split_name} split violates the CTB case contract: {failures}"
            )

    n_cases_total = len(train_cases) + len(test_cases)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_scope": "immutable train/test snapshot used for discovery",
        "train_path": str(train_path),
        "test_path": str(test_path),
        "test_size": len(test_cases) / n_cases_total,
        "seed": 42,
        "cutoff_arrival_ts": min_test_arrival.isoformat(),
        "n_cases_total": n_cases_total,
        "n_events_total": train["events"] + test["events"],
        "n_events_dropped_missing_ts": (
            train["missing_timestamp_events"] + test["missing_timestamp_events"]
        ),
        "train": {
            "label": "train",
            "n_events": train["events"],
            "n_cases": len(train_cases),
            "timestamp_min": train["timestamp_min"].isoformat(),
            "timestamp_max": train["timestamp_max"].isoformat(),
            "activity_counts": train["activity_counts"],
        },
        "test": {
            "label": "test",
            "n_events": test["events"],
            "n_cases": len(test_cases),
            "timestamp_min": test["timestamp_min"].isoformat(),
            "timestamp_max": test["timestamp_max"].isoformat(),
            "activity_counts": test["activity_counts"],
        },
        "case_structure_contract": {
            "required_order": ["Gate In", "one_or_more_yard_activities", "Gate Out"],
            "train": train["case_structure_contract"],
            "test": test["case_structure_contract"],
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    runner.record_internal(
        "verify existing discovery split",
        {
            "train_cases": len(train_cases),
            "test_cases": len(test_cases),
            "train_events": train["events"],
            "test_events": test["events"],
            "case_overlap": 0,
            "train_gate_only_cases": train["case_structure_contract"]["gate_only_cases"],
            "test_gate_only_cases": test["case_structure_contract"]["gate_only_cases"],
            "cutoff_arrival_ts": min_test_arrival.isoformat(),
            "manifest": str(manifest_path),
        },
    )


def verify_against_full_log(runner: Runner, train_path: Path, test_path: Path) -> None:
    """Optionally require the current full log to reproduce the discovery split."""
    if runner.dry_run:
        placeholder = Path("<temporary-directory>")
        runner.run(
                "strictly compare discovery split with current full log",
            [
                sys.executable,
                VALIDATION_DIR / "01_train_test_split.py",
                "--train-out",
                placeholder / "s6_train.csv",
                "--test-out",
                placeholder / "s6_test.csv",
                "--manifest",
                placeholder / "split_manifest.json",
                "--test-size",
                "0.20",
                "--seed",
                "42",
            ],
        )
        return

    with tempfile.TemporaryDirectory(prefix="ctb_validation_split_") as temp_name:
        temp_dir = Path(temp_name)
        candidate_train = temp_dir / "s6_train.csv"
        candidate_test = temp_dir / "s6_test.csv"
        candidate_manifest = temp_dir / "split_manifest.json"
        runner.run(
            "01 verify temporal train/test split",
            [
                sys.executable,
                VALIDATION_DIR / "01_train_test_split.py",
                "--train-out",
                candidate_train,
                "--test-out",
                candidate_test,
                "--manifest",
                candidate_manifest,
                "--test-size",
                "0.20",
                "--seed",
                "42",
            ],
        )
        if not train_path.exists() or not test_path.exists():
            raise PipelineError("Current train/test CSVs are missing; rerun discovery after splitting.")
        train_equal = filecmp.cmp(train_path, candidate_train, shallow=False)
        test_equal = filecmp.cmp(test_path, candidate_test, shallow=False)
        if not train_equal or not test_equal:
            changed = []
            if not train_equal:
                changed.append(str(train_path))
            if not test_equal:
                changed.append(str(test_path))
            raise PipelineError(
                "The reproducible split differs from the split used for discovery: "
                + ", ".join(changed)
                + ". Validation stopped before mixing stale parameters with new data. "
                "Replace the split and rerun ProSiT discovery first."
            )
        print("[pipeline] Current full log reproduces the discovery split exactly.")


def warn_about_waiting_metric(label: str) -> None:
    summary_path = VALIDATION_DIR / "results" / label / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    waiting = summary.get("mean_waiting_time_emd_min")
    if waiting is None or (isinstance(waiting, float) and math.isnan(waiting)):
        print(
            "[pipeline] NOTE: waiting-time EMD is unavailable because the real "
            "holdout has no model-derived enabled:timestamp. No artificial real "
            "waiting-time comparison will be plotted."
        )


def main() -> int:
    args = parse_args()
    if args.n_seeds < 2 and not args.skip_multi_seed:
        raise SystemExit("--n-seeds must be at least 2 for a confidence interval.")

    train_path = resolve_path(args.train)
    real_path = resolve_path(args.real)
    manifest_path = resolve_path(args.manifest)
    sim_path, params_path, discovery_t_start = load_discovery_paths(args)
    ci_label = args.ci_label or f"{args.label}_ci"
    runner = Runner(args.label, args.dry_run)

    print("[pipeline] Complete CTB validation")
    print(f"[pipeline] simulation : {sim_path}")
    print(f"[pipeline] parameters : {params_path}")
    print(f"[pipeline] real holdout: {real_path}")
    print(f"[pipeline] t_start     : {discovery_t_start or 'from split manifest'}")
    print(f"[pipeline] label      : {args.label}")

    try:
        if not args.skip_split_check:
            verify_existing_split(runner, train_path, real_path, manifest_path)
        if args.verify_against_full_log:
            verify_against_full_log(runner, train_path, real_path)

        if not args.skip_xes:
            runner.run("regenerate clean XES", [sys.executable, REPO_ROOT / "_regen_xes.py"])

        if not args.skip_contract_check:
            runner.run(
                "08 CTB structural and temporal case contract",
                [
                    sys.executable,
                    VALIDATION_DIR / "08_validate_eventlog_contract.py",
                    "--real",
                    real_path,
                    "--sim",
                    sim_path,
                    "--label",
                    args.label,
                ],
            )

        if not args.skip_single_run:
            runner.run(
                "02 held-out simulation validation",
                [
                    sys.executable,
                    VALIDATION_DIR / "02_validate_simulation.py",
                    "--real",
                    real_path,
                    "--sim",
                    sim_path,
                    "--label",
                    args.label,
                    "--seed",
                    str(args.base_seed),
                ],
            )
            if not args.dry_run:
                warn_about_waiting_metric(args.label)

        if not args.skip_plots:
            runner.run(
                "03 validation plots",
                [
                    sys.executable,
                    VALIDATION_DIR / "03_validation_plots.py",
                    "--run-label",
                    args.label,
                    "--max-activities",
                    str(args.max_activities),
                ],
            )

        if not args.skip_multi_seed:
            command = [
                sys.executable,
                VALIDATION_DIR / "06_multi_seed_ci.py",
                "--params",
                params_path,
                "--real",
                real_path,
                "--manifest",
                manifest_path,
                "--n-seeds",
                str(args.n_seeds),
                "--base-seed",
                str(args.base_seed),
                "--label",
                ci_label,
            ]
            if discovery_t_start:
                command.extend(["--t-start", discovery_t_start])
            runner.run("06 multi-seed confidence intervals", command)

        if args.include_sensitivity:
            runner.run(
                "04 descriptive transition sensitivity",
                [sys.executable, VALIDATION_DIR / "04_baseline_percentile_sensitivity.py"],
            )

        if args.compare_run_a:
            runner.run(
                "05 compare validation runs",
                [
                    sys.executable,
                    VALIDATION_DIR / "05_compare_runs.py",
                    "--run-a",
                    args.compare_run_a,
                    "--run-b",
                    args.label,
                    "--label-a",
                    args.compare_run_a,
                    "--label-b",
                    args.label,
                ],
            )

        if args.include_thesis_figures:
            command = [sys.executable, VALIDATION_DIR / "07_thesis_figures.py"]
            if args.latex_figures is not None:
                command.extend(["--latex-figures", resolve_path(args.latex_figures)])
            runner.run("07 thesis figure export", command)

    except (OSError, PipelineError, json.JSONDecodeError) as exc:
        runner.finish("failed", str(exc))
        print(f"\n[pipeline] FAILED: {exc}", file=sys.stderr)
        return 1

    runner.finish("dry-run" if args.dry_run else "completed")
    if args.dry_run:
        print("\n[pipeline] Dry run complete; no files were written.")
    else:
        print(f"\n[pipeline] COMPLETE. Execution log -> {runner.log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
