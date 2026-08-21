"""Rebuild the CTB log from stage 05, split, discover, simulate and audit.

The workflow stops immediately on any failed stage. Use
``--skip-eventlog-rebuild`` only when the current s6 log and train/test split
are already known to come from the corrected case-ordering pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-eventlog-rebuild",
        action="store_true",
        help="Reuse the current s6 log and splits; still regenerate XES and rerun discovery.",
    )
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-contract-validation", action="store_true")
    parser.add_argument(
        "--validation-label",
        default="prosit_sequential_calibrated_vs_holdout",
    )
    return parser.parse_args()


def run_stage(index: int, total: int, name: str, command: list[str]) -> None:
    print("\n" + "=" * 72)
    print(f"STEP {index}/{total}: {name}")
    print("=" * 72)
    print(subprocess.list2cmdline(command))
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode:
        raise SystemExit(
            f"Pipeline stopped: {name!r} returned {completed.returncode}."
        )


def latest_sequential_output() -> tuple[Path, Path]:
    candidates = list(
        (REPO_ROOT / "baseline" / "discovery_params").glob(
            "*_train80/prosit_discovery_workload_sequential*/prosit_run_summary.json"
        )
    )
    if not candidates:
        raise FileNotFoundError("No sequential ProSiT discovery output was created.")
    summary = max(candidates, key=lambda path: path.stat().st_mtime)
    with open(summary, "r") as fh:
        run_summary = json.load(fh)
    configured_output = run_summary.get("simulation", {}).get("output_csv")
    if not configured_output:
        raise FileNotFoundError(
            f"Run summary does not contain a simulation output path: {summary}"
        )
    sim = Path(configured_output)
    if not sim.is_absolute():
        sim = REPO_ROOT / sim
    if not sim.exists():
        raise FileNotFoundError(f"Expected simulation output not found: {sim}")
    return summary, sim


def main() -> int:
    args = parse_args()
    py = sys.executable
    stages: list[tuple[str, list[str]]] = []

    if not args.skip_eventlog_rebuild:
        stages.extend(
            [
                (
                    "Add demand features and persist explicit case order",
                    [py, "eventlog_building/05_add_demand_features_to_eventlog.py"],
                ),
                (
                    "Add utilization features and validate case structure",
                    [py, "eventlog_building/06_add_utils_to_eventlog.py"],
                ),
                (
                    "Map/filter yard activities without creating Gate-only cases",
                    [py, "eventlog_building/07_convert_to_RMG_filter_LL+HO2_only.py"],
                ),
                (
                    "Add target features and validate final real event log",
                    [py, "eventlog_building/08_add_all_target_features.py"],
                ),
                (
                    "Create fresh temporal train/test splits",
                    [
                        py,
                        "validation/01_train_test_split.py",
                        "--test-size",
                        "0.20",
                        "--seed",
                        "42",
                    ],
                ),
            ]
        )

    stages.append(("Regenerate order-preserving train/test XES", [py, "_regen_xes.py"]))
    discovery = [
        py,
        "baseline/07_run_prosit_discovery.py",
        "--xes",
        "data/processed/CTB/xes_files/s6_train.xes",
        "--test-xes",
        "data/processed/CTB/xes_files/s6_test.xes",
        "--enable-multitasking",
    ]
    if args.skip_figures:
        discovery.append("--skip-figures")
    stages.append(
        (
            "Discover, calibrate and simulate the sequential ProSiT model",
            discovery,
        )
    )

    total = len(stages) + (0 if args.skip_contract_validation else 1)
    for index, (name, command) in enumerate(stages, start=1):
        run_stage(index, total, name, command)

    summary, sim = latest_sequential_output()
    if not args.skip_contract_validation:
        run_stage(
            len(stages) + 1,
            total,
            "Reject Gate-only or within-case concurrent simulation output",
            [
                py,
                "validation/08_validate_eventlog_contract.py",
                "--real",
                "data/processed/CTB/s6_test.csv",
                "--sim",
                str(sim),
                "--label",
                args.validation_label,
            ],
        )

    print("\nPipeline complete.")
    print(f"Run summary: {summary}")
    print(f"Simulation : {sim}")
    print(
        "Next: python validation/run_full_validation.py "
        f"--run-summary \"{summary}\" --label {args.validation_label}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
