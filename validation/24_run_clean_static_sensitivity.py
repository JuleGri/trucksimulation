#!/usr/bin/env python3
"""Run the pre-specified CTB cleaned-static sensitivity analysis."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PARAM_ROOT = REPO / "baseline/discovery_params/params_20260816_214403_train80"
RESULTS = REPO / "validation/results"
STANDARDIZED_DONE = (
    RESULTS
    / "feature_source_factorial_standardized_20260829"
    / "factorial_manifest.json"
)
VISIT = PARAM_ROOT / "prosit_discovery_feature_visit_only_common_20260829"
CLEAN = PARAM_ROOT / "prosit_discovery_feature_clean_static_20260829"


def run_step(name: str, command: list[str], expected: Path) -> None:
    if expected.exists():
        print(f"\n[clean-static] SKIP {name}: {expected} already exists", flush=True)
        return
    print(f"\n[clean-static] START {name}", flush=True)
    print("  " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)
    if not expected.exists():
        raise FileNotFoundError(f"{name} finished but did not create {expected}")
    print(f"[clean-static] DONE {name}", flush=True)


def main() -> None:
    if not STANDARDIZED_DONE.exists():
        raise FileNotFoundError(
            "The standardized 2x2 experiment is not complete: " + str(STANDARDIZED_DONE)
        )
    if not (VISIT / "prosit_params.pkl").exists():
        raise FileNotFoundError("Standardized visit-only bundle is missing")

    python = sys.executable
    run_step(
        "discover semantically cleaned static-state model",
        [
            python,
            "baseline/07_run_prosit_discovery.py",
            "--no-use-workload-features",
            "--clean-static-attributes",
            "--out-suffix",
            "_feature_clean_static_20260829",
            "--skip-simulation",
            "--skip-figures",
        ],
        CLEAN / "prosit_params.pkl",
    )
    run_step(
        "ten-seed hold-out sensitivity",
        [
            python,
            "validation/06_multi_seed_ci.py",
            "--params",
            str(CLEAN / "prosit_params.pkl"),
            "--n-seeds",
            "10",
            "--base-seed",
            "42",
            "--label",
            "feature_clean_static_20260829_vs_holdout_ci",
        ],
        RESULTS / "feature_clean_static_20260829_vs_holdout_ci/mc_replications.csv",
    )

    ngd_output = RESULTS / "feature_clean_static_ngd_20260829"
    run_step(
        "paired visit-only versus clean-static 3-gram NGD",
        [
            python,
            "validation/19_ngram_control_flow_distance.py",
            "--output",
            str(ngd_output),
            "--n-seeds",
            "10",
            "--base-seed",
            "42",
            "--configuration",
            f"visit_only={VISIT / 'prosit_params.pkl'}",
            "--configuration",
            f"clean_static={CLEAN / 'prosit_params.pkl'}",
            "--contrast",
            "clean_static:visit_only",
        ],
        ngd_output / "ngd_run_manifest.json",
    )

    audit_output = RESULTS / "feature_clean_static_rule_audit_20260829"
    run_step(
        "retained-rule inventory",
        [
            python,
            "validation/20_feature_source_audit.py",
            "--full-params",
            str(CLEAN / "prosit_params.pkl"),
            "--output",
            str(audit_output),
            "--model-json",
            f"visit_only={VISIT / 'prosit_params.json'}",
            "--model-json",
            f"clean_static={CLEAN / 'prosit_params.json'}",
        ],
        audit_output / "audit_manifest.json",
    )
    run_step(
        "paired sensitivity summary",
        [python, "validation/23_compare_clean_static_sensitivity.py"],
        RESULTS / "feature_clean_static_sensitivity_20260829/sensitivity_manifest.json",
    )
    print("\n[clean-static] ALL SENSITIVITY STEPS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
