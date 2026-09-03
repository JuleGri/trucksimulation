#!/usr/bin/env python3
"""Run and resume the complete CTB 2x2 feature-source experiment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PARAM_ROOT = REPO / "baseline/discovery_params/params_20260816_214403_train80"
RESULTS = REPO / "validation/results"

MODELS = {
    "visit_only": PARAM_ROOT / "prosit_discovery_feature_visit_only_common_20260829",
    "static_only": PARAM_ROOT / "prosit_discovery_feature_static_only_20260829",
    "native_only": PARAM_ROOT / "prosit_discovery_feature_native_only_20260829",
    "both": PARAM_ROOT / "prosit_discovery_feature_both_common_20260829",
}


def run_step(name: str, command: list[str], expected: Path | None = None) -> None:
    if expected is not None and expected.exists():
        print(f"\n[feature-factorial] SKIP {name}: {expected} already exists", flush=True)
        return
    print(f"\n[feature-factorial] START {name}", flush=True)
    print("  " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)
    if expected is not None and not expected.exists():
        raise FileNotFoundError(f"{name} finished but did not create {expected}")
    print(f"[feature-factorial] DONE {name}", flush=True)


def main() -> None:
    python = sys.executable
    run_step(
        "discover visit-only model on the common CSV source",
        [
            python, "baseline/07_run_prosit_discovery.py",
            "--no-use-workload-features", "--workload-blind-attributes",
            "--out-suffix", "_feature_visit_only_common_20260829",
            "--skip-simulation", "--skip-figures",
        ],
        MODELS["visit_only"] / "prosit_params.pkl",
    )
    run_step(
        "discover static-only model",
        [
            python, "baseline/07_run_prosit_discovery.py",
            "--no-use-workload-features",
            "--out-suffix", "_feature_static_only_20260829",
            "--skip-simulation", "--skip-figures",
        ],
        MODELS["static_only"] / "prosit_params.pkl",
    )
    run_step(
        "discover native-only model",
        [
            python, "baseline/07_run_prosit_discovery.py",
            "--use-workload-features", "--workload-blind-attributes",
            "--out-suffix", "_feature_native_only_20260829",
            "--skip-simulation", "--skip-figures",
        ],
        MODELS["native_only"] / "prosit_params.pkl",
    )
    run_step(
        "discover both-sources model on the common CSV source",
        [
            python, "baseline/07_run_prosit_discovery.py",
            "--use-workload-features", "--no-workload-blind-attributes",
            "--out-suffix", "_feature_both_common_20260829",
            "--skip-simulation", "--skip-figures",
        ],
        MODELS["both"] / "prosit_params.pkl",
    )

    validation_labels = {
        "visit_only": "feature_factorial_20260829_visit_only_common_vs_holdout_ci",
        "static_only": "feature_factorial_20260829_static_only_vs_holdout_ci",
        "native_only": "feature_factorial_20260829_native_only_vs_holdout_ci",
        "both": "feature_factorial_20260829_both_common_vs_holdout_ci",
    }
    for configuration, label in validation_labels.items():
        run_step(
            f"ten-seed hold-out validation: {configuration}",
            [
                python, "validation/06_multi_seed_ci.py",
                "--params", str(MODELS[configuration] / "prosit_params.pkl"),
                "--n-seeds", "10", "--base-seed", "42",
                "--label", label,
            ],
            RESULTS / label / "mc_replications.csv",
        )

    audit_output = RESULTS / "feature_source_audit_standardized_20260830"
    audit_command = [
        python, "validation/20_feature_source_audit.py",
        "--full-params", str(MODELS["both"] / "prosit_params.pkl"),
        "--output", str(audit_output),
        "--model-json", f"visit_only={MODELS['visit_only'] / 'prosit_params.json'}",
        "--model-json", f"static_only={MODELS['static_only'] / 'prosit_params.json'}",
        "--model-json", f"native_only={MODELS['native_only'] / 'prosit_params.json'}",
        "--model-json", f"both={MODELS['both'] / 'prosit_params.json'}",
    ]
    run_step(
        "correlation, redundancy, and retained-rule audit",
        audit_command,
        audit_output / "audit_manifest.json",
    )

    ngd_output = RESULTS / "feature_source_factorial_standardized_ngd_20260829"
    ngd_command = [
        python, "validation/19_ngram_control_flow_distance.py",
        "--output", str(ngd_output),
        "--n-seeds", "10", "--base-seed", "42",
    ]
    for configuration, path in MODELS.items():
        ngd_command.extend([
            "--configuration", f"{configuration}={path / 'prosit_params.pkl'}"
        ])
    for contrast in (
        "static_only:visit_only",
        "native_only:visit_only",
        "both:static_only",
        "both:native_only",
    ):
        ngd_command.extend(["--contrast", contrast])
    run_step(
        "four-cell 3-gram NGD",
        ngd_command,
        ngd_output / "ngd_run_manifest.json",
    )

    run_step(
        "paired factorial effects and interaction",
        [python, "validation/21_feature_source_factorial.py"],
        RESULTS / "feature_source_factorial_standardized_20260829/factorial_manifest.json",
    )
    print("\n[feature-factorial] ALL EXPERIMENTS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
