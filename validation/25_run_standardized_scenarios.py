#!/usr/bin/env python3
"""Run and freeze the final standardized CTB scenario comparison.

This is a reproducibility-hygiene run, not an additional model-selection
experiment.  It applies the same RMG concurrency cap and the same two
white-box interventions to the exact standardized ``visit_only`` and ``both``
bundles from the completed 2x2 feature-source analysis.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
PARAM_ROOT = REPO / "baseline/discovery_params/params_20260816_214403_train80"
RESULTS = REPO / "validation/results"

MODELS = {
    "visit_only": PARAM_ROOT / "prosit_discovery_feature_visit_only_common_20260829",
    "static_only": PARAM_ROOT / "prosit_discovery_feature_static_only_20260829",
    "native_only": PARAM_ROOT / "prosit_discovery_feature_native_only_20260829",
    "both": PARAM_ROOT / "prosit_discovery_feature_both_common_20260829",
}
SCENARIO_LABELS = {
    "visit_only": "standardized_20260830_visit_only_scenarios_cap3_ci",
    "both": "standardized_20260830_both_scenarios_cap3_ci",
}
REFERENCE_VALIDATION_LABEL = (
    "standardized_20260830_visit_only_reference_cap3_vs_holdout_ci"
)
ROBUSTNESS_OUTPUT = RESULTS / "standardized_20260830_scenario_bundle_robustness"
MODEL_SELECTION_OUTPUT = RESULTS / "standardized_20260830_model_selection"
STANDARDIZED_AUDIT_OUTPUT = RESULTS / "feature_source_audit_standardized_20260830"

PREREQUISITES = (
    RESULTS
    / "feature_source_factorial_standardized_20260829"
    / "factorial_manifest.json",
    RESULTS
    / "feature_clean_static_sensitivity_20260829"
    / "sensitivity_manifest.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_scenario(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "completed"


def run_step(
    name: str,
    command: list[str],
    expected: Path,
    *,
    scenario_summary: bool = False,
) -> None:
    is_done = completed_scenario(expected) if scenario_summary else expected.exists()
    if is_done:
        print(f"\n[standardized-scenarios] SKIP {name}: {expected}", flush=True)
        return
    print(f"\n[standardized-scenarios] START {name}", flush=True)
    print("  " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)
    is_done = completed_scenario(expected) if scenario_summary else expected.exists()
    if not is_done:
        raise FileNotFoundError(f"{name} finished without a complete {expected}")
    print(f"[standardized-scenarios] DONE {name}", flush=True)


def write_provenance() -> Path:
    MODEL_SELECTION_OUTPUT.mkdir(parents=True, exist_ok=True)
    visit_scenario = RESULTS / SCENARIO_LABELS["visit_only"]
    both_scenario = RESULTS / SCENARIO_LABELS["both"]
    reference_cap3 = visit_scenario / "params_baseline_rmg_max_concurrency_3.pkl"

    model_rows = [
        {
            "configuration": "visit_only",
            "role": "historical_reference",
            "adopted": True,
            "reason": (
                "parsimonious model with the best activity-rate and local "
                "3-gram control-flow reproduction among the standardized candidates"
            ),
            "source_pickle": str(MODELS["visit_only"] / "prosit_params.pkl"),
            "source_json": str(MODELS["visit_only"] / "prosit_params.json"),
        },
        {
            "configuration": "static_only",
            "role": "factorial_candidate_not_adopted",
            "adopted": False,
            "reason": "worse activity-rate and 3-gram reproduction",
            "source_pickle": str(MODELS["static_only"] / "prosit_params.pkl"),
            "source_json": str(MODELS["static_only"] / "prosit_params.json"),
        },
        {
            "configuration": "native_only",
            "role": "factorial_candidate_not_adopted",
            "adopted": False,
            "reason": "higher turnaround EMD without a control-flow-fidelity gain",
            "source_pickle": str(MODELS["native_only"] / "prosit_params.pkl"),
            "source_json": str(MODELS["native_only"] / "prosit_params.json"),
        },
        {
            "configuration": "both",
            "role": "investigated_rejected_scenario_robustness",
            "adopted": False,
            "reason": (
                "static plus native state did not improve hold-out validity and "
                "worsened activity-rate and 3-gram reproduction"
            ),
            "source_pickle": str(MODELS["both"] / "prosit_params.pkl"),
            "source_json": str(MODELS["both"] / "prosit_params.json"),
        },
    ]
    pd.DataFrame(model_rows).to_csv(
        MODEL_SELECTION_OUTPUT / "model_role_summary.csv", index=False
    )

    artifacts = {
        "visit_only_source_pickle": MODELS["visit_only"] / "prosit_params.pkl",
        "visit_only_source_json": MODELS["visit_only"] / "prosit_params.json",
        "static_only_source_pickle": MODELS["static_only"] / "prosit_params.pkl",
        "static_only_source_json": MODELS["static_only"] / "prosit_params.json",
        "native_only_source_pickle": MODELS["native_only"] / "prosit_params.pkl",
        "native_only_source_json": MODELS["native_only"] / "prosit_params.json",
        "both_source_pickle": MODELS["both"] / "prosit_params.pkl",
        "both_source_json": MODELS["both"] / "prosit_params.json",
        "visit_only_cap3_baseline": reference_cap3,
        "visit_only_scenario_summary": visit_scenario / "scenario_run_summary.json",
        "both_scenario_summary": both_scenario / "scenario_run_summary.json",
        "scenario_robustness_summary": (
            ROBUSTNESS_OUTPUT / "scenario_state_ablation_summary.json"
        ),
        "standardized_feature_audit": STANDARDIZED_AUDIT_OUTPUT / "audit_manifest.json",
        "reference_holdout_replications": (
            RESULTS / REFERENCE_VALIDATION_LABEL / "mc_replications.csv"
        ),
    }
    missing = [str(path) for path in artifacts.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Cannot freeze provenance; missing: " + ", ".join(missing))

    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": (
            "Freeze the final standardized model roles and exact scenario artifacts "
            "used for thesis result synchronization."
        ),
        "design": {
            "training_test_split": "unchanged shared 80/20 arrival-time split",
            "seeds": list(range(42, 52)),
            "scenario_common_random_numbers": True,
            "rmg_max_concurrency": 3,
            "scenarios": ["baseline", "t22_closed", "demand_plus_20pct"],
            "demand_increase_pct": 20.0,
        },
        "model_roles": {
            "visit_only": {
                "role": "historical reference model",
                "selection_basis": (
                    "parsimony and superior activity-frequency and 3-gram fidelity; "
                    "selection was not based on turnaround EMD alone"
                ),
            },
            "both": {
                "role": "investigated but rejected model",
                "continued_use": (
                    "scenario-response robustness comparator only, not an adopted baseline"
                ),
            },
            "clean_static": {
                "role": "pre-specified post-hoc sensitivity",
                "conclusion": (
                    "no retained demand/utilisation split and worse activity-frequency "
                    "and 3-gram fidelity; no further ad-hoc feature combinations pursued"
                ),
            },
        },
        "evidence": {
            "standardized_2x2": str(PREREQUISITES[0]),
            "clean_static_sensitivity": str(PREREQUISITES[1]),
            "reference_holdout_validation": str(RESULTS / REFERENCE_VALIDATION_LABEL),
            "visit_only_scenarios": str(visit_scenario),
            "both_scenarios": str(both_scenario),
            "scenario_bundle_robustness": str(ROBUSTNESS_OUTPUT),
            "standardized_feature_audit": str(STANDARDIZED_AUDIT_OUTPUT),
        },
        "artifacts": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in artifacts.items()
        },
    }
    manifest = MODEL_SELECTION_OUTPUT / "model_role_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    missing = [str(path) for path in PREREQUISITES if not path.exists()]
    if missing:
        raise FileNotFoundError("Required model-selection evidence is missing: " + ", ".join(missing))
    for configuration, folder in MODELS.items():
        for filename in ("prosit_params.pkl", "prosit_params.json"):
            path = folder / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing standardized {configuration} artifact: {path}")

    python = sys.executable
    for configuration in ("visit_only", "both"):
        label = SCENARIO_LABELS[configuration]
        run_step(
            f"three matched-seed scenarios: {configuration}",
            [
                python,
                "-u",
                "validation/09_multi_seed_scenarios.py",
                "--params",
                str(MODELS[configuration] / "prosit_params.pkl"),
                "--label",
                label,
                "--n-seeds",
                "10",
                "--base-seed",
                "42",
                "--rmg-max-concurrency",
                "3",
            ],
            RESULTS / label / "scenario_run_summary.json",
            scenario_summary=True,
        )

    visit_cap3 = (
        RESULTS
        / SCENARIO_LABELS["visit_only"]
        / "params_baseline_rmg_max_concurrency_3.pkl"
    )
    run_step(
        "ten-seed hold-out validation of the adopted cap-3 reference",
        [
            python,
            "-u",
            "validation/06_multi_seed_ci.py",
            "--params",
            str(visit_cap3),
            "--n-seeds",
            "10",
            "--base-seed",
            "42",
            "--label",
            REFERENCE_VALIDATION_LABEL,
        ],
        RESULTS / REFERENCE_VALIDATION_LABEL / "mc_replications.csv",
    )
    run_step(
        "matched-seed scenario-response robustness",
        [
            python,
            "validation/17_compare_scenario_state_ablation.py",
            "--visit-only",
            str(RESULTS / SCENARIO_LABELS["visit_only"]),
            "--both",
            str(RESULTS / SCENARIO_LABELS["both"]),
            "--output",
            str(ROBUSTNESS_OUTPUT),
        ],
        ROBUSTNESS_OUTPUT / "scenario_state_ablation_summary.json",
    )
    run_step(
        "feature correlation and retained-rule audit from the standardized both bundle",
        [
            python,
            "validation/20_feature_source_audit.py",
            "--full-params",
            str(MODELS["both"] / "prosit_params.pkl"),
            "--output",
            str(STANDARDIZED_AUDIT_OUTPUT),
            "--model-json",
            f"visit_only={MODELS['visit_only'] / 'prosit_params.json'}",
            "--model-json",
            f"static_only={MODELS['static_only'] / 'prosit_params.json'}",
            "--model-json",
            f"native_only={MODELS['native_only'] / 'prosit_params.json'}",
            "--model-json",
            f"both={MODELS['both'] / 'prosit_params.json'}",
        ],
        STANDARDIZED_AUDIT_OUTPUT / "audit_manifest.json",
    )
    manifest = write_provenance()
    print(f"\n[standardized-scenarios] ALL STEPS COMPLETE -> {manifest}", flush=True)


if __name__ == "__main__":
    main()
