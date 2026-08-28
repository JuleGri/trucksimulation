"""Utilities used by the CTB reviewer notebook.

The functions in this module deliberately load the frozen ProSiT bundles and
reuse the thesis scenario runner's metric and contract functions. They do not
perform model discovery and they do not require the confidential source log.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr
from copy import deepcopy
from datetime import datetime
import hashlib
import io
import importlib.util
import json
import pickle
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = PACKAGE_ROOT / "runtime"
MODELS_DIR = PACKAGE_ROOT / "models"
EXPECTED_DIR = PACKAGE_ROOT / "expected_results"
THESIS_RESULTS_DIR = PACKAGE_ROOT / "thesis_results"
OUTPUTS_DIR = PACKAGE_ROOT / "outputs"

# Jupyter and Colab may save execution counts and outputs back into a notebook
# while it is running. These two interface files are therefore intentionally
# excluded from the frozen-artifact integrity check.
MUTABLE_NOTEBOOKS = {
    "CTB_colab_prosit_load_and_run.ipynb",
    "CTB_local_prosit_load_and_run.ipynb",
}

MODEL_FILES = {
    "discovered_source": MODELS_DIR / "params_discovered_rules_workload.pkl",
    "rules_only_workload_blind": MODELS_DIR / "params_rules_only_revised.pkl",
    "baseline": MODELS_DIR / "params_baseline_rmg_max_concurrency_3.pkl",
    "t22_closed": MODELS_DIR / "params_t22_closed.pkl",
    "demand_plus_20pct": MODELS_DIR / "params_demand_plus_20pct.pkl",
}

RMG_ACTIVITIES = ("RMG_receive", "RMG_delivery", "RMG_mixed")
EXPERIMENT_MODELS = ("baseline", "t22_closed", "demand_plus_20pct")

ANALYSIS_MODEL_FILES = {
    "no_rules": MODELS_DIR / "params_no_rules.pkl",
    "rules_only": MODELS_DIR / "params_rules_only_revised.pkl",
    "rules_workload": MODELS_DIR / "params_discovered_rules_workload.pkl",
    "precision_repair": THESIS_RESULTS_DIR / "structural_repair" / "prosit_params_gate_only_restricted.pkl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    with (PACKAGE_ROOT / "model_manifest.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def verify_package_files() -> pd.DataFrame:
    """Verify every frozen model and expected result before unpickling."""

    manifest = load_manifest()
    rows: list[dict[str, Any]] = []
    for relative, metadata in manifest["files"].items():
        if relative in MUTABLE_NOTEBOOKS:
            continue
        path = PACKAGE_ROOT / relative
        exists = path.is_file()
        actual = sha256(path) if exists else None
        expected = metadata["sha256"]
        rows.append(
            {
                "file": relative,
                "role": metadata["role"],
                "exists": exists,
                "sha256_matches": actual == expected,
                "size_mb": round(path.stat().st_size / 1024**2, 3) if exists else np.nan,
            }
        )
    report = pd.DataFrame(rows)
    failures = report[~report["exists"] | ~report["sha256_matches"]]
    if not failures.empty:
        raise RuntimeError(f"Package integrity check failed:\n{failures.to_string(index=False)}")
    return report


def load_models() -> dict[str, Any]:
    """Load the trusted parameter pickles after their hashes are checked."""

    verify_package_files()
    models: dict[str, Any] = {}
    for name, path in MODEL_FILES.items():
        with path.open("rb") as handle:
            models[name] = pickle.load(handle)
    return models


def load_analysis_models() -> dict[str, Any]:
    """Load the three state-ablation bundles and the structural repair."""

    verify_package_files()
    models: dict[str, Any] = {}
    for name, path in ANALYSIS_MODEL_FILES.items():
        with path.open("rb") as handle:
            models[name] = pickle.load(handle)
    return models


def _place_name(place: Any) -> str:
    return str(getattr(place, "name", place))


def _transition_key(transition: Any) -> str:
    return f"{getattr(transition, 'name', transition)}|{getattr(transition, 'label', None)}"


def petri_net_signature(params: Any) -> str:
    """Return a stable structural signature for a ProSiT parameter bundle."""

    net = params.net
    places = sorted(_place_name(place) for place in net.places)
    transitions = sorted(_transition_key(transition) for transition in net.transitions)
    arcs = sorted(
        f"{_place_name(arc.source) if arc.source in net.places else _transition_key(arc.source)}"
        f"->{_place_name(arc.target) if arc.target in net.places else _transition_key(arc.target)}"
        for arc in net.arcs
    )
    payload = json.dumps(
        {"places": places, "transitions": transitions, "arcs": arcs},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def petri_net_transitions(params: Any) -> pd.DataFrame:
    rows = []
    for transition in params.net.transitions:
        rows.append(
            {
                "transition_id": str(transition.name),
                "label": transition.label if transition.label is not None else "tau (silent)",
                "silent": transition.label is None,
                "incoming_arcs": len(transition.in_arcs),
                "outgoing_arcs": len(transition.out_arcs),
            }
        )
    return pd.DataFrame(rows).sort_values(["silent", "label", "transition_id"]).reset_index(drop=True)


def _active_calendar_slots(params: Any, resource: str) -> int | None:
    calendar = params.calendars.get(resource)
    if calendar is None:
        return None
    return int(sum(bool(active) for hours in calendar.values() for active in hours.values()))


def _rmg_resources(params: Any) -> list[str]:
    return sorted(
        {
            resource
            for activity in RMG_ACTIVITIES
            for resource in params.act_to_resources[activity]
        }
    )


def _arrival_samples(params: Any) -> np.ndarray:
    rules = getattr(params.arrival_time_distribution, "rules", {})
    values = rules.get(0, {}).get("sampled", [])
    return np.asarray(values, dtype=float)


def model_summary(models: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, params in models.items():
        rmg_resources = _rmg_resources(params)
        arrivals = _arrival_samples(params)
        rows.append(
            {
                "model": name,
                "rules_mode": bool(params.rules_mode),
                "workload_features": bool(params.use_workload_features),
                "places": len(params.net.places),
                "transitions": len(params.net.transitions),
                "arcs": len(params.net.arcs),
                "activities": len(params.act_to_resources),
                "resources": len(params.resources),
                "rmg_resources": len(rmg_resources),
                "aggregate_rmg_max_concurrency": int(
                    sum(params.max_concurrency[resource] for resource in rmg_resources)
                ),
                "t22_eligible_for_rmg": any(
                    "T22" in params.act_to_resources[activity]
                    for activity in RMG_ACTIVITIES
                ),
                "t22_active_calendar_slots": _active_calendar_slots(params, "T22"),
                "arrival_empirical_samples": int(arrivals.size),
                "mean_working_inter_arrival_min": float(arrivals.mean()),
                "petri_net_signature": petri_net_signature(params),
            }
        )
    return pd.DataFrame(rows)


def _rule_statistics(model: Any) -> dict[str, Any]:
    rules = getattr(model, "rules", None)
    if not isinstance(rules, dict):
        return {
            "model_type": "constant",
            "split_nodes": 0,
            "leaf_nodes": 1,
            "features": "",
            "stored_runtime_samples": 0,
        }
    nodes = [node for node in rules.values() if isinstance(node, dict)]
    features = sorted({str(node["feature"]) for node in nodes if "feature" in node})
    return {
        "model_type": "decision_rules",
        "split_nodes": sum("feature" in node for node in nodes),
        "leaf_nodes": sum("value" in node for node in nodes),
        "features": ", ".join(features),
        "stored_runtime_samples": int(
            sum(len(node.get("sampled", [])) for node in nodes)
        ),
    }


def parameter_component_inventory(params: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    families = {
        "routing": params.transition_weights,
        "execution_time": params.execution_time_distributions,
        "waiting_time": params.waiting_time_distributions,
        "resource_selection": params.resource_weights,
        "arrival_time": {"arrival": params.arrival_time_distribution},
    }
    for family, mapping in families.items():
        for component, model in mapping.items():
            row = {"family": family, "component": str(component)}
            row.update(_rule_statistics(model))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["family", "component"]).reset_index(drop=True)


def assert_model_contracts(models: dict[str, Any]) -> pd.DataFrame:
    """Assert the structural and scenario-difference contract stated in the thesis."""

    source = models["discovered_source"]
    baseline = models["baseline"]
    t22 = models["t22_closed"]
    demand = models["demand_plus_20pct"]
    signatures = {name: petri_net_signature(model) for name, model in models.items()}
    checks = {
        "source and three scenario bundles use the identical Petri net": len(
            {signatures[name] for name in ("discovered_source", "baseline", "t22_closed", "demand_plus_20pct")}
        ) == 1,
        "baseline RMG capacity is 22 x 3 = 66": model_summary({"x": baseline}).iloc[0]["aggregate_rmg_max_concurrency"] == 66,
        "source overlap-derived RMG capacity is 100": model_summary({"x": source}).iloc[0]["aggregate_rmg_max_concurrency"] == 100,
        "T22 is absent from every scenario-A RMG pool": all(
            "T22" not in t22.act_to_resources[activity] for activity in RMG_ACTIVITIES
        ),
        "T22 remains available in baseline": all(
            "T22" in baseline.act_to_resources[activity] for activity in RMG_ACTIVITIES
        ),
        "T22 calendar is fully closed only in scenario A": (
            _active_calendar_slots(t22, "T22") == 0
            and _active_calendar_slots(baseline, "T22") > 0
        ),
        "scenario B retains baseline resource mappings": demand.act_to_resources == baseline.act_to_resources,
        "scenario B retains baseline resource capacities": demand.max_concurrency == baseline.max_concurrency,
        "scenario B scales mean inter-arrival time by 1/1.20": np.isclose(
            _arrival_samples(demand).mean(),
            _arrival_samples(baseline).mean() / 1.20,
        ),
        "scenario A retains the baseline arrival sample": np.array_equal(
            _arrival_samples(t22), _arrival_samples(baseline)
        ),
    }
    report = pd.DataFrame(
        [{"contract": name, "passed": bool(passed)} for name, passed in checks.items()]
    )
    if not report["passed"].all():
        raise AssertionError(f"Model contract failed:\n{report.to_string(index=False)}")
    return report


def export_and_reload_official_json(params: Any, output_path: Path) -> dict[str, Any]:
    """Demonstrate ProSiT's documented JSON API and report its CTB caveat."""

    from prosit import SimulatorParameters

    output_path.parent.mkdir(parents=True, exist_ok=True)
    params.to_json(str(output_path))
    restored = SimulatorParameters(params.net, params.initial_marking, params.final_marking)
    original_samples = _arrival_samples(params)
    report = {
        "json_file": str(output_path),
        "json_export_succeeded": output_path.is_file(),
        "json_import_succeeded": False,
        "original_arrival_sample_count": int(original_samples.size),
        "restored_arrival_sample_count": 0,
        "exact_ctb_runtime_state_restored": False,
        "import_error": None,
    }
    try:
        restored.from_json(str(output_path))
    except Exception as exc:  # The failure is part of the audited ProSiT 1.0.3 contract.
        report["import_error"] = f"{type(exc).__name__}: {exc}"
        return report

    restored_samples = _arrival_samples(restored)
    report.update(
        {
            "json_import_succeeded": True,
            "rules_mode_restored": restored.rules_mode == params.rules_mode,
            "workload_flag_restored": (
                restored.use_workload_features == params.use_workload_features
            ),
            "resource_mapping_restored": restored.act_to_resources == params.act_to_resources,
            "max_concurrency_restored": restored.max_concurrency == params.max_concurrency,
            "restored_arrival_sample_count": int(restored_samples.size),
            "exact_ctb_runtime_state_restored": bool(
                original_samples.size == restored_samples.size
                and np.array_equal(original_samples, restored_samples)
            ),
        }
    )
    return report


def _load_scenario_module():
    path = RUNTIME_ROOT / "validation" / "09_multi_seed_scenarios.py"
    if not path.is_file():
        raise FileNotFoundError(
            "The frozen reviewer runtime is incomplete; " f"missing {path}."
        )
    spec = importlib.util.spec_from_file_location("ctb_scenario_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import scenario runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _stable_transition_sampling():
    """Make fixed-seed ProSiT routing stable after pickle reloads.

    ProSiT 1.0.3 receives enabled transitions from a set, while PM4Py hashes
    transition objects by identity. The resulting list order may therefore
    change between Python processes. Canonicalising that list immediately
    before ProSiT performs its existing weighted draw preserves the model and
    probabilities while making the seeded run reproducible.
    """

    import prosit.simulator as prosit_simulator

    original = prosit_simulator.return_fired_transition

    def stable_return_fired_transition(transition_weights, enabled_transitions):
        ordered = sorted(
            enabled_transitions,
            key=lambda transition: (
                str(transition.name),
                "" if transition.label is None else str(transition.label),
            ),
        )
        return original(transition_weights, ordered)

    prosit_simulator.return_fired_transition = stable_return_fired_transition
    try:
        yield
    finally:
        prosit_simulator.return_fired_transition = original


def run_saved_models(
    *,
    mode: str = "full",
    output_label: str | None = None,
) -> Path:
    """Run the exact saved baseline and scenario bundles.

    ``full`` reproduces 10 seeds x 3 models x 17,892 cases. ``smoke`` runs two
    seeds x 250 cases to verify mechanics only; it cannot reproduce thesis KPIs.
    """

    if mode not in {"full", "smoke"}:
        raise ValueError("mode must be 'full' or 'smoke'")
    models = load_models()
    assert_model_contracts(models)
    runner = _load_scenario_module()
    manifest = load_manifest()["experiment"]
    seeds = manifest["seeds"] if mode == "full" else manifest["seeds"][:2]
    n_traces = int(manifest["n_traces"] if mode == "full" else 250)
    t_start = datetime.fromisoformat(manifest["t_start"])
    label = output_label or f"{mode}_reproduction"
    out_dir = OUTPUTS_DIR / label
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    started = datetime.now().astimezone()
    for seed in seeds:
        print(f"seed={seed}")
        for scenario_name in EXPERIMENT_MODELS:
            print(f"  {scenario_name} ...", end=" ", flush=True)
            with _stable_transition_sampling(), redirect_stderr(io.StringIO()):
                simulated = runner.simulate_ctb(
                    deepcopy(models[scenario_name]),
                    n_traces=n_traces,
                    t_start=t_start,
                    seed=int(seed),
                    timestamp_resolution="min",
                )
            metrics = runner._summarize_simulation(
                simulated,
                seed=int(seed),
                scenario=scenario_name,
            )
            audit = runner._contract_audit(
                simulated,
                seed=int(seed),
                scenario=scenario_name,
                n_traces=n_traces,
                blocked_blocks={"T22"},
            )
            runner._assert_audit(audit, metrics)
            rows.append(metrics)
            audits.append(audit)
            print(f"turnaround={metrics['mean_turnaround_min']:.3f} min; PASS")

    replications = pd.DataFrame(rows)
    contracts = pd.DataFrame(audits)
    deltas = runner._paired_deltas(replications)
    kpi_summary = runner._scenario_kpi_summary(replications)
    delta_summary = runner._paired_delta_summary(deltas)

    replications.to_csv(out_dir / "scenario_replications.csv", index=False)
    contracts.to_csv(out_dir / "scenario_contracts.csv", index=False)
    deltas.to_csv(out_dir / "scenario_paired_deltas.csv", index=False)
    kpi_summary.to_csv(out_dir / "scenario_kpi_summary.csv", index=False)
    delta_summary.to_csv(out_dir / "scenario_paired_delta_summary.csv", index=False)
    runner._plot_paired_deltas(
        delta_summary,
        figures_dir / "scenario_paired_deltas_ci.png",
    )
    runner._plot_t22_receive_delivery(
        delta_summary,
        figures_dir / "t22_receive_delivery_deltas_ci.png",
    )
    summary = {
        "mode": mode,
        "started_at": started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "n_traces": n_traces,
        "seeds": seeds,
        "t_start": t_start.isoformat(),
        "all_contracts_passed": True,
        "model_files": {name: str(path) for name, path in MODEL_FILES.items()},
    }
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return out_dir


def compare_with_frozen_results(output_dir: Path) -> pd.DataFrame:
    """Compare a complete fresh run with the frozen thesis result tables."""

    comparisons = []
    for filename in (
        "scenario_replications.csv",
        "scenario_contracts.csv",
        "scenario_kpi_summary.csv",
        "scenario_paired_deltas.csv",
        "scenario_paired_delta_summary.csv",
    ):
        expected = pd.read_csv(EXPECTED_DIR / filename)
        actual = pd.read_csv(output_dir / filename)
        same_columns = list(expected.columns) == list(actual.columns)
        same_shape = expected.shape == actual.shape
        equal = False
        max_numeric_difference = np.nan
        if same_columns and same_shape:
            numeric = expected.select_dtypes(include=[np.number]).columns
            text = [column for column in expected.columns if column not in numeric]
            numeric_equal = np.allclose(
                expected[numeric].to_numpy(dtype=float),
                actual[numeric].to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-10,
                equal_nan=True,
            )
            text_equal = expected[text].fillna("").equals(actual[text].fillna(""))
            equal = bool(numeric_equal and text_equal)
            if len(numeric):
                max_numeric_difference = float(
                    np.nanmax(
                        np.abs(
                            expected[numeric].to_numpy(dtype=float)
                            - actual[numeric].to_numpy(dtype=float)
                        )
                    )
                )
        comparisons.append(
            {
                "file": filename,
                "same_shape": same_shape,
                "same_columns": same_columns,
                "values_match": equal,
                "max_abs_numeric_difference": max_numeric_difference,
            }
        )
    report = pd.DataFrame(comparisons)
    if not report["values_match"].all():
        raise AssertionError(f"Reproduction differs from frozen results:\n{report.to_string(index=False)}")
    return report


def package_versions() -> pd.DataFrame:
    import importlib.metadata as metadata

    packages = [
        "prosit-pm",
        "pm4py",
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "joblib",
        "river",
        "matplotlib",
        "typing-extensions",
    ]
    return pd.DataFrame(
        [{"package": name, "version": metadata.version(name)} for name in packages]
    )


def frozen_evidence_inventory() -> pd.DataFrame:
    """List the non-confidential evidence shipped for every thesis claim."""

    rows = []
    for path in sorted(THESIS_RESULTS_DIR.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "file": path.relative_to(PACKAGE_ROOT).as_posix(),
                    "size_kb": round(path.stat().st_size / 1024, 1),
                    "sha256": sha256(path),
                }
            )
    return pd.DataFrame(rows)


def reconstruct_historical_ablation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute the three-configuration headline means and paired contrasts.

    The per-seed tables are outputs of validation against the confidential
    hold-out log. Shipping them permits exact arithmetic reconstruction without
    disclosing terminal records. It does not claim to rerun event-level
    validation from the raw log.
    """

    from scipy.stats import t

    folders = {
        "no_rules": THESIS_RESULTS_DIR / "historical" / "no_rules",
        "rules_only": THESIS_RESULTS_DIR / "historical" / "rules_only",
        "rules_workload": THESIS_RESULTS_DIR / "historical" / "rules_workload",
    }
    metrics = (
        "case_turnaround_emd_min",
        "case_turnaround_sim_mean",
        "case_turnaround_sim_p90",
        "yard_service_time_emd_frequency_weighted_min",
        "yard_activity_rate_l1_error",
        "gate_only_cases",
    )
    replications = {
        name: pd.read_csv(folder / "mc_replications.csv")
        for name, folder in folders.items()
    }

    summary_rows = []
    for config, frame in replications.items():
        for metric in metrics:
            values = frame[metric].astype(float).dropna()
            half_width = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
            summary_rows.append(
                {
                    "configuration": config,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "ci95_lo": float(values.mean() - half_width),
                    "ci95_hi": float(values.mean() + half_width),
                }
            )

    contrast_rows = []
    for contrast, left_name, right_name in (
        ("rules_only - no_rules", "rules_only", "no_rules"),
        ("rules_workload - rules_only", "rules_workload", "rules_only"),
    ):
        paired = replications[left_name].merge(
            replications[right_name], on="seed", suffixes=("_left", "_right"), validate="one_to_one"
        )
        for metric in metrics:
            values = paired[f"{metric}_left"] - paired[f"{metric}_right"]
            half_width = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
            contrast_rows.append(
                {
                    "contrast": contrast,
                    "metric": metric,
                    "n": len(values),
                    "mean_delta": float(values.mean()),
                    "ci95_lo": float(values.mean() - half_width),
                    "ci95_hi": float(values.mean() + half_width),
                    "resolved": bool(values.mean() - half_width > 0 or values.mean() + half_width < 0),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows)


def load_claim_evidence() -> dict[str, Any]:
    """Load the compact evidence objects behind the manuscript's main claims."""

    def read_json(relative: str) -> Any:
        with (THESIS_RESULTS_DIR / relative).open(encoding="utf-8") as handle:
            return json.load(handle)

    evidence: dict[str, Any] = {
        "temporal_transfer": read_json("temporal_transfer/temporal_transfer_summary.json"),
        "structural_repair": read_json("structural_repair/structural_repair_summary.json"),
        "bottleneck": read_json("bottleneck/bottleneck_analysis_summary.json"),
        "capacity_pressure": read_json("capacity_pressure/rmg_capacity_pressure_summary.json"),
        "bottleneck_ranking": pd.read_csv(
            THESIS_RESULTS_DIR / "bottleneck" / "activity_bottleneck_summary.csv"
        ),
        "capacity_by_block": pd.read_csv(
            THESIS_RESULTS_DIR / "capacity_pressure" / "rmg_block_capacity_pressure.csv"
        ),
    }
    for state in ("rules_only", "rules_workload"):
        folder = THESIS_RESULTS_DIR / "scenarios" / state
        if folder.is_dir():
            evidence[f"scenario_deltas_{state}"] = pd.read_csv(
                folder / "scenario_paired_delta_summary.csv"
            )
    did = THESIS_RESULTS_DIR / "scenario_state_ablation" / "scenario_response_difference_in_differences.csv"
    if did.is_file():
        evidence["scenario_state_ablation"] = pd.read_csv(did)
    return evidence
