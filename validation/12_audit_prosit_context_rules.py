#!/usr/bin/env python3
"""Audit the frozen ProSiT rules for utilisation and demand effects.

The script is read-only with respect to the frozen parameter pickle. It
separates execution-time, waiting-time, resource-selection and transition-
routing rules and reconstructs the process prefix of every transition in the
sequential variant trie. This prevents routing rules from being misreported as
duration or operational-priority rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS = (
    ROOT
    / "baseline"
    / "discovery_params"
    / "params_20260816_214403_train80"
    / "prosit_discovery_workload_sequential_calibrated"
    / "prosit_params.pkl"
)
DEFAULT_RESULTS = ROOT / "validation" / "results"
FOCUS_ACTIVITIES = {"RMG_receive", "RMG_delivery", "HO2_receive", "HO2_delivery"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--label", default="prosit_context_rule_audit")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_decision_rules(value: Any) -> bool:
    return hasattr(value, "rules") and isinstance(value.rules, dict)


def feature_flags(feature: str) -> dict[str, bool]:
    lower = feature.lower()
    return {
        "is_utilisation_feature": "utilization" in lower or "utilisation" in lower,
        "is_demand_feature": "demand" in lower,
        "is_queue_feature": "queue" in lower,
        "is_area_feature": "target_area" in lower,
        "is_complexity_feature": any(
            token in lower
            for token in ("visit_complexity", "n_stops", "n_deliveries", "n_receives", "full_ratio")
        ),
    }


def distribution_name(node: dict[str, Any]) -> str | None:
    distribution = node.get("dist")
    if not distribution:
        return None
    scipy_distribution = distribution[0]
    return getattr(scipy_distribution, "name", type(scipy_distribution).__name__)


def rule_details(model: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not is_decision_rules(model):
        return (
            {
                "model_type": type(model).__name__,
                "n_nodes": 0,
                "n_split_nodes": 0,
                "n_leaves": 1,
                "max_depth": 0,
                "root_is_leaf": True,
                "root_value": float(model) if isinstance(model, (int, float)) else None,
                "root_distribution": None,
                "root_n_sampled": None,
            },
            [],
        )

    rules = model.rules
    if not rules:
        return (
            {
                "model_type": type(model).__name__,
                "n_nodes": 0,
                "n_split_nodes": 0,
                "n_leaves": 0,
                "max_depth": 0,
                "root_is_leaf": False,
                "root_value": None,
                "root_distribution": None,
                "root_n_sampled": None,
            },
            [],
        )

    root_id = 0 if 0 in rules else min(rules)
    queue: deque[tuple[Any, int]] = deque([(root_id, 0)])
    seen: set[Any] = set()
    split_rows: list[dict[str, Any]] = []
    leaf_count = 0
    max_depth = 0
    while queue:
        node_id, depth = queue.popleft()
        if node_id in seen:
            continue
        seen.add(node_id)
        max_depth = max(max_depth, depth)
        node = rules[node_id]
        if "feature" in node:
            feature = str(node["feature"])
            split_rows.append(
                {
                    "node_id": node_id,
                    "depth": depth,
                    "feature": feature,
                    "threshold": node.get("threshold"),
                    **feature_flags(feature),
                }
            )
            for child in node.get("children", {}).values():
                queue.append((child, depth + 1))
        else:
            leaf_count += 1

    root = rules[root_id]
    sampled = root.get("sampled")
    metadata = {
        "model_type": type(model).__name__,
        "n_nodes": len(seen),
        "n_split_nodes": len(split_rows),
        "n_leaves": leaf_count,
        "max_depth": max_depth,
        "root_is_leaf": "feature" not in root,
        "root_value": root.get("value"),
        "root_distribution": distribution_name(root),
        "root_n_sampled": len(sampled) if sampled is not None else None,
    }
    return metadata, split_rows


def rule_leaf_paths(model: Any) -> list[dict[str, Any]]:
    """Return human-readable root-to-leaf conditions for one rule model."""
    if not is_decision_rules(model) or not model.rules:
        return []
    rules = model.rules
    root_id = 0 if 0 in rules else min(rules)
    queue: deque[tuple[Any, tuple[str, ...], tuple[str, ...]]] = deque(
        [(root_id, tuple(), tuple())]
    )
    rows: list[dict[str, Any]] = []
    while queue:
        node_id, conditions, features = queue.popleft()
        node = rules[node_id]
        if "feature" not in node:
            sampled = node.get("sampled")
            rows.append(
                {
                    "leaf_node_id": node_id,
                    "rule_path": " AND ".join(conditions) if conditions else "<all observations>",
                    "path_features": " | ".join(features),
                    "leaf_value": node.get("value"),
                    "leaf_distribution": distribution_name(node),
                    "leaf_n_sampled": len(sampled) if sampled is not None else None,
                }
            )
            continue
        feature = str(node["feature"])
        threshold = node["threshold"]
        children = node.get("children", {})
        if True in children:
            queue.append(
                (
                    children[True],
                    (*conditions, f"{feature} <= {threshold}"),
                    (*features, feature),
                )
            )
        if False in children:
            queue.append(
                (
                    children[False],
                    (*conditions, f"{feature} > {threshold}"),
                    (*features, feature),
                )
            )
    return rows


def transition_contexts(parameters: Any) -> dict[str, dict[str, str]]:
    """Map trie transition names to labels and process prefixes."""
    initial_places = list(parameters.initial_marking.keys())
    queue: deque[tuple[Any, tuple[str, ...]]] = deque((place, tuple()) for place in initial_places)
    place_context: dict[Any, tuple[str, ...]] = {}
    contexts: dict[str, dict[str, str]] = {}
    while queue:
        place, prefix = queue.popleft()
        previous = place_context.get(place)
        if previous is not None:
            if previous != prefix:
                raise RuntimeError(
                    f"The net is not a prefix trie: place {place} has contexts {previous} and {prefix}."
                )
            continue
        place_context[place] = prefix
        for arc in place.out_arcs:
            transition = arc.target
            label = str(transition.label)
            name = str(transition.name)
            contexts[name] = {
                "target_label": label,
                "process_prefix": " > ".join(prefix) if prefix else "<start>",
            }
            if label == "Gate Out":
                continue
            if label not in {"Gate In", "Gate Out", "None"}:
                next_prefix = (*prefix, label)
            else:
                next_prefix = prefix
            for out_arc in transition.out_arcs:
                queue.append((out_arc.target, next_prefix))
    return contexts


def reverse_resource_map(parameters: Any) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for activity, resources in parameters.act_to_resources.items():
        for resource in resources:
            result.setdefault(str(resource), []).append(str(activity))
    return {resource: sorted(activities) for resource, activities in result.items()}


def iter_families(parameters: Any) -> Iterable[tuple[str, dict[Any, Any]]]:
    yield "execution_time", parameters.execution_time_distributions
    yield "waiting_time", parameters.waiting_time_distributions
    yield "resource_selection", parameters.resource_weights
    yield "transition_routing", parameters.transition_weights


def serializable_number(value: Any) -> Any:
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def main() -> None:
    args = parse_args()
    params_path = args.params.resolve()
    output = args.results_root.resolve() / args.label
    output.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(params_path)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with params_path.open("rb") as handle:
        parameters = pickle.load(handle)
    # Some ProSiT dependency versions alter streams at import time. Keep the
    # audit usable as a normal command-line validation step.
    sys.stdout, sys.stderr = original_stdout, original_stderr

    transition_map = transition_contexts(parameters)
    resource_map = reverse_resource_map(parameters)
    inventory_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    leaf_rows: list[dict[str, Any]] = []

    for family, models in iter_families(parameters):
        for raw_key, model in models.items():
            key = str(raw_key)
            if family in {"execution_time", "waiting_time"}:
                target_labels = [key]
                prefix = None
            elif family == "resource_selection":
                target_labels = resource_map.get(key, [])
                prefix = None
            else:
                context = transition_map.get(key, {})
                target_labels = [context.get("target_label", "<unknown>")]
                prefix = context.get("process_prefix", "<unmapped>")

            metadata, model_splits = rule_details(model)
            features = sorted({row["feature"] for row in model_splits})
            flags = {
                name: any(row[name] for row in model_splits)
                for name in (
                    "is_utilisation_feature",
                    "is_demand_feature",
                    "is_queue_feature",
                    "is_area_feature",
                    "is_complexity_feature",
                )
            }
            focus_target = bool(FOCUS_ACTIVITIES.intersection(target_labels))
            inventory_rows.append(
                {
                    "family": family,
                    "model_key": key,
                    "target_labels": " | ".join(target_labels),
                    "process_prefix": prefix,
                    "focus_target": focus_target,
                    "split_features": " | ".join(features),
                    **flags,
                    **metadata,
                }
            )
            for split in model_splits:
                split_rows.append(
                    {
                        "family": family,
                        "model_key": key,
                        "target_labels": " | ".join(target_labels),
                        "process_prefix": prefix,
                        "focus_target": focus_target,
                        **split,
                    }
                )
            for leaf in rule_leaf_paths(model):
                leaf_rows.append(
                    {
                        "family": family,
                        "model_key": key,
                        "target_labels": " | ".join(target_labels),
                        "process_prefix": prefix,
                        "focus_target": focus_target,
                        "model_uses_utilisation": flags["is_utilisation_feature"],
                        "model_uses_demand": flags["is_demand_feature"],
                        **leaf,
                    }
                )

    inventory = pd.DataFrame(inventory_rows)
    split_inventory = pd.DataFrame(split_rows)
    leaf_inventory = pd.DataFrame(leaf_rows)
    inventory.to_csv(output / "rule_inventory.csv", index=False)
    split_inventory.to_csv(output / "split_feature_inventory.csv", index=False)
    leaf_inventory.to_csv(output / "rule_leaf_paths.csv", index=False)

    focus_execution = inventory[
        (inventory["family"] == "execution_time")
        & inventory["model_key"].isin(FOCUS_ACTIVITIES)
    ].copy()
    focus_execution.to_csv(output / "focus_execution_models.csv", index=False)

    focused_context = split_inventory[
        split_inventory["focus_target"]
        | split_inventory["is_utilisation_feature"]
        | split_inventory["is_demand_feature"]
    ].copy()
    focused_context.to_csv(output / "focused_context_splits.csv", index=False)

    family_summary: dict[str, dict[str, Any]] = {}
    for family, frame in inventory.groupby("family"):
        family_splits = split_inventory[split_inventory["family"] == family]
        family_summary[family] = {
            "n_models": int(len(frame)),
            "n_decision_rule_models": int((frame["model_type"] == "DecisionRules").sum()),
            "n_models_with_splits": int((frame["n_split_nodes"] > 0).sum()),
            "n_split_nodes": int(frame["n_split_nodes"].sum()),
            "n_utilisation_splits": int(family_splits["is_utilisation_feature"].sum()),
            "n_demand_splits": int(family_splits["is_demand_feature"].sum()),
            "n_queue_splits": int(family_splits["is_queue_feature"].sum()),
            "n_area_splits": int(family_splits["is_area_feature"].sum()),
            "n_complexity_splits": int(family_splits["is_complexity_feature"].sum()),
        }

    feature_counts = (
        split_inventory.groupby(["family", "feature"]).size().reset_index(name="count")
        .sort_values(["family", "count", "feature"], ascending=[True, False, True])
    )
    feature_counts.to_csv(output / "split_feature_counts.csv", index=False)

    context_effects = split_inventory[
        split_inventory["is_utilisation_feature"] | split_inventory["is_demand_feature"]
    ].copy()
    context_effects.to_csv(output / "utilisation_demand_rule_splits.csv", index=False)
    context_leaf_paths = leaf_inventory[
        leaf_inventory["model_uses_utilisation"] | leaf_inventory["model_uses_demand"]
    ].copy()
    context_leaf_paths.to_csv(output / "utilisation_demand_rule_leaf_paths.csv", index=False)

    focus_context_leaf_paths = context_leaf_paths[context_leaf_paths["focus_target"]].copy()
    focus_context_leaf_paths.to_csv(output / "focus_context_rule_leaf_paths.csv", index=False)

    focus_models_json = []
    for row in focus_execution.to_dict(orient="records"):
        focus_models_json.append(
            {
                "activity": row["model_key"],
                "n_split_nodes": int(row["n_split_nodes"]),
                "root_value_min": serializable_number(row["root_value"]),
                "distribution": row["root_distribution"],
                "n_sampled": int(row["root_n_sampled"]) if pd.notna(row["root_n_sampled"]) else None,
            }
        )

    summary = {
        "source_pickle": str(params_path),
        "source_pickle_sha256": source_hash,
        "rules_mode": bool(parameters.rules_mode),
        "use_workload_features": bool(parameters.use_workload_features),
        "net_places": len(parameters.net.places),
        "net_transitions": len(parameters.net.transitions),
        "mapped_transition_contexts": len(transition_map),
        "family_summary": family_summary,
        "focused_execution_models": focus_models_json,
        "key_interpretation": {
            "execution_duration_uses_utilisation_or_demand": bool(
                family_summary["execution_time"]["n_utilisation_splits"]
                or family_summary["execution_time"]["n_demand_splits"]
            ),
            "waiting_duration_uses_utilisation_or_demand": bool(
                family_summary["waiting_time"]["n_utilisation_splits"]
                or family_summary["waiting_time"]["n_demand_splits"]
            ),
            "resource_selection_uses_utilisation_or_demand": bool(
                family_summary["resource_selection"]["n_utilisation_splits"]
                or family_summary["resource_selection"]["n_demand_splits"]
            ),
            "transition_routing_uses_utilisation_or_demand": bool(
                family_summary["transition_routing"]["n_utilisation_splits"]
                or family_summary["transition_routing"]["n_demand_splits"]
            ),
            "routing_effect_is_duration_prioritisation_evidence": False,
        },
    }
    (output / "rule_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    family_lines = []
    for family in ("execution_time", "waiting_time", "resource_selection", "transition_routing"):
        values = family_summary[family]
        family_lines.append(
            f"| {family} | {values['n_models']} | {values['n_split_nodes']} | "
            f"{values['n_utilisation_splits']} | {values['n_demand_splits']} | "
            f"{values['n_queue_splits']} | {values['n_area_splits']} |"
        )

    focus_lines = []
    for item in focus_models_json:
        focus_lines.append(
            f"| {item['activity']} | {item['root_value_min']:.3f} | "
            f"{item['distribution']} | {item['n_sampled']} | {item['n_split_nodes']} |"
        )

    context_lines = []
    for row in context_effects.to_dict(orient="records"):
        context_lines.append(
            f"| {row['model_key']} | {row['target_labels']} | {row['process_prefix']} | "
            f"{row['feature']} | {row['threshold']} |"
        )
    if not context_lines:
        context_lines = ["| - | - | - | No utilisation/demand splits | - |"]

    focus_path_lines = []
    for row in focus_context_leaf_paths.to_dict(orient="records"):
        focus_path_lines.append(
            f"| {row['model_key']} | {row['target_labels']} | {row['process_prefix']} | "
            f"{row['rule_path']} | {row['leaf_value']:.6g} |"
        )
    if not focus_path_lines:
        focus_path_lines = ["| - | - | - | No focused context paths | - |"]

    report = f"""# Frozen ProSiT context-rule audit

Source SHA-256: `{source_hash}`

## Rule-layer inventory

| Layer | Models | Split nodes | Utilisation | Demand | Queue | Area |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(family_lines)}

## Focused operation-response models

| Activity | Root value [min] | Distribution | Training samples | Splits |
|---|---:|---|---:|---:|
{chr(10).join(focus_lines)}

The focused execution-time models are single-leaf distributions when their split count
is zero. In that case, ProSiT did not retain utilisation, demand, operation type, block or
another context feature for the activity's execution-duration parameter.

## Where utilisation or demand actually occurs

| Transition/model | Target label | Process prefix | Feature | Threshold |
|---|---|---|---|---:|
{chr(10).join(context_lines)}

These rows must be interpreted according to their layer. In particular, a split in
`transition_routing` changes the relative next-transition weight at a specific process
prefix. It does **not** change the ready-to-completion duration, prove that a crane
prioritises receive jobs, or establish a physical capacity response.

### Focused context-rule leaf paths

| Model | Target | Prefix | Complete path | Relative leaf weight |
|---|---|---|---|---:|
{chr(10).join(focus_path_lines)}

For transition routing, these are raw **relative weights**, not normalized probabilities.
The probability of a next step also depends on all other transitions enabled at that
prefix. ProSiT follows the `<=` branch when the stated condition is true.

## Defensible conclusion

The audit provides exact white-box provenance for every selected feature. Absence of a
feature from the final trees means only that the configured discovery procedure did not
retain it in that model layer; it is not proof that the operational effect does not exist.
The separate receive-versus-delivery analysis tests the observational association more
directly and keeps that evidence distinct from the frozen simulator's learned rules.
"""
    (output / "rule_audit_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote frozen-rule audit to: {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
