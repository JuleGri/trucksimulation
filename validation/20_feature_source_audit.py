#!/usr/bin/env python3
"""Audit CTB feature redundancy and the rules retained by frozen ProSiT models.

The audit is descriptive: correlation is not interpreted as causation or as a
requirement that predictors be independent. Static case attributes are audited
once per truck visit. Their association with ProSiT's native ``res_workload``
and ``queue_length`` is audited at the aligned event-decision level.

Only aggregate tables are written. Case identifiers and event-level feature
rows never leave memory.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from _eventlog_contract import (  # noqa: E402
    CASE_COL,
    PROSIT_CASE_ATTRIBUTE_ALLOWLIST,
    select_prosit_dataframe,
    to_pm4py_event_log,
)
from prosit.utils.common_utils import build_df_features  # noqa: E402


DEFAULT_TRAIN = REPO_ROOT / "data/processed/CTB/s6_train.csv"
DEFAULT_FULL_PARAMS = (
    REPO_ROOT
    / "baseline/discovery_params/params_20260816_214403_train80"
    / "prosit_discovery_workload_inductive_calibrated/prosit_params.pkl"
)
DEFAULT_OUTPUT = REPO_ROOT / "validation/results/feature_source_audit_20260829"

CLUSTERS = {
    "visit_process": {
        "n_containers", "n_stops", "n_deliveries", "n_receives",
        "visit_complexity", "process_flow_type",
    },
    "container": {"has_hazardous", "has_reefer", "full_ratio"},
    "location": {"primary_target_area", "target_area"},
    "demand": {
        "gate_demand", "rmg_demand", "vc_demand", "mt_demand",
        "target_demand", "target_demand_bin",
    },
    "utilisation": {
        "gate_utilization", "rmg_utilization", "vc_utilization",
        "mt_utilization", "target_utilization", "target_utilization_bin",
        "target_rank", "target_rank_group",
    },
    "native_dynamic": {"res_workload", "workload", "queue_length"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--full-params", type=Path, default=DEFAULT_FULL_PARAMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model-json",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Frozen ProSiT JSON to inventory. Repeat for multiple models.",
    )
    return parser.parse_args()


def cluster_for(feature: str) -> str:
    raw = feature.split(" = ", 1)[0]
    for cluster, names in CLUSTERS.items():
        if raw in names:
            return cluster
    if raw in {"hour", "weekday", "is_weekend"}:
        return "calendar"
    if raw.startswith("resource") or raw.startswith("handover_from_"):
        return "resource_history"
    if raw.startswith("last_activity_") or raw.startswith("waiting_activity"):
        return "activity_history"
    return "other_or_activity_history"


def mean_ci_description(values: pd.Series) -> dict:
    sample = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "n": int(len(sample)),
        "mean": float(sample.mean()) if len(sample) else np.nan,
        "sd": float(sample.std(ddof=1)) if len(sample) > 1 else 0.0,
        "min": float(sample.min()) if len(sample) else np.nan,
        "max": float(sample.max()) if len(sample) else np.nan,
    }


def spearman_rows(frame: pd.DataFrame, level: str) -> list[dict]:
    numeric = frame.select_dtypes(include=[np.number, "bool"]).copy()
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    corr = numeric.corr(method="spearman", min_periods=50)
    rows = []
    for left, right in combinations(corr.columns, 2):
        value = corr.loc[left, right]
        if pd.isna(value):
            continue
        rows.append({
            "level": level,
            "feature_left": left,
            "cluster_left": cluster_for(left),
            "feature_right": right,
            "cluster_right": cluster_for(right),
            "spearman_rho": float(value),
            "abs_spearman_rho": float(abs(value)),
        })
    return rows


def vif_rows(frame: pd.DataFrame) -> list[dict]:
    numeric = frame.select_dtypes(include=[np.number, "bool"]).copy()
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    z = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, np.nan)
    z = z.dropna(axis=1).to_numpy(float)
    names = list(numeric.columns[numeric.std(ddof=0).gt(0)])
    rows = []
    for index, name in enumerate(names):
        y = z[:, index]
        x = np.delete(z, index, axis=1)
        x = np.column_stack([np.ones(len(x)), x])
        fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
        denominator = float(np.square(y - y.mean()).sum())
        r2 = 1.0 - float(np.square(y - fitted).sum()) / denominator
        vif = np.inf if r2 >= 1.0 else 1.0 / max(1.0 - r2, 1e-12)
        rows.append({
            "feature": name,
            "cluster": cluster_for(name),
            "vif": float(vif),
            "multiple_r2": float(r2),
        })
    return rows


def cluster_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()
    work = pairwise.copy()
    work["cluster_pair"] = work.apply(
        lambda row: " | ".join(sorted((row["cluster_left"], row["cluster_right"]))),
        axis=1,
    )
    return (
        work.groupby(["level", "cluster_pair"], as_index=False)
        .agg(
            n_pairs=("abs_spearman_rho", "size"),
            mean_abs_rho=("abs_spearman_rho", "mean"),
            median_abs_rho=("abs_spearman_rho", "median"),
            max_abs_rho=("abs_spearman_rho", "max"),
        )
        .sort_values(["level", "max_abs_rho"], ascending=[True, False])
    )


def walk_features(value, family: str, path: str = ""):
    if isinstance(value, dict):
        if isinstance(value.get("feature"), str):
            yield family, path, value["feature"]
        for key, child in value.items():
            next_path = f"{path}/{key}" if path else str(key)
            yield from walk_features(child, family, next_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_features(child, family, f"{path}/{index}")


def inventory_rows(values: list[str]) -> list[dict]:
    rows = []
    for specification in values:
        if "=" not in specification:
            raise ValueError(f"Invalid --model-json {specification!r}; expected NAME=PATH")
        model, raw_path = specification.split("=", 1)
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for family, family_payload in payload.items():
            for _, node_path, feature in walk_features(family_payload, family):
                rows.append({
                    "model": model,
                    "family": family,
                    "feature": feature,
                    "base_feature": feature.split(" = ", 1)[0],
                    "cluster": cluster_for(feature),
                    "node_path": node_path,
                })
    return rows


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.train)
    selected, _ = select_prosit_dataframe(raw, label="feature-source audit train")
    case_columns = [
        column for column in PROSIT_CASE_ATTRIBUTE_ALLOWLIST
        if column in selected.columns
    ]
    cases = selected.groupby(CASE_COL, sort=False)[case_columns].first()

    missingness = []
    for column in case_columns:
        missingness.append({
            "feature": column,
            "cluster": cluster_for(column),
            "n_cases": int(len(cases)),
            "missing_cases": int(cases[column].isna().sum()),
            "unique_values": int(cases[column].nunique(dropna=True)),
        })
    pd.DataFrame(missingness).to_csv(args.output / "case_feature_inventory.csv", index=False)

    static_numeric = cases.select_dtypes(include=[np.number, "bool"])
    pairwise_rows = spearman_rows(static_numeric, "case_static")
    pd.DataFrame(vif_rows(static_numeric)).sort_values("vif", ascending=False).to_csv(
        args.output / "case_numeric_vif.csv", index=False
    )

    with args.full_params.open("rb") as handle:
        params = pickle.load(handle)
    log = to_pm4py_event_log(selected, label="feature-source audit train")
    features = build_df_features(
        log,
        params.net,
        params.initial_marking,
        params.final_marking,
        params.act_to_resources,
        params.net_transition_labels,
        params.resources,
        params.label_data_attributes,
    )
    sync = features.loc[features["resource"].notna()].copy()
    static_native_columns = [
        column for column in case_columns + ["res_workload", "queue_length"]
        if column in sync.columns
    ]
    event_numeric = sync[static_native_columns].select_dtypes(include=[np.number, "bool"])
    pairwise_rows.extend(spearman_rows(event_numeric, "aligned_event_static_native"))

    pairwise = pd.DataFrame(pairwise_rows).sort_values(
        ["level", "abs_spearman_rho"], ascending=[True, False]
    )
    pairwise.to_csv(args.output / "numeric_spearman_pairs.csv", index=False)
    pairwise.loc[pairwise["abs_spearman_rho"] >= 0.70].to_csv(
        args.output / "high_numeric_correlations_abs_rho_ge_070.csv", index=False
    )
    cluster_summary(pairwise).to_csv(
        args.output / "numeric_cluster_association_summary.csv", index=False
    )

    native_summary = []
    for column in ("res_workload", "queue_length"):
        if column in sync.columns:
            native_summary.append({"feature": column, **mean_ci_description(sync[column])})
    pd.DataFrame(native_summary).to_csv(
        args.output / "native_dynamic_feature_summary.csv", index=False
    )

    inventory = pd.DataFrame(inventory_rows(args.model_json))
    if not inventory.empty:
        inventory.to_csv(args.output / "retained_rule_nodes.csv", index=False)
        (
            inventory.groupby(["model", "family", "cluster", "base_feature"], as_index=False)
            .size()
            .rename(columns={"size": "retained_split_nodes"})
            .sort_values(["model", "retained_split_nodes"], ascending=[True, False])
            .to_csv(args.output / "retained_rule_inventory.csv", index=False)
        )
        (
            inventory.groupby(["model", "cluster"], as_index=False)
            .size()
            .rename(columns={"size": "retained_split_nodes"})
            .to_csv(args.output / "retained_rule_cluster_summary.csv", index=False)
        )

    manifest = {
        "schema": "ctb-feature-source-audit-1",
        "train_log": str(args.train.resolve()),
        "full_params": str(args.full_params.resolve()),
        "n_cases": int(len(cases)),
        "n_aligned_sync_events": int(len(sync)),
        "interpretation": (
            "Associations diagnose redundancy; they neither require predictor "
            "independence nor establish causal effects. Factorial hold-out simulation "
            "is the primary source-attribution experiment."
        ),
        "privacy": "Only aggregate associations and rule counts are persisted.",
    }
    with (args.output / "audit_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[feature-audit] Results written to {args.output}")


if __name__ == "__main__":
    main()
