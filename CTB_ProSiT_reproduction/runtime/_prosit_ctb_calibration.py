"""CTB-specific, training-only calibration around ProSiT.

ProSiT remains the simulation framework.  This module makes three modelling
assumptions explicit where the generic defaults do not match the CTB log:

* observed zero inter-arrival times are retained as a point mass instead of
  being smoothed away by a continuous distribution;
* trucks admitted before the weekly gate closure finish non-preemptively;
  their yard service is not paused until the terminal reopens;
* optional routing calibration is retained for legacy sequential-trie runs;
  discovered Inductive-Miner models keep ProSiT's native transition models.

No holdout observation is accepted by the calibration API.  This is important:
the temporal test split remains a genuine out-of-sample validation set.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import pandas as pd

from _eventlog_contract import ACT_COL, GATE_IN, GATE_OUT, TS_COMPLETE, TS_START


CALIBRATION_VERSION = 1
TIMESTAMP_COLUMNS = ("enabled:timestamp", TS_START, TS_COMPLETE)


def _full_week_calendar() -> dict[int, dict[int, bool]]:
    return {weekday: {hour: True for hour in range(24)} for weekday in range(7)}


def _trace_variant(trace) -> tuple[str, ...]:
    activities = tuple(str(event[ACT_COL]) for event in trace)
    if len(activities) < 3 or activities[0] != GATE_IN or activities[-1] != GATE_OUT:
        raise ValueError(f"Invalid CTB trace supplied to calibration: {activities!r}")
    return activities[1:-1]


@dataclass(frozen=True)
class _TrieLayout:
    yard_transition: dict[tuple[tuple[str, ...], str], str]
    gate_out_transition: dict[tuple[str, ...], str]
    outgoing: dict[tuple[str, ...], tuple[str, ...]]


def _trie_layout(log) -> _TrieLayout:
    """Recreate the deterministic transition naming used by the trie builder."""

    variants = {_trace_variant(trace) for trace in log}
    edges: set[tuple[tuple[str, ...], str, tuple[str, ...]]] = set()
    for variant in variants:
        for index, activity in enumerate(variant):
            prefix = variant[:index]
            child = variant[: index + 1]
            edges.add((prefix, activity, child))

    yard_transition: dict[tuple[tuple[str, ...], str], str] = {}
    outgoing: dict[tuple[str, ...], list[str]] = defaultdict(list)
    sorted_edges = sorted(edges, key=lambda value: (len(value[0]), value[0], value[1]))
    for index, (prefix, activity, _child) in enumerate(sorted_edges):
        transition_name = f"yard_{index:03d}"
        yard_transition[(prefix, activity)] = transition_name
        outgoing[prefix].append(transition_name)

    gate_out_transition: dict[tuple[str, ...], str] = {}
    for index, terminal in enumerate(sorted(variants, key=lambda value: (len(value), value))):
        transition_name = f"gate_out_{index:03d}"
        gate_out_transition[terminal] = transition_name
        outgoing[terminal].append(transition_name)

    return _TrieLayout(
        yard_transition=yard_transition,
        gate_out_transition=gate_out_transition,
        outgoing={prefix: tuple(names) for prefix, names in outgoing.items()},
    )


def _raw_trace_attributes(trace, params) -> dict[str, Any]:
    event = trace[0]
    features: dict[str, Any] = {}
    categorical = set(params.label_data_attributes_categorical)
    for attribute in params.label_data_attributes:
        value = event.get(attribute)
        if attribute in categorical:
            for category in params.attribute_values_label_categorical[attribute]:
                features[f"{attribute} = {category}"] = int(value == category)
        else:
            features[attribute] = value
    return features


def _features_at_prefix(trace, prefix: tuple[str, ...], params) -> dict[str, Any]:
    features = _raw_trace_attributes(trace, params)
    history = {label: 0 for label in params.net_transition_labels}
    history[GATE_IN] = 1
    for activity in prefix:
        history[activity] += 1
    features.update(history)
    return features


def _model_weight(model, features: dict[str, Any]) -> float:
    if hasattr(model, "apply"):
        value = model.apply(features)
    elif isinstance(model, (int, float, np.number)):
        value = model
    else:
        value = 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.0
    return value if np.isfinite(value) else 0.0


def _floor_model_weights(model, probability_floor: float):
    if hasattr(model, "rules"):
        for node in model.rules.values():
            if isinstance(node, dict) and "value" in node:
                node["value"] = max(probability_floor, float(node["value"]))
        return model
    if isinstance(model, (int, float, np.number)):
        return max(probability_floor, float(model))
    return probability_floor


def _scale_model_weights(model, factor: float):
    if hasattr(model, "rules"):
        for node in model.rules.values():
            if isinstance(node, dict) and "value" in node:
                node["value"] = float(node["value"]) * factor
        return model
    return float(model) * factor


def _calibrate_routing(
    params,
    train_log,
    *,
    probability_floor: float = 1e-9,
    tolerance: float = 1e-8,
    max_iterations: int = 500,
) -> dict:
    """Calibrate marginal prefix probabilities without erasing data effects."""

    layout = _trie_layout(train_log)
    contexts: dict[tuple[str, ...], list[tuple[Any, str]]] = defaultdict(list)
    counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)

    for trace in train_log:
        variant = _trace_variant(trace)
        for index in range(len(variant) + 1):
            prefix = variant[:index]
            chosen = (
                layout.yard_transition[(prefix, variant[index])]
                if index < len(variant)
                else layout.gate_out_transition[prefix]
            )
            contexts[prefix].append((trace, chosen))
            counts[prefix][chosen] += 1

    prefix_reports = []
    for prefix, transition_names in sorted(
        layout.outgoing.items(), key=lambda item: (len(item[0]), item[0])
    ):
        if len(transition_names) < 2:
            continue

        for name in transition_names:
            params.transition_weights[name] = _floor_model_weights(
                params.transition_weights[name], probability_floor
            )

        rows = contexts[prefix]
        base = np.empty((len(rows), len(transition_names)), dtype=float)
        for row_index, (trace, _chosen) in enumerate(rows):
            features = _features_at_prefix(trace, prefix, params)
            for col_index, name in enumerate(transition_names):
                base[row_index, col_index] = max(
                    probability_floor,
                    _model_weight(params.transition_weights[name], features),
                )

        target = np.asarray(
            [counts[prefix][name] / len(rows) for name in transition_names],
            dtype=float,
        )
        unscaled = base / base.sum(axis=1, keepdims=True)
        before = unscaled.mean(axis=0)

        factors = np.ones(len(transition_names), dtype=float)
        current = before
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            weighted = base * factors
            probabilities = weighted / weighted.sum(axis=1, keepdims=True)
            current = probabilities.mean(axis=0)
            if float(np.max(np.abs(current - target))) <= tolerance:
                break
            factors *= target / np.maximum(current, probability_floor)
            # Only relative factors matter.  Normalising prevents overflow.
            factors /= float(np.exp(np.mean(np.log(np.maximum(factors, probability_floor)))))

        for name, factor in zip(transition_names, factors, strict=True):
            params.transition_weights[name] = _scale_model_weights(
                params.transition_weights[name], float(factor)
            )

        prefix_reports.append(
            {
                "prefix": list(prefix),
                "n_training_decisions": len(rows),
                "iterations": iterations,
                "transition_names": list(transition_names),
                "target_probabilities": target.tolist(),
                "predicted_before": before.tolist(),
                "predicted_after": current.tolist(),
                "multiplicative_factors": factors.tolist(),
                "max_abs_error_after": float(np.max(np.abs(current - target))),
            }
        )

    return {
        "method": "per_prefix_multiplicative_weight_calibration",
        "holdout_used": False,
        "conditional_data_effects_preserved": True,
        "probability_floor": probability_floor,
        "calibrated_branching_prefixes": len(prefix_reports),
        "max_abs_training_marginal_error": max(
            (row["max_abs_error_after"] for row in prefix_reports), default=0.0
        ),
        "prefixes": prefix_reports,
    }


def _calibrate_arrivals(params, train_log) -> dict:
    """Replace continuous-only arrival samples by the robust empirical PMF."""

    from prosit.utils.common_utils import count_working_minutes
    from prosit.utils.distribution_utils import remove_outliers
    from prosit.utils.rule_utils import DecisionRules

    arrival_times = sorted(
        min(pd.Timestamp(event[TS_START]).to_pydatetime() for event in trace)
        for trace in train_log
    )
    working_minutes = [
        count_working_minutes(
            previous,
            current,
            params.arrival_calendar,
        )
        for previous, current in zip(arrival_times, arrival_times[1:])
    ]
    robust_minutes = [float(value) for value in remove_outliers(working_minutes)]
    if not robust_minutes:
        raise ValueError("No inter-arrival observations remain after robust filtering.")

    mean_value = float(np.mean(robust_minutes))
    if params.rules_mode:
        empirical_model = DecisionRules()
        # ``sampled`` is the runtime source used by DecisionRules.  ``dist`` is
        # a portable fallback for ProSiT JSON, while the authoritative pickle
        # and calibration audit preserve the zero-inflated sample exactly.
        empirical_model.rules = {
            0: {
                "value": mean_value,
                "dist": ("fixed", (mean_value,), min(robust_minutes), max(robust_minutes)),
                "sampled": robust_minutes,
            }
        }
        params.arrival_time_distribution = empirical_model
        runtime_representation = "decision_rules_empirical_sample"
    else:
        # ProSiT 1.0.3 expects a five-element distribution tuple whenever
        # ``rules_mode`` is false.  Keep a compact sentinel in that tuple and
        # preserve the empirical sample on the authoritative pickle.  The CTB
        # simulation adapter recognises the sentinel and samples from this
        # array; the stock engine would otherwise try to unpack DecisionRules.
        params.arrival_time_distribution = (
            "ctb_empirical",
            (),
            min(robust_minutes),
            max(robust_minutes),
            mean_value,
        )
        params.ctb_arrival_empirical_sample = robust_minutes
        runtime_representation = "no_rules_tuple_sentinel"

    support = Counter(robust_minutes)
    return {
        "method": "robust_empirical_working_minute_pmf",
        "holdout_used": False,
        "n_raw_inter_arrivals": len(working_minutes),
        "n_robust_inter_arrivals": len(robust_minutes),
        "mean_working_minutes": mean_value,
        "zero_probability": float(support[0.0] / len(robust_minutes)),
        "runtime_representation": runtime_representation,
        "support_counts": {
            str(int(value) if float(value).is_integer() else value): int(count)
            for value, count in sorted(support.items())
        },
    }


def _configure_nonpreemptive_completion(params) -> dict:
    """Let already admitted trucks finish across the weekly gate closure."""

    original_active_slots = {
        str(resource): int(
            sum(bool(active) for hours in calendar.values() for active in hours.values())
        )
        for resource, calendar in params.calendars.items()
    }
    params.calendars = {
        resource: _full_week_calendar() for resource in params.resources
    }
    return {
        "method": "continuous_nonpreemptive_completion_after_admission",
        "holdout_used": False,
        "arrival_calendar_unchanged": True,
        "original_active_slots_per_week": original_active_slots,
        "calibrated_active_slots_per_week": 168,
        "interpretation": (
            "The gate calendar still controls admission. Resources finish work on trucks "
            "already inside instead of pausing an activity until the next opening window."
        ),
    }


def calibrate_ctb_parameters(
    params,
    train_log,
    *,
    calibrate_routing: bool = True,
) -> dict:
    """Apply all CTB corrections to freshly discovered parameters in place."""

    if getattr(params, "ctb_calibration", None):
        raise ValueError("CTB calibration has already been applied to these parameters.")

    routing_report = (
        _calibrate_routing(params, train_log)
        if calibrate_routing
        else {
            "method": "native_prosit_transition_weight_discovery",
            "holdout_used": False,
            "applied": False,
            "reason": (
                "Prefix-marginal calibration is specific to the legacy "
                "sequential trie and is not applied to the Inductive-Miner net."
            ),
        }
    )
    report = {
        "version": CALIBRATION_VERSION,
        "training_traces": int(len(train_log)),
        "holdout_used": False,
        "routing": routing_report,
        "arrivals": _calibrate_arrivals(params, train_log),
        "resource_calendars": _configure_nonpreemptive_completion(params),
        "timestamp_observation_model": {
            "resolution": "minute",
            "method": "floor",
            "reason": "All CTB source timestamps are recorded at minute resolution.",
        },
    }
    # The marker travels with the authoritative pickle and prevents accidental
    # double calibration in later validation scripts.
    params.ctb_calibration = deepcopy(report)
    return report


def quantize_simulated_timestamps(
    simulation: pd.DataFrame,
    *,
    resolution: str | None = "min",
) -> pd.DataFrame:
    """Apply the CTB timestamp observation resolution to simulator output."""

    out = simulation.loc[:, ~simulation.columns.duplicated()].copy()
    if resolution is None:
        return out
    for column in TIMESTAMP_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce").dt.floor(resolution)
    return out


def simulate_ctb(
    params,
    *,
    n_traces: int,
    t_start=None,
    seed: int = 42,
    timestamp_resolution: str | None = "min",
) -> pd.DataFrame:
    """Run ProSiT reproducibly and apply only the CTB observation model."""

    from prosit import SimulatorEngine
    import prosit.simulator as prosit_simulator

    random.seed(seed)
    np.random.seed(seed)
    engine = SimulatorEngine(params)

    # ProSiT 1.0.3 no-rules mode may pass numpy arrays to random.choice().
    original_choice = random.choice
    original_sampling_from_dist = prosit_simulator.sampling_from_dist

    def safe_choice(sequence):
        if isinstance(sequence, np.ndarray):
            return sequence[np.random.randint(len(sequence))]
        return original_choice(sequence)

    def sampling_with_ctb_empirical(
        dist,
        distribution_params,
        min_value,
        max_value,
        mean_value,
        n_sample=1000,
    ):
        if dist == "ctb_empirical":
            empirical = np.asarray(
                getattr(params, "ctb_arrival_empirical_sample", []),
                dtype=float,
            )
            if empirical.size == 0:
                raise ValueError("Missing CTB empirical arrival sample on parameter bundle.")
            return np.random.choice(empirical, size=n_sample, replace=True)
        return original_sampling_from_dist(
            dist,
            distribution_params,
            min_value,
            max_value,
            mean_value,
            n_sample=n_sample,
        )

    random.choice = safe_choice
    prosit_simulator.sampling_from_dist = sampling_with_ctb_empirical
    try:
        if t_start is None:
            simulated = engine.apply(n_traces=n_traces)
        else:
            simulated = engine.apply(n_traces=n_traces, t_start=t_start)
    finally:
        random.choice = original_choice
        prosit_simulator.sampling_from_dist = original_sampling_from_dist

    return quantize_simulated_timestamps(
        simulated,
        resolution=timestamp_resolution,
    )
