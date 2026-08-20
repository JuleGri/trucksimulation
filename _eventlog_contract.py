"""Shared CTB event-log ordering and case-structure contract.

The observed timestamps are measurements and are never rewritten here.  The
explicit case order is a separate control-flow concept: Gate In, one or more
yard activities, Gate Out.  Keeping those two concepts separate prevents a
small timestamp overlap from being interpreted as within-case concurrency by
PM4Py's discovery algorithms.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import numpy as np
import pandas as pd


CASE_COL = "case:concept:name"
ACT_COL = "concept:name"
RES_COL = "org:resource"
TS_START = "start:timestamp"
TS_COMPLETE = "time:timestamp"
ORDER_COL = "case:event:order"

# Explicit ProSiT feature boundary.  ProSiT 1.0.3 treats every non-standard
# XES column as a case attribute and one-hot encodes categorical values.  An
# event-specific identifier/timestamp such as ocr_timestamp would therefore
# create tens of thousands of dummy columns and exhaust memory.
PROSIT_CASE_ATTRIBUTE_ALLOWLIST = (
    "process_flow_type",
    "n_containers",
    "n_stops",
    "n_deliveries",
    "n_receives",
    "has_hazardous",
    "has_reefer",
    "full_ratio",
    "visit_complexity",
    "gate_demand",
    "rmg_demand",
    "vc_demand",
    "mt_demand",
    "gate_utilization",
    "rmg_utilization",
    "vc_utilization",
    "mt_utilization",
    "primary_target_area",
    "target_area",
    "target_utilization",
    "target_demand",
    "target_utilization_bin",
    "target_demand_bin",
    "target_rank",
    "target_rank_group",
)
PROSIT_STANDARD_COLUMNS = (
    CASE_COL,
    ACT_COL,
    RES_COL,
    TS_START,
    TS_COMPLETE,
)

GATE_IN = "Gate In"
GATE_OUT = "Gate Out"
GATE_ACTIVITIES = frozenset({GATE_IN, GATE_OUT})
TECHNICAL_COLUMNS = frozenset({ORDER_COL, "@@index", "@@case_index"})


class EventLogContractError(ValueError):
    """Raised when a log cannot represent a valid CTB truck process."""


def select_prosit_dataframe(
    df: pd.DataFrame, *, label: str = "event log"
) -> tuple[pd.DataFrame, list[str]]:
    """Return an ordered dataframe containing only approved ProSiT inputs."""

    missing = set(PROSIT_STANDARD_COLUMNS).difference(df.columns)
    if missing:
        raise EventLogContractError(
            f"[{label}] Missing required ProSiT columns: {sorted(missing)}"
        )
    ordered = canonicalize_case_order(df)
    allowed = set(PROSIT_STANDARD_COLUMNS) | set(PROSIT_CASE_ATTRIBUTE_ALLOWLIST) | {
        ORDER_COL
    }
    dropped = sorted(str(col) for col in ordered.columns if col not in allowed)
    selected_columns = [
        col
        for col in (
            *PROSIT_STANDARD_COLUMNS,
            *PROSIT_CASE_ATTRIBUTE_ALLOWLIST,
            ORDER_COL,
        )
        if col in ordered.columns
    ]
    selected = ordered.loc[:, selected_columns].copy()

    # Data attributes are sampled once per case by ProSiT; changing values
    # within a trace would be semantically ambiguous and must fail early.
    case_attributes = [
        col for col in PROSIT_CASE_ATTRIBUTE_ALLOWLIST if col in selected.columns
    ]
    if case_attributes:
        varying = selected.groupby(CASE_COL, sort=False)[case_attributes].nunique(
            dropna=False
        )
        bad = varying.gt(1).any(axis=0)
        if bad.any():
            raise EventLogContractError(
                f"[{label}] Attributes vary within a case: {bad.index[bad].tolist()}"
            )
    validate_eventlog_contract(selected, label=label, _already_ordered=True)
    return selected, dropped


def restrict_pm4py_log_for_prosit(log) -> list[str]:
    """Remove non-whitelisted event/trace attributes from an EventLog in place."""

    allowed_event = set(PROSIT_STANDARD_COLUMNS) | set(
        PROSIT_CASE_ATTRIBUTE_ALLOWLIST
    )
    dropped: set[str] = set()
    for trace in log:
        for key in list(trace.attributes.keys()):
            if key != "concept:name":
                dropped.add(str(key))
                del trace.attributes[key]
        case_id = str(trace.attributes.get("concept:name"))
        for event in trace:
            event[CASE_COL] = case_id
            for key in list(event.keys()):
                if key not in allowed_event:
                    dropped.add(str(key))
                    del event[key]
    return sorted(dropped)


def _sample_ids(mask: pd.Series, limit: int = 20) -> list[str]:
    return [str(value) for value in mask.index[mask].tolist()[:limit]]


def canonicalize_case_order(df: pd.DataFrame) -> pd.DataFrame:
    """Return a stable Gate In -> yard -> Gate Out row order per case.

    Existing ``case:event:order`` values are authoritative.  Older snapshots
    without that column are migrated from their existing yard-event row order;
    only the two gate events are moved to their domain-defined boundaries.
    Timestamps are deliberately left untouched.
    """

    missing = {CASE_COL, ACT_COL}.difference(df.columns)
    if missing:
        raise EventLogContractError(
            f"Event log is missing required columns: {sorted(missing)}"
        )
    if df[CASE_COL].isna().any() or df[ACT_COL].isna().any():
        raise EventLogContractError("Case IDs and activity labels must not be missing.")

    out = df.copy()
    out["_contract_input_row"] = np.arange(len(out), dtype=np.int64)
    out["_contract_case_row"] = out.groupby(CASE_COL, sort=False)[
        "_contract_input_row"
    ].transform("min")

    if ORDER_COL in out.columns:
        numeric_order = pd.to_numeric(out[ORDER_COL], errors="coerce")
        if numeric_order.isna().any():
            raise EventLogContractError(f"{ORDER_COL} contains missing/non-numeric values.")
        out[ORDER_COL] = numeric_order.astype(np.int64)
        duplicate_order = out.duplicated([CASE_COL, ORDER_COL], keep=False)
        if duplicate_order.any():
            examples = out.loc[duplicate_order, CASE_COL].astype(str).unique()[:20]
            raise EventLogContractError(
                f"{ORDER_COL} is not unique within a case; examples={examples.tolist()}"
            )
        sort_columns = ["_contract_case_row", ORDER_COL, "_contract_input_row"]
    else:
        rank = np.where(
            out[ACT_COL].eq(GATE_IN),
            0,
            np.where(out[ACT_COL].eq(GATE_OUT), 2, 1),
        )
        out["_contract_activity_rank"] = rank
        sort_columns = [
            "_contract_case_row",
            "_contract_activity_rank",
            "_contract_input_row",
        ]

    out = out.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    out[ORDER_COL] = out.groupby(CASE_COL, sort=False).cumcount().astype(np.int64)
    return out.drop(
        columns=[
            "_contract_input_row",
            "_contract_case_row",
            "_contract_activity_rank",
        ],
        errors="ignore",
    )


def eventlog_contract_report(
    df: pd.DataFrame,
    *,
    overlap_tolerance_min: float = 0.0,
    _already_ordered: bool = False,
) -> dict:
    """Summarize structural and temporal case-level consistency."""

    ordered = df if _already_ordered else canonicalize_case_order(df)
    grouped = ordered.groupby(CASE_COL, sort=False)
    is_gate_in = ordered[ACT_COL].eq(GATE_IN)
    is_gate_out = ordered[ACT_COL].eq(GATE_OUT)
    is_yard = ~ordered[ACT_COL].isin(GATE_ACTIVITIES)
    # Grouping integer masks is much faster than one Python callback per case
    # on the 89k-case CTB log.
    n_gate_in = is_gate_in.astype("int8").groupby(ordered[CASE_COL], sort=False).sum()
    n_gate_out = is_gate_out.astype("int8").groupby(ordered[CASE_COL], sort=False).sum()
    n_yard = is_yard.astype("int8").groupby(ordered[CASE_COL], sort=False).sum()
    first_activity = grouped[ACT_COL].first()
    last_activity = grouped[ACT_COL].last()

    missing_gate_in = n_gate_in.ne(1)
    missing_gate_out = n_gate_out.ne(1)
    no_yard = n_yard.lt(1)
    wrong_boundaries = first_activity.ne(GATE_IN) | last_activity.ne(GATE_OUT)

    report = {
        "events": int(len(ordered)),
        "cases": int(ordered[CASE_COL].nunique()),
        "activities": int(ordered[ACT_COL].nunique()),
        "gate_only_cases": int(no_yard.sum()),
        "invalid_gate_in_count_cases": int(missing_gate_in.sum()),
        "invalid_gate_out_count_cases": int(missing_gate_out.sum()),
        "wrong_case_boundary_cases": int(wrong_boundaries.sum()),
        "gate_only_case_examples": _sample_ids(no_yard),
        "invalid_gate_in_examples": _sample_ids(missing_gate_in),
        "invalid_gate_out_examples": _sample_ids(missing_gate_out),
        "wrong_case_boundary_examples": _sample_ids(wrong_boundaries),
        "yard_events_per_case": {
            "min": int(n_yard.min()) if len(n_yard) else 0,
            "max": int(n_yard.max()) if len(n_yard) else 0,
            "mean": float(n_yard.mean()) if len(n_yard) else float("nan"),
        },
        "activity_counts": {
            str(key): int(value)
            for key, value in Counter(ordered[ACT_COL].astype(str)).items()
        },
    }

    if TS_START in ordered.columns and TS_COMPLETE in ordered.columns:
        starts = pd.to_datetime(ordered[TS_START], errors="coerce")
        completes = pd.to_datetime(ordered[TS_COMPLETE], errors="coerce")
        previous_complete = completes.groupby(ordered[CASE_COL], sort=False).shift(1)
        tolerance = pd.Timedelta(minutes=float(overlap_tolerance_min))
        overlaps = starts < (previous_complete - tolerance)
        decreasing_complete = completes < previous_complete
        negative_service = completes < starts

        yard_complete = completes.where(~ordered[ACT_COL].isin(GATE_ACTIVITIES)).groupby(
            ordered[CASE_COL], sort=False
        ).max()
        gate_out_start = starts.where(ordered[ACT_COL].eq(GATE_OUT)).groupby(
            ordered[CASE_COL], sort=False
        ).min()
        gate_out_before_yard = gate_out_start < (yard_complete - tolerance)

        overlap_cases = ordered.loc[overlaps.fillna(False), CASE_COL].astype(str).unique()
        decrease_cases = ordered.loc[
            decreasing_complete.fillna(False), CASE_COL
        ].astype(str).unique()
        negative_cases = ordered.loc[negative_service.fillna(False), CASE_COL].astype(str).unique()

        report.update(
            {
                "overlap_tolerance_min": float(overlap_tolerance_min),
                "within_case_overlap_events": int(overlaps.fillna(False).sum()),
                "within_case_overlap_cases": int(len(overlap_cases)),
                "within_case_overlap_case_examples": overlap_cases[:20].tolist(),
                "decreasing_completion_events": int(
                    decreasing_complete.fillna(False).sum()
                ),
                "decreasing_completion_cases": int(len(decrease_cases)),
                "decreasing_completion_case_examples": decrease_cases[:20].tolist(),
                "negative_service_events": int(negative_service.fillna(False).sum()),
                "negative_service_cases": int(len(negative_cases)),
                "gate_out_before_final_yard_cases": int(
                    gate_out_before_yard.fillna(False).sum()
                ),
                "gate_out_before_final_yard_examples": _sample_ids(
                    gate_out_before_yard.fillna(False)
                ),
            }
        )

    return report


def validate_eventlog_contract(
    df: pd.DataFrame,
    *,
    label: str = "event log",
    enforce_temporal_sequence: bool = False,
    overlap_tolerance_min: float = 0.0,
    _already_ordered: bool = False,
) -> dict:
    """Validate the CTB case contract and return its audit report."""

    report = eventlog_contract_report(
        df,
        overlap_tolerance_min=overlap_tolerance_min,
        _already_ordered=_already_ordered,
    )
    structural_failures = {
        key: report[key]
        for key in (
            "gate_only_cases",
            "invalid_gate_in_count_cases",
            "invalid_gate_out_count_cases",
            "wrong_case_boundary_cases",
        )
        if report[key]
    }
    temporal_failures = {}
    if enforce_temporal_sequence:
        temporal_failures = {
            key: report.get(key, 0)
            for key in (
                "within_case_overlap_cases",
                "decreasing_completion_cases",
                "negative_service_events",
                "gate_out_before_final_yard_cases",
            )
            if report.get(key, 0)
        }
    failures = structural_failures | temporal_failures
    if failures:
        raise EventLogContractError(f"[{label}] CTB event-log contract failed: {failures}")
    return report


def to_pm4py_event_log(df: pd.DataFrame, *, label: str = "event log"):
    """Build a PM4Py EventLog without timestamp-based reordering.

    ``pm4py.format_dataframe`` sorts by the primary timestamp.  That is useful
    for point-event logs, but wrong here because CTB completion timestamps can
    overlap slightly.  Manual construction preserves the explicit case order
    while retaining the original start/complete timestamps for ProSiT's time
    and resource-concurrency discovery.
    """

    import pm4py

    # Pipeline stages return canonicalized data with an explicit order column.
    # Reuse that frame to avoid another ~50 MB copy of the training log.
    ordered = df if ORDER_COL in df.columns else canonicalize_case_order(df)
    validate_eventlog_contract(ordered, label=label, _already_ordered=True)
    timestamp_columns = [
        col for col in (TS_START, TS_COMPLETE) if col in ordered.columns
    ]
    if any(not pd.api.types.is_datetime64_any_dtype(ordered[col]) for col in timestamp_columns):
        ordered = ordered.copy()
        for col in timestamp_columns:
            ordered[col] = pd.to_datetime(ordered[col], errors="coerce")
    # Direct conversion preserves dataframe row order.  The problematic sort
    # happens in pm4py.format_dataframe(), which is intentionally not called.
    # PM4Py moves case:* columns to trace attributes; restore the case ID on
    # events because ProSiT 1.0.3 reads it from trace[0].
    log = pm4py.convert_to_event_log(ordered, case_id_key=CASE_COL)
    for trace in log:
        case_id = str(trace.attributes.get("concept:name"))
        trace.attributes.pop("event:order", None)
        for key in TECHNICAL_COLUMNS:
            trace.attributes.pop(key, None)
        for event in trace:
            event[CASE_COL] = case_id
            for key in TECHNICAL_COLUMNS:
                if key in event:
                    del event[key]
    return log


def normalize_pm4py_event_log(log, *, label: str = "event log"):
    """Validate an XES-loaded log and remove technical feature columns."""

    # PM4Py 2.7/2026 returns a DataFrame from read_xes() by default, while
    # older releases returned EventLog. Support both APIs explicitly.
    if isinstance(log, pd.DataFrame):
        return to_pm4py_event_log(log, label=label)

    rows = []
    for trace in log:
        case_id = trace.attributes.get("concept:name")
        for order, event in enumerate(trace):
            event[CASE_COL] = str(case_id)
            rows.append(
                {
                    CASE_COL: str(case_id),
                    ACT_COL: event.get(ACT_COL),
                    ORDER_COL: order,
                    TS_START: event.get(TS_START),
                    TS_COMPLETE: event.get(TS_COMPLETE),
                }
            )
            for key in TECHNICAL_COLUMNS:
                if key in event:
                    del event[key]
        for key in TECHNICAL_COLUMNS:
            trace.attributes.pop(key, None)
    validate_eventlog_contract(pd.DataFrame(rows), label=label)
    return log


@dataclass(frozen=True)
class VariantTrieInfo:
    train_variants: int
    places: int
    transitions: int
    yard_transition_edges: int
    terminal_prefixes: int


def _yard_variants(log) -> set[tuple[str, ...]]:
    variants: set[tuple[str, ...]] = set()
    for trace in log:
        activities = tuple(str(event[ACT_COL]) for event in trace)
        if (
            len(activities) < 3
            or activities[0] != GATE_IN
            or activities[-1] != GATE_OUT
            or any(act in GATE_ACTIVITIES for act in activities[1:-1])
        ):
            raise EventLogContractError(
                "Cannot build sequential control flow from an invalid trace: "
                f"case={trace.attributes.get('concept:name')!r}, activities={activities!r}"
            )
        variants.add(activities[1:-1])
    if not variants:
        raise EventLogContractError("Cannot build a control-flow model from an empty log.")
    return variants


def build_sequential_variant_trie(log):
    """Discover a sequential, observed-variant Petri net.

    The result is a state-machine/trie: one token represents one truck case,
    every path contains at least one yard activity, and no path can enable two
    activities in the same case concurrently.  Repeated labels on different
    trie edges are intentional; ProSiT models service/resources by label and
    routing probabilities by the uniquely named Petri-net transition.
    """

    from pm4py.objects.petri_net.obj import Marking, PetriNet
    from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to

    variants = _yard_variants(log)
    prefixes = {()}
    edges: set[tuple[tuple[str, ...], str, tuple[str, ...]]] = set()
    for variant in variants:
        for index, activity in enumerate(variant):
            prefix = variant[:index]
            child = variant[: index + 1]
            prefixes.add(child)
            edges.add((prefix, activity, child))

    net = PetriNet("CTB sequential observed-variant trie")
    source = PetriNet.Place("source")
    final = PetriNet.Place("sink")
    net.places.update({source, final})

    prefix_places = {}
    for index, prefix in enumerate(sorted(prefixes, key=lambda value: (len(value), value))):
        place = PetriNet.Place(f"yard_prefix_{index:03d}")
        prefix_places[prefix] = place
        net.places.add(place)

    gate_in = PetriNet.Transition("gate_in", GATE_IN)
    net.transitions.add(gate_in)
    add_arc_from_to(source, gate_in, net)
    add_arc_from_to(gate_in, prefix_places[()], net)

    sorted_edges = sorted(edges, key=lambda value: (len(value[0]), value[0], value[1]))
    for index, (prefix, activity, child) in enumerate(sorted_edges):
        transition = PetriNet.Transition(f"yard_{index:03d}", activity)
        net.transitions.add(transition)
        add_arc_from_to(prefix_places[prefix], transition, net)
        add_arc_from_to(transition, prefix_places[child], net)

    for index, terminal in enumerate(sorted(variants, key=lambda value: (len(value), value))):
        gate_out = PetriNet.Transition(f"gate_out_{index:03d}", GATE_OUT)
        net.transitions.add(gate_out)
        add_arc_from_to(prefix_places[terminal], gate_out, net)
        add_arc_from_to(gate_out, final, net)

    initial_marking = Marking({source: 1})
    final_marking = Marking({final: 1})
    info = VariantTrieInfo(
        train_variants=len(variants),
        places=len(net.places),
        transitions=len(net.transitions),
        yard_transition_edges=len(sorted_edges),
        terminal_prefixes=len(variants),
    )
    return net, initial_marking, final_marking, info


def variant_coverage(train_log, test_log) -> dict:
    train_variants = _yard_variants(train_log)
    test_sequences = []
    for trace in test_log:
        activities = tuple(str(event[ACT_COL]) for event in trace)
        test_sequences.append(activities[1:-1])
    unseen_cases = [sequence for sequence in test_sequences if sequence not in train_variants]
    return {
        "train_variants": int(len(train_variants)),
        "test_variants": int(len(set(test_sequences))),
        "unseen_test_variants": int(len(set(unseen_cases))),
        "unseen_test_cases": int(len(unseen_cases)),
        "unseen_test_case_share": (
            float(len(unseen_cases) / len(test_sequences)) if test_sequences else 0.0
        ),
    }
