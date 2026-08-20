"""Smoke-test sequential ProSiT discovery without input enablement timestamps."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
from prosit import SimulatorEngine, SimulatorParameters

from _eventlog_contract import (
    ACT_COL,
    CASE_COL,
    GATE_IN,
    GATE_OUT,
    ORDER_COL,
    build_sequential_variant_trie,
    canonicalize_case_order,
    eventlog_contract_report,
    select_prosit_dataframe,
    to_pm4py_event_log,
)


def main() -> int:
    # Read enough rows to obtain 100 complete cases rather than cutting the
    # final trace in the middle.
    df = pd.read_csv("data/processed/CTB/s6_train.csv", nrows=5000)
    activities = df.groupby(CASE_COL, sort=False)[ACT_COL].agg(set)
    complete_cases = activities[
        activities.apply(
            lambda values: GATE_IN in values
            and GATE_OUT in values
            and len(values.difference({GATE_IN, GATE_OUT})) > 0
        )
    ].index[:100]
    df = canonicalize_case_order(df[df[CASE_COL].isin(complete_cases)].copy())
    print(f"Columns: {sorted(df.columns.tolist())}")
    print(f"Has enabled:timestamp: {'enabled:timestamp' in df.columns}")
    assert "enabled:timestamp" not in df.columns

    df["start:timestamp"] = pd.to_datetime(df["start:timestamp"], errors="coerce")
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")
    df, dropped = select_prosit_dataframe(df, label="smoke training log")
    print(f"Excluded non-ProSiT attributes: {dropped}")

    log = to_pm4py_event_log(df, label="smoke training log")
    net, im, fm, trie_info = build_sequential_variant_trie(log)
    print(f"pm4py log: {len(log)} traces")
    print(f"Net: {len(net.places)} places, {len(net.transitions)} transitions")
    print(f"Observed sequential yard variants: {trie_info.train_variants}")

    params = SimulatorParameters(net, im, fm)
    params.discover_from_eventlog(
        log,
        max_depth_tree=0,
        attribute_mode="empirical",
        enable_multitasking=True,
        random_state=42,
        verbose=False,
    )
    print("ProSiT discovery completed without enabled:timestamp in the input.")

    engine = SimulatorEngine(params)
    original_choice = random.choice

    def safe_choice(sequence):
        if isinstance(sequence, np.ndarray):
            return sequence[np.random.randint(len(sequence))]
        return original_choice(sequence)

    random.choice = safe_choice
    try:
        sim_log = engine.apply(n_traces=50)
    finally:
        random.choice = original_choice

    sim_log[ORDER_COL] = sim_log.groupby(CASE_COL, sort=False).cumcount()
    contract = eventlog_contract_report(sim_log, _already_ordered=True)
    for key in (
        "gate_only_cases",
        "wrong_case_boundary_cases",
        "within_case_overlap_cases",
        "decreasing_completion_cases",
        "gate_out_before_final_yard_cases",
    ):
        assert contract.get(key, 0) == 0, (
            f"Simulation contract failed: {key}={contract[key]}"
        )

    print(f"Simulated {len(sim_log)} events")
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    # Required on Windows because ProSiT/joblib spawns worker processes.
    raise SystemExit(main())
