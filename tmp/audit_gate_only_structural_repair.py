from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys
import time

import pm4py
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.utils import petri_utils


REPO = Path(r"C:\Users\Jule\Documents\Master\Masterthesis\trucksimulation")
TEST_XES = REPO / "data" / "processed" / "CTB" / "xes_files" / "s6_test.xes"
PARAMS = (
    REPO
    / "baseline"
    / "discovery_params"
    / "params_20260816_214403_train80"
    / "prosit_discovery_workload_inductive_calibrated"
    / "prosit_params.pkl"
)
sys.path.insert(0, str(REPO))
from _eventlog_contract import normalize_pm4py_event_log, restrict_pm4py_log_for_prosit  # noqa: E402


def repair_gate_only_bypass(net: PetriNet) -> dict:
    gate_in = next(t for t in net.transitions if t.label == "Gate In")
    after_gate_in = next(iter(gate_in.out_arcs)).target
    outgoing = [arc.target for arc in after_gate_in.out_arcs]
    ll_mixed = next(t for t in outgoing if t.label == "LL_mixed")
    initial_skip = next(t for t in outgoing if t.label is None)
    shared_after_optional_ll = next(iter(ll_mixed.out_arcs)).target
    if next(iter(initial_skip.out_arcs)).target != shared_after_optional_ll:
        raise AssertionError("Unexpected Inductive-Miner optional-LL layout.")

    shared_outgoing = [arc.target for arc in shared_after_optional_ll.out_arcs]
    exit_skip = next(
        t
        for t in shared_outgoing
        if any(arc.target for arc in t.out_arcs)
        and any(
            downstream.label == "Gate Out"
            for arc in t.out_arcs
            for downstream_arc in arc.target.out_arcs
            for downstream in [downstream_arc.target]
        )
    )
    enter_loop = next(t for t in shared_outgoing if t is not exit_skip)
    loop_entry_place = next(iter(enter_loop.out_arcs)).target

    skip_only_place = PetriNet.Place("after_gate_in_without_ll_mixed")
    net.places.add(skip_only_place)
    original_arc = next(iter(initial_skip.out_arcs))
    petri_utils.remove_arc(net, original_arc)
    petri_utils.add_arc_from_to(initial_skip, skip_only_place, net)
    forced_loop_entry = PetriNet.Transition("enter_yard_after_initial_skip", None)
    net.transitions.add(forced_loop_entry)
    petri_utils.add_arc_from_to(skip_only_place, forced_loop_entry, net)
    petri_utils.add_arc_from_to(forced_loop_entry, loop_entry_place, net)

    return {
        "repair": "split post-Gate-In state by whether LL_mixed has fired",
        "initial_skip": initial_skip.name,
        "exit_skip": exit_skip.name,
        "original_loop_entry": enter_loop.name,
        "added_place": skip_only_place.name,
        "added_transition": forced_loop_entry.name,
        "language_effect": (
            "Forbids Gate In -> Gate Out while preserving LL_mixed-only cases "
            "and all cases with at least one other yard activity."
        ),
    }


def metrics(log, net, im, fm) -> dict:
    start = time.time()
    fitness = pm4py.fitness_token_based_replay(log, net, im, fm)
    precision = pm4py.precision_token_based_replay(log, net, im, fm)
    return {
        "fitness": float(fitness["log_fitness"]),
        "fit_traces_percent": float(fitness["perc_fit_traces"]),
        "precision": float(precision),
        "generalization": float(pm4py.generalization_tbr(log, net, im, fm)),
        "simplicity": float(pm4py.simplicity_petri_net(net, im, fm)),
        "seconds": time.time() - start,
    }


def main() -> None:
    test = normalize_pm4py_event_log(
        pm4py.read_xes(str(TEST_XES), return_legacy_log_object=True), label="test"
    )
    restrict_pm4py_log_for_prosit(test)
    with PARAMS.open("rb") as handle:
        params = pickle.load(handle)
    net, im, fm = params.net, params.initial_marking, params.final_marking
    before = {"test": metrics(test, net, im, fm)}
    repair = repair_gate_only_bypass(net)
    after = {"test": metrics(test, net, im, fm)}
    result = {
        "before": before,
        "after": after,
        "repair": repair,
        "places_after": len(net.places),
        "transitions_after": len(net.transitions),
        "arcs_after": len(net.arcs),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
