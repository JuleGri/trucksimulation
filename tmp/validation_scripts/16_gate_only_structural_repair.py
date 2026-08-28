"""Create and audit a precision-oriented repair of the CTB yard bypass.

The repaired Petri net is a separate robustness model.  The script does not
silently overwrite the automatically discovered baseline.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import pickle
import sys
import time

import pm4py
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.utils import petri_utils


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from _eventlog_contract import (  # noqa: E402
    normalize_pm4py_event_log,
    restrict_pm4py_log_for_prosit,
)


DEFAULT_PARAMS = (
    REPO
    / "baseline/discovery_params/params_20260816_214403_train80"
    / "prosit_discovery_workload_inductive_calibrated/prosit_params.pkl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument(
        "--train-xes",
        type=Path,
        default=REPO / "data/processed/CTB/xes_files/s6_train.xes",
    )
    parser.add_argument(
        "--test-xes",
        type=Path,
        default=REPO / "data/processed/CTB/xes_files/s6_test.xes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "validation/results/gate_only_structural_repair",
    )
    return parser.parse_args()


def load_log(path: Path, label: str):
    log = normalize_pm4py_event_log(
        pm4py.read_xes(str(path), return_legacy_log_object=True), label=label
    )
    restrict_pm4py_log_for_prosit(log)
    return log


def repair_gate_only_bypass(net: PetriNet) -> dict[str, str]:
    gate_in = next(transition for transition in net.transitions if transition.label == "Gate In")
    after_gate_in = next(iter(gate_in.out_arcs)).target
    outgoing = [arc.target for arc in after_gate_in.out_arcs]
    ll_mixed = next(transition for transition in outgoing if transition.label == "LL_mixed")
    initial_skip = next(transition for transition in outgoing if transition.label is None)
    shared_after_optional_ll = next(iter(ll_mixed.out_arcs)).target
    if next(iter(initial_skip.out_arcs)).target != shared_after_optional_ll:
        raise AssertionError("Unexpected Inductive-Miner optional-LL layout.")

    shared_outgoing = [arc.target for arc in shared_after_optional_ll.out_arcs]
    exit_skip = next(
        transition
        for transition in shared_outgoing
        if any(
            downstream.label == "Gate Out"
            for arc in transition.out_arcs
            for downstream_arc in arc.target.out_arcs
            for downstream in [downstream_arc.target]
        )
    )
    enter_loop = next(transition for transition in shared_outgoing if transition is not exit_skip)
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
            "forbids Gate In -> Gate Out while preserving LL_mixed-only cases "
            "and all cases with at least one other yard activity"
        ),
    }


def metrics(log, net, initial_marking, final_marking) -> dict[str, float]:
    start = time.time()
    fitness = pm4py.fitness_token_based_replay(
        log, net, initial_marking, final_marking
    )
    precision = pm4py.precision_token_based_replay(
        log, net, initial_marking, final_marking
    )
    return {
        "fitness": float(fitness["log_fitness"]),
        "fit_traces_percent": float(fitness["perc_fit_traces"]),
        "precision": float(precision),
        "generalization": float(
            pm4py.generalization_tbr(log, net, initial_marking, final_marking)
        ),
        "simplicity": float(
            pm4py.simplicity_petri_net(net, initial_marking, final_marking)
        ),
        "seconds": time.time() - start,
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train = load_log(args.train_xes, "train")
    test = load_log(args.test_xes, "test")
    with args.params.open("rb") as handle:
        source_params = pickle.load(handle)
    repaired_params = copy.deepcopy(source_params)
    net = repaired_params.net
    initial_marking = repaired_params.initial_marking
    final_marking = repaired_params.final_marking

    before = {
        "train": metrics(train, source_params.net, source_params.initial_marking, source_params.final_marking),
        "test": metrics(test, source_params.net, source_params.initial_marking, source_params.final_marking),
    }
    repair = repair_gate_only_bypass(net)
    # The added silent transition is the sole enabled successor of the new
    # place.  Give it an explicit positive weight so the exported parameter
    # bundle remains executable without relying on an engine-specific
    # all-zero-weight fallback.
    repaired_params.transition_weights[repair["added_transition"]] = 1.0
    after = {
        "train": metrics(train, net, initial_marking, final_marking),
        "test": metrics(test, net, initial_marking, final_marking),
    }

    pnml_path = args.output / "ctb_inductive_gate_only_restricted.pnml"
    pickle_path = args.output / "prosit_params_gate_only_restricted.pkl"
    pm4py.write_pnml(net, initial_marking, final_marking, str(pnml_path))
    with pickle_path.open("wb") as handle:
        pickle.dump(repaired_params, handle)

    result = {
        "design": "separate precision-oriented structural robustness model",
        "source_params": str(args.params),
        "before": before,
        "after": after,
        "repair": repair,
        "added_transition_weight": 1.0,
        "structure_after": {
            "places": len(net.places),
            "transitions": len(net.transitions),
            "arcs": len(net.arcs),
        },
        "outputs": {"pnml": str(pnml_path), "pickle": str(pickle_path)},
        "warning": (
            "The repaired model is not the unchanged discovery result. Its simulation "
            "and scenarios must be revalidated before it replaces any primary baseline."
        ),
    }
    with (args.output / "structural_repair_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
