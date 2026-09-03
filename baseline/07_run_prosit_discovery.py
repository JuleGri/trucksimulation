"""
07_run_prosit_discovery.py

Purpose:
- run the actual ProSiT (Vinci et al., 2026) simulation-parameter discovery
  on the held-out training log s6_train.csv;
- discover an Inductive-Miner process tree from the order-preserving training
  log and use its equivalent Petri net as the simulation control flow;
- reject the discovered model if it contains a parallel operator, because one
  physical truck cannot execute two yard activities concurrently;
- report conformance (fitness, precision, generalisation, simplicity) on
  BOTH the training log and the held-out testing log s6_test.csv, which
  addresses reviewer Tier 1 item #1 (held-out validation) directly;
- generate publication-quality Graphviz visualisations of every discovered
  artefact, including a frequency-decorated ("coloured") Petri net and a
  performance-decorated Petri net;
- discover ProSiT SimulatorParameters, persist them as JSON (Prosit >= 1.0.2
  API) and as a pickle for the legacy what_if_* scripts;
- simulate cases with the fitted parameters and write a CSV ready to be
  consumed by validation/02_validate_simulation.py.

Inputs:
- data/processed/CTB/s6_train.csv (default; override with --input);
- data/processed/CTB/s6_test.csv  (default; override with --test).

Outputs (under baseline/discovery_params/<paramset>_train80/prosit_discovery/):
- prosit_params.json                   ProSiT SimulatorParameters (JSON)
- prosit_params.pkl                    Same, pickled for the what_if scripts
- prosit_conformance.csv               fitness / precision / etc. on train and test
- prosit_variants_top20.csv            variant coverage (top 20)
- prosit_run_summary.json              run metadata (timing, hyperparameters, paths)
- sim_baseline_train80.csv             simulated event log at the test-window start
- figures/
    petri_net.png                      discovered Inductive-Miner Petri net
    petri_net_frequency.png            transitions coloured by token replay frequency
    petri_net_performance.png          transitions coloured by mean sojourn time
    dfg_frequency.png                  directly-follows graph (frequency)
    dfg_performance.png                directly-follows graph (mean waiting time)
    process_tree_inductive_diagnostic.png  process-tree view of the simulated net
    bpmn.png                           BPMN translation of the Petri net
    variants_top20.png                 variant coverage bar chart

Methodological note:
- PM4Py receives explicit trace order instead of completion-time order.  The
  authoritative Petri net is converted from the resulting Inductive-Miner
  process tree.  A hard structural guard rejects any model containing an
  ``and`` operator; sequential generalisation across yard activities is
  accepted as intended process-discovery behaviour;
- ProSiT resource multitasking remains enabled across different truck cases.
  The discovered Petri net prevents concurrency only within one case;
- a training-only CTB calibration retains zero inter-arrival mass and lets
  trucks that entered before the weekly closure finish without a multi-day
  interruption.  Routing remains the direct ProSiT discovery result because
  the former prefix calibration was specific to the removed trie;
- fitness / precision are computed with token-based replay for tractability
  on the full 71k-case training log; alignment-based fitness is optionally
  computed on a 1000-case sample (--alignments-sample);
- ProSiT hyperparameters are set to the README's "Standard" configuration
  (max_depth_tree=3, min_samples_leaf_cv=[50, 100, 200], random_state=42)
  so results are directly comparable with published Prosit case studies;
- the simulation start timestamp t_start is read from
  validation/results/split_manifest.json (cutoff_arrival_ts), so the
  simulation and the held-out test log cover the same operational window.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Ensure the shared helper is importable regardless of cwd.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from _eventlog_source import (  # noqa: E402
    add_input_arg,
    is_train_log,
    paramset_suffix_for,
    resolve_event_csv,
    resolve_paramset_dir,
)


CASE_COL = "case:concept:name"
ACT_COL = "concept:name"
RES_COL = "org:resource"
TS_ENABLED = "enabled:timestamp"
TS_START = "start:timestamp"
TS_COMPLETE = "time:timestamp"

_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
from _eventlog_contract import (  # noqa: E402
    PROSIT_CASE_ATTRIBUTE_ALLOWLIST,
    normalize_pm4py_event_log,
    restrict_pm4py_log_for_prosit,
    select_prosit_dataframe,
    to_pm4py_event_log,
    variant_coverage,
)
from _prosit_ctb_calibration import (  # noqa: E402
    calibrate_ctb_parameters,
    simulate_ctb,
)

DEFAULT_TEST = _REPO_ROOT / "data" / "processed" / "CTB" / "s6_test.csv"
DEFAULT_MANIFEST = _REPO_ROOT / "validation" / "results" / "split_manifest.json"
DEFAULT_XES_DIR = _REPO_ROOT / "data" / "processed" / "CTB" / "xes_files"

# These event-log fields encode contemporaneous terminal load directly or are
# deterministic derivatives of it.  They are removed from the rules-only
# ablation so that the middle configuration differs from the workload-aware
# endpoint by design, rather than merely because a fitted tree happened not to
# retain a workload split.
WORKLOAD_PROXY_ATTRIBUTES = frozenset(
    {
        "gate_demand",
        "rmg_demand",
        "vc_demand",
        "mt_demand",
        "gate_utilization",
        "rmg_utilization",
        "vc_utilization",
        "mt_utilization",
        "target_utilization",
        "target_demand",
        "target_utilization_bin",
        "target_demand_bin",
        "target_rank",
        "target_rank_group",
    }
)
WORKLOAD_BLIND_CASE_ATTRIBUTE_ALLOWLIST = tuple(
    attribute
    for attribute in PROSIT_CASE_ATTRIBUTE_ALLOWLIST
    if attribute not in WORKLOAD_PROXY_ATTRIBUTES
)

# Sensitivity configuration with one semantically coherent terminal-state
# representation.  The three yard-area demand and utilisation measurements
# remain available; aggregate, target-derived, binned, and rank variants are
# removed to avoid presenting several transformations of the same state to the
# discovery trees.
CLEAN_STATIC_REMOVED_ATTRIBUTES = frozenset(
    {
        "gate_demand",
        "gate_utilization",
        "target_demand",
        "target_utilization",
        "target_demand_bin",
        "target_utilization_bin",
        "target_rank",
        "target_rank_group",
    }
)
CLEAN_STATIC_CASE_ATTRIBUTE_ALLOWLIST = tuple(
    attribute
    for attribute in PROSIT_CASE_ATTRIBUTE_ALLOWLIST
    if attribute not in CLEAN_STATIC_REMOVED_ATTRIBUTES
)


def configured_case_attribute_allowlist(args) -> tuple[str, ...]:
    if args.workload_blind_attributes:
        return WORKLOAD_BLIND_CASE_ATTRIBUTE_ALLOWLIST
    if args.clean_static_attributes:
        return CLEAN_STATIC_CASE_ATTRIBUTE_ALLOWLIST
    return PROSIT_CASE_ATTRIBUTE_ALLOWLIST


def configured_removed_attributes(args) -> frozenset[str]:
    if args.workload_blind_attributes:
        return WORKLOAD_PROXY_ATTRIBUTES
    if args.clean_static_attributes:
        return CLEAN_STATIC_REMOVED_ATTRIBUTES
    return frozenset()


def process_tree_parallel_count(tree) -> int:
    """Return the number of explicit parallel operators in a process tree."""

    from pm4py.objects.process_tree.obj import Operator

    count = 0
    stack = [tree]
    while stack:
        node = stack.pop()
        if node.operator == Operator.PARALLEL:
            count += 1
        stack.extend(node.children)
    return count


def build_ctb_sequential_contract_tree(train_log):
    """Build the expert-validated CTB case language over observed yard labels.

    The Inductive Miner remains the diagnostic discovery method. This tree is
    the pre-specified source-model repair used when the raw tree admits
    within-truck parallelism or a Gate-only bypass. It permits one or more
    yard activities, in any sequential order and with arbitrary repetition,
    between the mandatory Gate In and Gate Out events.
    """

    from pm4py.objects.process_tree.obj import Operator, ProcessTree

    yard_labels = sorted(
        {
            event[ACT_COL]
            for trace in train_log
            for event in trace
            if event.get(ACT_COL) not in {None, "Gate In", "Gate Out"}
        }
    )
    if not yard_labels:
        raise RuntimeError("No yard activities observed in the training log.")

    def leaf(label):
        return ProcessTree(label=label)

    yard_choice = ProcessTree(
        operator=Operator.XOR,
        children=[leaf(label) for label in yard_labels],
    )
    for child in yard_choice.children:
        child.parent = yard_choice

    # LOOP(do, redo=tau) executes ``do`` at least once and may repeat it.
    repeat_yard = ProcessTree(
        operator=Operator.LOOP,
        children=[yard_choice, ProcessTree(label=None)],
    )
    for child in repeat_yard.children:
        child.parent = repeat_yard

    root = ProcessTree(
        operator=Operator.SEQUENCE,
        children=[leaf("Gate In"), repeat_yard, leaf("Gate Out")],
    )
    for child in root.children:
        child.parent = root
    return root, yard_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProSiT discovery + visualisation on the training split.")
    add_input_arg(parser)
    parser.add_argument("--xes", type=Path, default=None,
                        help="Path to a pre-built XES file for the training log. "
                             "If provided, the CSV --input is ignored and the XES is "
                             "loaded directly with pm4py.read_xes() following Vinci et al.")
    parser.add_argument("--test-xes", type=Path, default=None,
                        help="Path to a pre-built XES file for the test log.")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST,
                        help="Held-out test CSV used for test-set conformance.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="split_manifest.json produced by validation/01_train_test_split.py.")
    parser.add_argument("--max-depth-tree", type=int, default=3,
                        help="ProSiT DecisionTree max_depth (README default: 3).")
    parser.add_argument(
        "--control-flow-policy",
        choices=("reject_parallel", "expert_sequential_contract"),
        default="reject_parallel",
        help=(
            "How to handle the discovered control flow. 'reject_parallel' keeps "
            "the historical hard guard. 'expert_sequential_contract' records the "
            "raw Inductive-Miner tree but discovers ProSiT parameters on the "
            "expert-validated CTB language: Gate In, one or more sequential yard "
            "activities in arbitrary order with repetitions, Gate Out."
        ),
    )
    parser.add_argument("--min-samples-leaf-cv", type=int, nargs="+", default=[50, 100, 200],
                        help="ProSiT min_samples_leaf CV grid (README default: 50 100 200).")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--enable-multitasking", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable ProSiT resource multitasking detection. When True, ProSiT "
                             "computes max_concurrency per resource from observed overlap in the "
                             "training log (threshold: 5%% of events). Essential for CTB where "
                             "each RMG block has multiple cranes. Default: True.")
    parser.add_argument("--multitasking-thr", type=float, default=0.05,
                        help="Fraction of events that must show concurrent work for a resource "
                             "to be classified as multitasking (ProSiT default: 0.05).")
    parser.add_argument("--attribute-mode", choices=("empirical", "distribution"), default="empirical",
                        help="How ProSiT models case-level data attributes. 'empirical' samples "
                             "observed tuples (preserves correlations, safe for categorical); "
                             "'distribution' fits one distribution per attribute (crashes on "
                             "categorical columns in prosit 1.0.3).")
    parser.add_argument("--n-sim-traces", type=int, default=None,
                        help="Number of cases to simulate. Defaults to the size of the test set.")
    parser.add_argument("--alignments-sample", type=int, default=0,
                        help="If >0, compute alignment-based fitness on N sampled traces.")
    parser.add_argument("--skip-simulation", action="store_true",
                        help="Discover parameters but do not simulate.")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip Graphviz visualisations (useful when running headless).")
    parser.add_argument("--write-xes", action="store_true",
                        help="Write XES copies of the train and test logs to data/processed/CTB/xes_files/.")
    parser.add_argument("--use-workload-features", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable ProSiT workload/queue-length features on the waiting-time and "
                             "resource-selection models (README section 'What the Models Learn'). "
                             "Use --no-use-workload-features for the distribution-only comparator.")
    parser.add_argument(
        "--workload-blind-attributes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Remove demand, utilisation, and their derived rank/bin attributes from "
            "every event before discovery. Combine with "
            "--no-use-workload-features for strict visit-only rules, or with "
            "--use-workload-features to isolate ProSiT's native dynamic state."
        ),
    )
    parser.add_argument(
        "--clean-static-attributes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep visit/process, location, and the continuous RMG/VC/MT demand "
            "and utilisation attributes, while removing aggregate, target-derived, "
            "binned, and rank workload proxies. This is the pre-specified cleaned "
            "static-state sensitivity configuration."
        ),
    )
    parser.add_argument("--out-suffix", type=str, default=None,
                        help="Suffix appended to the prosit_discovery/ folder and sim CSV filename so a "
                             "new run does not overwrite a previous one. Defaults to "
                             "a calibrated or uncalibrated Inductive-model suffix derived from the "
                             "selected workload/calibration flags.")
    parser.add_argument(
        "--ctb-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply the training-only CTB fidelity calibration: empirical zero-inflated "
            "arrivals, native ProSiT routing, non-preemptive completion "
            "across the weekly closure, and minute-resolution output. Default: True."
        ),
    )
    parser.add_argument(
        "--timestamp-resolution",
        choices=("minute", "native"),
        default="minute",
        help=(
            "Observation resolution of simulated timestamps. CTB source timestamps are "
            "minute-granular, so 'minute' is the calibrated default."
        ),
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Event log loading
# ----------------------------------------------------------------------

def load_log(csv_path: str, label: str) -> "pd.DataFrame":
    print(f"[{label}] Reading {csv_path}")
    df = pd.read_csv(csv_path)
    # Strip pm4py technical columns that must not be model features
    for col in ("@@index", "@@case_index"):
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"[{label}] Dropped technical column {col}")
    for col in (TS_ENABLED, TS_START, TS_COMPLETE):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # Drop events with missing primary timestamp — pm4py requires it.
    if TS_COMPLETE not in df.columns:
        raise KeyError(f"[{label}] Column {TS_COMPLETE!r} is required.")
    n_before = len(df)
    df = df.dropna(subset=[TS_COMPLETE]).copy()
    dropped = n_before - len(df)
    if dropped:
        print(f"[{label}] Dropped {dropped:,} events with missing {TS_COMPLETE}")
    df, dropped_features = select_prosit_dataframe(df, label=label)
    if dropped_features:
        print(
            f"[{label}] Excluded non-ProSiT attributes: "
            + ", ".join(dropped_features)
        )
    contract = {
        "gate_only_cases": 0,
        "yard_events_per_case": {
            "min": int(
                (~df[ACT_COL].isin(["Gate In", "Gate Out"]))
                .groupby(df[CASE_COL], sort=False)
                .sum()
                .min()
            )
        },
    }
    print(
        f"[{label}] events={len(df):,}  cases={df[CASE_COL].nunique():,}  "
        f"gate-only={contract['gate_only_cases']}  "
        f"min-yard/case={contract['yard_events_per_case']['min']}"
    )
    return df


def to_pm4py_log(df: pd.DataFrame, label: str):
    # PM4Py's dataframe formatter sorts on the completion timestamp.  CTB has
    # a few legitimate timestamp overlaps, so that sort can move Gate Out
    # ahead of a yard event and make the Inductive Miner infer concurrency.
    # The shared builder keeps logical case order and original timestamps.
    log = to_pm4py_event_log(df, label=label)
    print(f"[{label}] pm4py EventLog with {len(log)} traces")
    return log


def maybe_write_xes(log, out_path: Path, label: str) -> None:
    import pm4py
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{label}] Writing XES -> {out_path}")
    # pm4py.format_dataframe adds technical ordering attributes.  Keep them
    # available to the in-memory log, but do not persist them to XES where
    # downstream discovery could mistake them for data attributes.
    removed_attributes = []
    for trace in log:
        for key in ("@@index", "@@case_index"):
            if key in trace.attributes:
                removed_attributes.append((trace.attributes, key, trace.attributes.pop(key)))
        for event in trace:
            for key in ("@@index", "@@case_index"):
                if key in event:
                    value = event[key]
                    del event[key]
                    removed_attributes.append((event, key, value))
    try:
        pm4py.write_xes(log, str(out_path))
    finally:
        for attributes, key, value in removed_attributes:
            attributes[key] = value


# ----------------------------------------------------------------------
# Petri net conformance
# ----------------------------------------------------------------------

def token_based_conformance(log, net, im, fm, label: str) -> dict[str, float]:
    import pm4py
    print(f"[{label}] Token-based replay ...")
    t0 = time.time()
    fitness = pm4py.fitness_token_based_replay(log, net, im, fm)
    precision = pm4py.precision_token_based_replay(log, net, im, fm)
    dt = time.time() - t0
    print(f"[{label}]   fitness={fitness['log_fitness']:.4f}  precision={precision:.4f}  ({dt:.1f}s)")
    return {
        "log": label,
        "fitness_log": float(fitness.get("log_fitness", float("nan"))),
        "fitness_perc_fit_traces": float(fitness.get("perc_fit_traces", float("nan"))),
        "fitness_average_trace": float(fitness.get("average_trace_fitness", float("nan"))),
        "precision_token": float(precision),
        "compute_seconds": float(dt),
    }


def evaluation_metrics(log, net, im, fm, label: str) -> dict[str, float]:
    import pm4py
    metrics: dict[str, float] = {}
    # Both metrics take (log, net, im, fm) in pm4py 2.7.x. simplicity_petri_net
    # additionally accepts (net, im, fm) — we pass all four for compatibility.
    for name in ("generalization_tbr", "simplicity_petri_net"):
        func = getattr(pm4py, name, None)
        if func is None:
            metrics[name.split("_")[0]] = float("nan")
            continue
        try:
            if name == "generalization_tbr":
                metrics["generalization"] = float(func(log, net, im, fm))
            else:
                metrics["simplicity"] = float(func(net, im, fm))
        except TypeError:
            try:
                metrics["simplicity"] = float(func(net))
            except Exception as exc:  # pragma: no cover
                print(f"[{label}] simplicity failed: {exc}")
                metrics["simplicity"] = float("nan")
        except Exception as exc:
            print(f"[{label}] {name} failed: {exc}")
            metrics[name.split("_")[0]] = float("nan")
    print(f"[{label}]   generalization={metrics.get('generalization', float('nan')):.4f}  "
          f"simplicity={metrics.get('simplicity', float('nan')):.4f}")
    return metrics


def alignment_conformance(log, net, im, fm, sample_size: int, seed: int, label: str) -> dict[str, float]:
    import pm4py
    if sample_size <= 0 or len(log) == 0:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(log), size=min(sample_size, len(log)), replace=False)
    sampled = [log[i] for i in sorted(idx)]
    from pm4py.objects.log.obj import EventLog
    sampled_log = EventLog(sampled)
    print(f"[{label}] Alignment-based conformance on {len(sampled_log)} traces (subsample) ...")
    t0 = time.time()
    try:
        fitness = pm4py.fitness_alignments(sampled_log, net, im, fm)
        precision = pm4py.precision_alignments(sampled_log, net, im, fm)
    except Exception as exc:
        print(f"[{label}] Alignment computation failed: {exc}")
        return {}
    dt = time.time() - t0
    print(f"[{label}]   align_fitness={fitness.get('log_fitness', float('nan')):.4f}  "
          f"align_precision={float(precision):.4f}  ({dt:.1f}s)")
    return {
        "fitness_alignments": float(fitness.get("log_fitness", float("nan"))),
        "precision_alignments": float(precision),
        "align_sample_size": int(len(sampled_log)),
        "align_compute_seconds": float(dt),
    }


# ----------------------------------------------------------------------
# Visualisations
# ----------------------------------------------------------------------

def _save_gviz(gviz, out_path: Path, label: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from pm4py.visualization.common import save
    save.save(gviz, str(out_path))
    print(f"[figures] Wrote {out_path}")


def render_visualisations(train_log, net, im, fm, diagnostic_tree, fig_dir: Path) -> None:
    import pm4py
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Vanilla Petri net (structure only).
    try:
        from pm4py.visualization.petri_net import visualizer as pn_vis
        gviz = pn_vis.apply(net, im, fm)
        _save_gviz(gviz, fig_dir / "petri_net.png", "petri_net")
    except Exception as exc:
        print(f"[figures] petri_net failed: {exc}")

    # 2. Frequency-decorated ("coloured") Petri net.
    try:
        from pm4py.visualization.petri_net import visualizer as pn_vis
        gviz = pn_vis.apply(
            net, im, fm, log=train_log,
            variant=pn_vis.Variants.FREQUENCY,
            parameters={pn_vis.Variants.FREQUENCY.value.Parameters.FORMAT: "png"},
        )
        _save_gviz(gviz, fig_dir / "petri_net_frequency.png", "petri_net_frequency")
    except Exception as exc:
        print(f"[figures] petri_net_frequency failed: {exc}")

    # 3. Performance-decorated Petri net.
    try:
        from pm4py.visualization.petri_net import visualizer as pn_vis
        gviz = pn_vis.apply(
            net, im, fm, log=train_log,
            variant=pn_vis.Variants.PERFORMANCE,
            parameters={pn_vis.Variants.PERFORMANCE.value.Parameters.FORMAT: "png"},
        )
        _save_gviz(gviz, fig_dir / "petri_net_performance.png", "petri_net_performance")
    except Exception as exc:
        print(f"[figures] petri_net_performance failed: {exc}")

    # 4. Directly-follows graphs (frequency + performance).
    try:
        dfg_freq, dfg_freq_start, dfg_freq_end = pm4py.discover_directly_follows_graph(train_log)
        gviz = pm4py.visualization.dfg.visualizer.apply(
            dfg_freq, log=train_log,
            variant=pm4py.visualization.dfg.visualizer.Variants.FREQUENCY,
        )
        _save_gviz(gviz, fig_dir / "dfg_frequency.png", "dfg_frequency")
    except Exception as exc:
        print(f"[figures] dfg_frequency failed: {exc}")

    try:
        dfg_perf, sa, ea = pm4py.discover_performance_dfg(train_log)
        gviz = pm4py.visualization.dfg.visualizer.apply(
            dfg_perf, log=train_log,
            variant=pm4py.visualization.dfg.visualizer.Variants.PERFORMANCE,
        )
        _save_gviz(gviz, fig_dir / "dfg_performance.png", "dfg_performance")
    except Exception as exc:
        print(f"[figures] dfg_performance failed: {exc}")

    # 5. Process-tree view of the authoritative Inductive-Miner model.
    try:
        from pm4py.visualization.process_tree import visualizer as tree_vis
        gviz = tree_vis.apply(diagnostic_tree)
        _save_gviz(
            gviz,
            fig_dir / "process_tree_inductive_diagnostic.png",
            "process_tree_inductive_diagnostic",
        )
    except Exception as exc:
        print(f"[figures] diagnostic process_tree failed: {exc}")

    # 6. BPMN translation of the authoritative constrained Petri net.
    try:
        bpmn = pm4py.convert_to_bpmn(net, im, fm)
        from pm4py.visualization.bpmn import visualizer as bpmn_vis
        gviz = bpmn_vis.apply(bpmn)
        _save_gviz(gviz, fig_dir / "bpmn.png", "bpmn")
    except Exception as exc:
        print(f"[figures] bpmn failed: {exc}")

    # 7. Variants top-20 bar chart.
    try:
        variants = pm4py.get_variants_as_tuples(train_log)
        rows = [
            {"variant": " -> ".join(v), "n_cases": len(cases), "length": len(v)}
            for v, cases in variants.items()
        ]
        v_df = pd.DataFrame(rows).sort_values("n_cases", ascending=False).reset_index(drop=True)
        v_df.head(20).to_csv(fig_dir.parent / "prosit_variants_top20.csv", index=False)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = v_df.head(20)
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
        ax.barh(range(len(top))[::-1], top["n_cases"], color="#2c7fb8")
        labels = [v if len(v) < 60 else v[:57] + "..." for v in top["variant"]]
        ax.set_yticks(range(len(top))[::-1])
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("cases")
        total = int(v_df["n_cases"].sum())
        covered = int(top["n_cases"].sum())
        ax.set_title(f"Top 20 variants — cover {covered:,} / {total:,} cases ({covered/total:.1%})")
        fig.tight_layout()
        fig.savefig(fig_dir / "variants_top20.png", dpi=160)
        plt.close(fig)
        print(f"[figures] Wrote {fig_dir / 'variants_top20.png'}")
    except Exception as exc:
        print(f"[figures] variants_top20 failed: {exc}")


# ----------------------------------------------------------------------
# ProSiT parameter discovery + simulation
# ----------------------------------------------------------------------

def discover_prosit_params(train_log, net, im, fm, args) -> Any:
    from prosit import SimulatorParameters
    params = SimulatorParameters(net, im, fm)
    print(f"[prosit] discover_from_eventlog(max_depth_tree={args.max_depth_tree}, "
          f"min_samples_leaf_cv={args.min_samples_leaf_cv}, "
          f"attribute_mode={args.attribute_mode!r}, "
          f"use_workload_features={args.use_workload_features}, "
          f"enable_multitasking={args.enable_multitasking}, "
          f"multitasking_thr={args.multitasking_thr}, "
          f"random_state={args.random_state}) ...")
    print(
        "[prosit] Approved case attributes: "
        + ", ".join(
            configured_case_attribute_allowlist(args)
        )
    )
    t0 = time.time()
    params.discover_from_eventlog(
        train_log,
        max_depth_tree=args.max_depth_tree,
        min_samples_leaf_cv=args.min_samples_leaf_cv,
        attribute_mode=args.attribute_mode,
        use_workload_features=args.use_workload_features,
        enable_multitasking=args.enable_multitasking,
        multitasking_thr=args.multitasking_thr,
        random_state=args.random_state,
        verbose=True,
    )
    dt = time.time() - t0
    print(f"[prosit] Discovery finished in {dt:.1f}s")
    return params, dt


def simulate_prosit(
    params,
    n_traces: int,
    t_start: datetime | None,
    seed: int,
    timestamp_resolution: str,
) -> pd.DataFrame:
    print(f"[prosit] Simulating n_traces={n_traces:,}  t_start={t_start}  ...")
    t0 = time.time()
    sim_log = simulate_ctb(
        params,
        n_traces=n_traces,
        t_start=t_start,
        seed=seed,
        timestamp_resolution=(
            "min" if timestamp_resolution == "minute" else None
        ),
    )
    dt = time.time() - t0
    print(f"[prosit] Simulation finished in {dt:.1f}s (events={len(sim_log):,})")
    return sim_log


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _read_manifest(path: Path) -> dict:
    if not path.exists():
        print(f"[main] Split manifest not found at {path}; simulation t_start will be None.")
        return {}
    with open(path, "r") as fh:
        return json.load(fh)


def _drop_event_attributes(log, attributes: frozenset[str], label: str) -> dict[str, int]:
    """Remove forbidden attributes from every event and report their counts."""

    removed = Counter()
    for trace in log:
        for event in trace:
            for attribute in attributes:
                if attribute in event:
                    del event[attribute]
                    removed[attribute] += 1
    if removed:
        print(
            f"[{label}] Configured attribute removal: "
            + ", ".join(f"{name} ({count:,})" for name, count in sorted(removed.items()))
        )
    return dict(sorted(removed.items()))


def main() -> int:
    args = parse_args()
    import pm4py

    if args.workload_blind_attributes and args.clean_static_attributes:
        raise ValueError(
            "--workload-blind-attributes and --clean-static-attributes are mutually exclusive"
        )
    if args.clean_static_attributes and args.use_workload_features:
        raise ValueError(
            "The cleaned static-state sensitivity requires --no-use-workload-features"
        )

    # ``workload_blind_attributes`` removes the manually engineered demand,
    # utilisation, and rank/bin proxy columns.  It is intentionally
    # independent of ProSiT's native dynamic workload/queue features so the
    # two information sources can be separated in a 2x2 factorial ablation.

    # --- Load logs: prefer XES (Vinci approach) over CSV ---
    if args.xes and args.xes.exists():
        print(f"[main] Reading training XES: {args.xes}")
        train_log = normalize_pm4py_event_log(
            pm4py.read_xes(str(args.xes), return_legacy_log_object=True),
            label="training XES",
        )
        dropped_features = restrict_pm4py_log_for_prosit(train_log)
        if dropped_features:
            print(
                "[train] Excluded non-ProSiT XES attributes: "
                + ", ".join(dropped_features)
            )
        print(f"[main] Training log: {len(train_log)} traces (from XES)")
        train_csv = str(args.xes)  # for paramset suffix
    else:
        train_csv = resolve_event_csv(args.input_csv)
        if not is_train_log(train_csv):
            print("[main] WARNING: input is not s6_train.csv.")
        train_df = load_log(train_csv, "train")
        train_log = to_pm4py_log(train_df, "train")
        del train_df

    if args.test_xes and args.test_xes.exists():
        print(f"[main] Reading test XES: {args.test_xes}")
        test_log = normalize_pm4py_event_log(
            pm4py.read_xes(str(args.test_xes), return_legacy_log_object=True),
            label="test XES",
        )
        dropped_features = restrict_pm4py_log_for_prosit(test_log)
        if dropped_features:
            print(
                "[test] Excluded non-ProSiT XES attributes: "
                + ", ".join(dropped_features)
            )
        print(f"[main] Test log: {len(test_log)} traces (from XES)")
    else:
        if not args.test.exists():
            raise FileNotFoundError(f"Test log not found at {args.test}. Run validation/01_train_test_split.py first.")
        test_df = load_log(str(args.test), "test")
        test_log = to_pm4py_log(test_df, "test")
        del test_df

    removed_attributes = configured_removed_attributes(args)
    attribute_removal_counts = {"train": {}, "test": {}}
    if removed_attributes:
        attribute_removal_counts = {
            "train": _drop_event_attributes(
                train_log, removed_attributes, "train"
            ),
            "test": _drop_event_attributes(
                test_log, removed_attributes, "test"
            ),
        }

    if args.write_xes:
        maybe_write_xes(train_log, DEFAULT_XES_DIR / "s6_train.xes", "train")
        maybe_write_xes(test_log, DEFAULT_XES_DIR / "s6_test.xes", "test")

    # 1. Discover the authoritative control flow from the order-preserving log.
    # Sequential generalisation is acceptable for CTB; within-case parallelism
    # is not.  The explicit operator guard makes that domain requirement
    # executable rather than relying on visual inspection.
    import pm4py
    print("[main] Discovering authoritative process tree (inductive miner) ...")
    t0 = time.time()
    diagnostic_tree = pm4py.discover_process_tree_inductive(train_log)
    raw_parallel_operators = process_tree_parallel_count(diagnostic_tree)
    if raw_parallel_operators and args.control_flow_policy == "reject_parallel":
        raise RuntimeError(
            "Inductive-Miner control flow contains "
            f"{raw_parallel_operators} parallel operator(s); CTB requires sequential "
            "activity execution within each truck case."
        )
    if args.control_flow_policy == "expert_sequential_contract":
        authoritative_tree, yard_labels = build_ctb_sequential_contract_tree(train_log)
        repair_applied = str(authoritative_tree) != str(diagnostic_tree)
        print(
            "[main] Applying expert sequential CTB control-flow contract; "
            f"raw parallel operators={raw_parallel_operators}, "
            f"yard activities={len(yard_labels)}"
        )
    else:
        authoritative_tree = diagnostic_tree
        yard_labels = sorted(
            {
                event[ACT_COL]
                for trace in train_log
                for event in trace
                if event.get(ACT_COL) not in {None, "Gate In", "Gate Out"}
            }
        )
        repair_applied = False
    authoritative_parallel_operators = process_tree_parallel_count(authoritative_tree)
    if authoritative_parallel_operators:
        raise RuntimeError(
            "Authoritative CTB source model still contains parallel operators."
        )
    net, im, fm = pm4py.convert_to_petri_net(authoritative_tree)
    coverage = variant_coverage(train_log, test_log)
    print(
        f"[main] Authoritative control flow: {len(net.places)} places, "
        f"{len(net.transitions)} transitions, "
        f"{authoritative_parallel_operators} parallel operators; "
        f"unseen holdout variants={coverage['unseen_test_variants']} "
        f"({coverage['unseen_test_case_share']:.3%})"
    )
    print(f"[main] Control-flow discovery done in {time.time() - t0:.1f}s")

    # 2. Conformance on train AND on test
    conformance_rows: list[dict] = []
    train_conf = token_based_conformance(train_log, net, im, fm, "train")
    train_conf.update(evaluation_metrics(train_log, net, im, fm, "train"))
    train_conf.update(
        alignment_conformance(train_log, net, im, fm, args.alignments_sample, args.random_state, "train")
    )
    conformance_rows.append(train_conf)

    test_conf = token_based_conformance(test_log, net, im, fm, "test")
    test_conf.update(evaluation_metrics(test_log, net, im, fm, "test"))
    test_conf.update(
        alignment_conformance(test_log, net, im, fm, args.alignments_sample, args.random_state, "test")
    )
    conformance_rows.append(test_conf)

    conformance_df = pd.DataFrame(conformance_rows)

    # 3. Output directories — suffix keeps parallel runs (baseline vs
    # workload-features) from overwriting each other. Now that workload
    # features are the ADOPTED DEFAULT, "_workload" is the default suffix
    # and the no-workload variant lands in "_no_workload".
    if args.out_suffix is None:
        if args.use_workload_features:
            args.out_suffix = (
                "_workload_inductive_calibrated"
                if args.ctb_calibration
                else "_workload_inductive"
            )
        elif args.workload_blind_attributes:
            args.out_suffix = (
                "_rules_only_workload_blind_inductive_calibrated"
                if args.ctb_calibration
                else "_rules_only_workload_blind_inductive"
            )
        elif args.clean_static_attributes:
            args.out_suffix = (
                "_clean_static_inductive_calibrated"
                if args.ctb_calibration
                else "_clean_static_inductive"
            )
        else:
            args.out_suffix = (
                "_no_workload_inductive_calibrated"
                if args.ctb_calibration
                else "_no_workload_inductive"
            )
    # s6_train.xes is the same persisted training split as s6_train.csv.
    # Do not mislabel an XES-based run as a full-log discovery.
    source_name = Path(train_csv).stem.lower()
    paramset_suffix = "train80" if source_name == "s6_train" else paramset_suffix_for(train_csv)
    param_dir = Path(resolve_paramset_dir(str(_SCRIPT_DIR), paramset_suffix))
    prosit_dir = param_dir / f"prosit_discovery{args.out_suffix}"
    prosit_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = prosit_dir / "figures"
    print(f"[main] Output folder: {prosit_dir}")

    conformance_df.to_csv(prosit_dir / "prosit_conformance.csv", index=False)
    print(f"[main] Wrote {prosit_dir / 'prosit_conformance.csv'}")

    control_flow_contract = {
        "authoritative_model": (
            "expert_sequential_contract_repair"
            if args.control_flow_policy == "expert_sequential_contract"
            else "inductive_miner_petri_net"
        ),
        "control_flow_policy": args.control_flow_policy,
        "raw_inductive_process_tree": str(diagnostic_tree),
        "raw_parallel_operator_count": raw_parallel_operators,
        "process_tree": str(authoritative_tree),
        "repair_applied": repair_applied,
        "repair_basis": (
            "Terminal experts validated mandatory Gate In and Gate Out, at least "
            "one yard activity, arbitrary sequential yard order, repetitions, and "
            "no within-truck parallel activity."
            if args.control_flow_policy == "expert_sequential_contract"
            else None
        ),
        "yard_activity_labels": yard_labels,
        "observed_case_language": ["Gate In", "one_or_more_yard_activities", "Gate Out"],
        "model_language": ["Gate In", "one_or_more_sequential_yard_activities", "Gate Out"],
        "within_case_parallelism": False,
        "parallel_operator_count": authoritative_parallel_operators,
        "cross_case_resource_multitasking": bool(args.enable_multitasking),
        "petri_net": {
            "places": len(net.places),
            "transitions": len(net.transitions),
            "arcs": len(net.arcs),
        },
        "holdout_variant_coverage": coverage,
    }
    with open(prosit_dir / "control_flow_contract.json", "w") as fh:
        json.dump(control_flow_contract, fh, indent=2)
    print(f"[main] Wrote {prosit_dir / 'control_flow_contract.json'}")

    # 4. Figures
    if not args.skip_figures:
        render_visualisations(train_log, net, im, fm, authoritative_tree, fig_dir)
    else:
        print("[main] --skip-figures: not generating Graphviz outputs.")

    # 5. ProSiT parameter discovery
    params, disc_seconds = discover_prosit_params(train_log, net, im, fm, args)

    calibration_report: dict[str, Any] = {
        "enabled": False,
        "holdout_used": False,
    }
    if args.ctb_calibration:
        print("[ctb] Applying training-only arrival/calendar calibration ...")
        calibration_report = calibrate_ctb_parameters(
            params,
            train_log,
            calibrate_routing=False,
        )
        calibration_report["enabled"] = True
        calibration_path = prosit_dir / "ctb_calibration.json"
        with open(calibration_path, "w") as fh:
            json.dump(calibration_report, fh, indent=2, default=str)
        print(
            "[ctb] Calibration complete: "
            f"arrival P(IAT=0)={calibration_report['arrivals']['zero_probability']:.3%}, "
            "routing=native ProSiT discovery, "
            "resource completion calendar=168 h/week"
        )
        print(f"[ctb] Wrote {calibration_path}")

    params_json_path = prosit_dir / "prosit_params.json"
    params_pkl_path = prosit_dir / "prosit_params.pkl"
    try:
        params.to_json(str(params_json_path))
        print(f"[main] Wrote {params_json_path}")
    except Exception as exc:
        print(f"[main] to_json failed: {exc}")
    try:
        with open(params_pkl_path, "wb") as fh:
            pickle.dump(params, fh)
        print(f"[main] Wrote {params_pkl_path}")
    except Exception as exc:
        print(f"[main] pickle failed: {exc}")

    # 6. Simulation
    manifest = _read_manifest(args.manifest)
    cutoff_str = manifest.get("cutoff_arrival_ts")
    t_start = datetime.fromisoformat(cutoff_str) if cutoff_str else None
    n_traces = args.n_sim_traces
    if n_traces is None:
        test_meta = manifest.get("test", {})
        n_traces = int(test_meta.get("n_cases", len(test_log)))

    sim_summary = {}
    sim_path: Path | None = None
    if args.skip_simulation:
        print("[main] --skip-simulation: not running SimulatorEngine.apply.")
    else:
        sim_log = simulate_prosit(
            params,
            n_traces=n_traces,
            t_start=t_start,
            seed=args.random_state,
            timestamp_resolution=args.timestamp_resolution,
        )
        sim_path = prosit_dir / f"sim_baseline_train80{args.out_suffix}.csv"
        sim_log.to_csv(sim_path, index=False)
        print(f"[main] Wrote {sim_path}")
        sim_summary = {
            "n_traces": int(n_traces),
            "n_events": int(len(sim_log)),
            "t_start": None if t_start is None else t_start.isoformat(),
            "output_csv": str(sim_path),
            "random_seed": int(args.random_state),
            "timestamp_resolution": args.timestamp_resolution,
        }

    # 7. Run summary
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "train_csv": train_csv,
        "test_csv": str(args.test),
        "paramset_folder": str(param_dir),
        "prosit_output_folder": str(prosit_dir),
        "hyperparameters": {
            "max_depth_tree": args.max_depth_tree,
            "min_samples_leaf_cv": args.min_samples_leaf_cv,
            "attribute_mode": args.attribute_mode,
            "use_workload_features": args.use_workload_features,
            "workload_blind_attributes": args.workload_blind_attributes,
            "clean_static_attributes": args.clean_static_attributes,
            "case_attribute_mode": (
                "workload_blind"
                if args.workload_blind_attributes
                else "clean_static"
                if args.clean_static_attributes
                else "all_available"
            ),
            "workload_proxy_attribute_universe": sorted(WORKLOAD_PROXY_ATTRIBUTES),
            "workload_proxy_attributes_removed": sorted(removed_attributes),
            "attribute_removal_counts": attribute_removal_counts,
            "workload_blind_removal_counts": attribute_removal_counts,
            "ctb_calibration": args.ctb_calibration,
            "timestamp_resolution": args.timestamp_resolution,
            "out_suffix": args.out_suffix,
            "random_state": args.random_state,
            "prosit_case_attribute_allowlist": list(
                configured_case_attribute_allowlist(args)
            ),
        },
        "control_flow": control_flow_contract,
        "ctb_calibration": calibration_report,
        "discovery_seconds": disc_seconds,
        "conformance": conformance_rows,
        "simulation": sim_summary,
        "parameter_artifacts": {
            "authoritative_pickle": str(params_pkl_path),
            "portable_prosit_json": str(params_json_path),
            "calibration_companion": (
                str(prosit_dir / "ctb_calibration.json")
                if args.ctb_calibration
                else None
            ),
            "note": (
                "The JSON is the portable ProSiT representation. The pickle is the "
                "authoritative numerical-reproduction snapshot because it preserves "
                "the exact post-calibration in-memory object used for these runs. The "
                "calibration companion records the compact arrival PMF and routing audit."
            ),
        },
    }
    with open(prosit_dir / "prosit_run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"[main] Wrote {prosit_dir / 'prosit_run_summary.json'}")

    # 8. Console recap
    print("\n[main] === Conformance summary ===")
    print(conformance_df.to_string(index=False))
    print("\n[main] Next step:")
    if sim_path:
        rel_sim = os.path.relpath(sim_path, _REPO_ROOT)
        rel_test = os.path.relpath(args.test, _REPO_ROOT)
        print(f"    python validation/02_validate_simulation.py \\")
        print(f"        --sim  {rel_sim} \\")
        print(f"        --real {rel_test} \\")
        print(f"        --label prosit_train80_vs_holdout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
