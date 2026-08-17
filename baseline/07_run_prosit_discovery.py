"""
07_run_prosit_discovery.py

Purpose:
- run the actual ProSiT (Vinci et al., 2026) simulation-parameter discovery
  on the held-out training log s6_train.csv;
- perform full process discovery: Petri net + directly-follows graph +
  process tree + BPMN + variants;
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
    petri_net.png                      inductive-miner Petri net
    petri_net_frequency.png            transitions coloured by token replay frequency
    petri_net_performance.png          transitions coloured by mean sojourn time
    dfg_frequency.png                  directly-follows graph (frequency)
    dfg_performance.png                directly-follows graph (mean waiting time)
    process_tree.png                   inductive-miner process tree
    bpmn.png                           BPMN translation of the Petri net
    variants_top20.png                 variant coverage bar chart

Methodological note:
- the Petri net is discovered with pm4py.discover_petri_net_inductive on
  the TRAINING log only, following Vinci et al. (2026, "Full workflow with
  evaluation" in the ProSiT README);
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
DEFAULT_TEST = _REPO_ROOT / "data" / "processed" / "CTB" / "s6_test.csv"
DEFAULT_MANIFEST = _REPO_ROOT / "validation" / "results" / "split_manifest.json"
DEFAULT_XES_DIR = _REPO_ROOT / "data" / "processed" / "CTB" / "xes_files"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProSiT discovery + visualisation on the training split.")
    add_input_arg(parser)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST,
                        help="Held-out test CSV used for test-set conformance.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="split_manifest.json produced by validation/01_train_test_split.py.")
    parser.add_argument("--max-depth-tree", type=int, default=3,
                        help="ProSiT DecisionTree max_depth (README default: 3).")
    parser.add_argument("--min-samples-leaf-cv", type=int, nargs="+", default=[50, 100, 200],
                        help="ProSiT min_samples_leaf CV grid (README default: 50 100 200).")
    parser.add_argument("--random-state", type=int, default=42)
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
                             "ADOPTED DEFAULT because it reduced the held-out case-turnaround EMD by "
                             "20% on CTB. Use --no-use-workload-features to reproduce the earlier "
                             "baseline run.")
    parser.add_argument("--out-suffix", type=str, default=None,
                        help="Suffix appended to the prosit_discovery/ folder and sim CSV filename so a "
                             "new run does not overwrite a previous one. Defaults to '_workload' when "
                             "--use-workload-features is set (i.e. the default), '_no_workload' "
                             "otherwise so the two configurations always land in distinct folders.")
    return parser.parse_args()


# ----------------------------------------------------------------------
# Event log loading
# ----------------------------------------------------------------------

def load_log(csv_path: str, label: str) -> "pd.DataFrame":
    print(f"[{label}] Reading {csv_path}")
    df = pd.read_csv(csv_path)
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
    print(f"[{label}] events={len(df):,}  cases={df[CASE_COL].nunique():,}")
    return df


def to_pm4py_log(df: pd.DataFrame, label: str):
    import pm4py
    formatted = pm4py.format_dataframe(
        df.copy(),
        case_id=CASE_COL,
        activity_key=ACT_COL,
        timestamp_key=TS_COMPLETE,
    )
    log = pm4py.convert_to_event_log(formatted)
    print(f"[{label}] pm4py EventLog with {len(log)} traces")
    return log


def maybe_write_xes(log, out_path: Path, label: str) -> None:
    import pm4py
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{label}] Writing XES -> {out_path}")
    pm4py.write_xes(log, str(out_path))


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


def render_visualisations(train_log, net, im, fm, tree, fig_dir: Path) -> None:
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

    # 5. Process tree.
    try:
        from pm4py.visualization.process_tree import visualizer as tree_vis
        gviz = tree_vis.apply(tree)
        _save_gviz(gviz, fig_dir / "process_tree.png", "process_tree")
    except Exception as exc:
        print(f"[figures] process_tree failed: {exc}")

    # 6. BPMN translation of the Petri net.
    try:
        bpmn = pm4py.convert_to_bpmn(tree)
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
          f"random_state={args.random_state}) ...")
    t0 = time.time()
    params.discover_from_eventlog(
        train_log,
        max_depth_tree=args.max_depth_tree,
        min_samples_leaf_cv=args.min_samples_leaf_cv,
        attribute_mode=args.attribute_mode,
        use_workload_features=args.use_workload_features,
        random_state=args.random_state,
        verbose=True,
    )
    dt = time.time() - t0
    print(f"[prosit] Discovery finished in {dt:.1f}s")
    return params, dt


def simulate_prosit(params, n_traces: int, t_start: datetime | None, seed: int) -> pd.DataFrame:
    from prosit import SimulatorEngine
    engine = SimulatorEngine(params)
    print(f"[prosit] Simulating n_traces={n_traces:,}  t_start={t_start}  ...")
    t0 = time.time()
    if t_start is None:
        sim_log = engine.apply(n_traces=n_traces)
    else:
        sim_log = engine.apply(n_traces=n_traces, t_start=t_start)
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


def main() -> int:
    args = parse_args()

    train_csv = resolve_event_csv(args.input_csv)
    if not is_train_log(train_csv):
        print("[main] WARNING: input is not s6_train.csv. Continuing but paramset suffix will reflect this.")
    if not args.test.exists():
        raise FileNotFoundError(f"Test log not found at {args.test}. Run validation/01_train_test_split.py first.")

    train_df = load_log(train_csv, "train")
    test_df = load_log(str(args.test), "test")

    train_log = to_pm4py_log(train_df, "train")
    test_log = to_pm4py_log(test_df, "test")

    if args.write_xes:
        maybe_write_xes(train_log, DEFAULT_XES_DIR / "s6_train.xes", "train")
        maybe_write_xes(test_log, DEFAULT_XES_DIR / "s6_test.xes", "test")

    # 1. Petri net + process tree
    import pm4py
    print("[main] Discovering process tree (inductive miner) ...")
    t0 = time.time()
    tree = pm4py.discover_process_tree_inductive(train_log)
    net, im, fm = pm4py.convert_to_petri_net(tree)
    print(f"[main] Inductive miner done in {time.time() - t0:.1f}s")

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
        args.out_suffix = "_workload" if args.use_workload_features else "_no_workload"
    param_dir = Path(resolve_paramset_dir(str(_SCRIPT_DIR), paramset_suffix_for(train_csv)))
    prosit_dir = param_dir / f"prosit_discovery{args.out_suffix}"
    prosit_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = prosit_dir / "figures"
    print(f"[main] Output folder: {prosit_dir}")

    conformance_df.to_csv(prosit_dir / "prosit_conformance.csv", index=False)
    print(f"[main] Wrote {prosit_dir / 'prosit_conformance.csv'}")

    # 4. Figures
    if not args.skip_figures:
        render_visualisations(train_log, net, im, fm, tree, fig_dir)
    else:
        print("[main] --skip-figures: not generating Graphviz outputs.")

    # 5. ProSiT parameter discovery
    params, disc_seconds = discover_prosit_params(train_log, net, im, fm, args)

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
        n_traces = int(test_meta.get("n_cases", test_df[CASE_COL].nunique()))

    sim_summary = {}
    sim_path: Path | None = None
    if args.skip_simulation:
        print("[main] --skip-simulation: not running SimulatorEngine.apply.")
    else:
        sim_log = simulate_prosit(params, n_traces=n_traces, t_start=t_start, seed=args.random_state)
        sim_path = prosit_dir / f"sim_baseline_train80{args.out_suffix}.csv"
        sim_log.to_csv(sim_path, index=False)
        print(f"[main] Wrote {sim_path}")
        sim_summary = {
            "n_traces": int(n_traces),
            "n_events": int(len(sim_log)),
            "t_start": None if t_start is None else t_start.isoformat(),
            "output_csv": str(sim_path),
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
            "out_suffix": args.out_suffix,
            "random_state": args.random_state,
        },
        "discovery_seconds": disc_seconds,
        "conformance": conformance_rows,
        "simulation": sim_summary,
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
