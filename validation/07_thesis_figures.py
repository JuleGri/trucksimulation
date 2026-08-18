"""
07_thesis_figures.py

Purpose:
- generate the composite figures the thesis needs but that are not
  produced by any of the earlier validation scripts, namely
  (i)  a three-way point-plus-CI comparison of the ProSiT configurations
       used as headline evidence in Section 5.4, and
  (ii) a scenario-comparison bar chart used in Section 5.7;
- write the figures directly into the LaTeX repository so
  Chapter 5 can \\includegraphics them without further copying.

Inputs:
- validation/results/prosit_train80_no_rules_ci/mc_summary.csv
- validation/results/prosit_train80_noworkload_ci/mc_summary.csv
- validation/results/prosit_train80_workload_ci/mc_summary.csv
- data/processed/CTB/prosit_simulations/what_if_T22_closed/scenario_comparison_summary.csv
- data/processed/CTB/prosit_simulations/what_if_reduced_rmg_resources/scenario_comparison_summary.csv

Outputs (under the LaTeX repo's figures/ folder):
- figures/three_way_ci.png    grouped-bar CI comparison
- figures/scenario_deltas.png bar chart of baseline vs perturbed KPIs

Usage (from the workspace root of the code repo):
    python validation/07_thesis_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

CI_RUNS = [
    ("no-rules",         "prosit_train80_no_rules_ci",   "#1f77b4"),
    ("rules",            "prosit_train80_noworkload_ci", "#ff7f0e"),
    ("rules+workload",   "prosit_train80_workload_ci",   "#d62728"),
]

METRIC_LABELS = [
    ("mean_service_time_emd_min",  "Service EMD (min)"),
    ("mean_waiting_time_emd_min",  "Waiting EMD (min)"),
    ("case_turnaround_emd_min",    "Case turnaround EMD (min)"),
    ("inter_arrival_emd_min",      "Inter-arrival EMD (min)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latex-figures",
        type=Path,
        default=Path(r"c:\Users\Jule\Documents\Master\Masterthesis"
                     r"\Data-Aware-Process-Simulation-at-CTB\figures"),
        help="Directory in the LaTeX repository where the figures land.",
    )
    return parser.parse_args()


def _load_ci_summary(run_label: str) -> pd.DataFrame:
    path = REPO_ROOT / "validation" / "results" / run_label / "mc_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"CI summary not found at {path}")
    return pd.read_csv(path)


def _load_scenario_summary(subfolder: str) -> pd.DataFrame:
    path = (REPO_ROOT / "data" / "processed" / "CTB" / "prosit_simulations"
            / subfolder / "scenario_comparison_summary.csv")
    if not path.exists():
        raise FileNotFoundError(f"Scenario summary not found at {path}")
    return pd.read_csv(path)


def _row_for_metric(df: pd.DataFrame, metric: str) -> pd.Series:
    hits = df[df["metric"] == metric]
    if hits.empty:
        raise KeyError(f"metric {metric!r} not in summary")
    return hits.iloc[0]


def plot_three_way_ci(out_path: Path) -> None:
    summaries = {label: _load_ci_summary(run) for label, run, _ in CI_RUNS}

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.2))
    for ax, (key, xlabel) in zip(axes, METRIC_LABELS):
        means = []
        yerr_lo = []
        yerr_hi = []
        colors = []
        for label, _, color in CI_RUNS:
            row = _row_for_metric(summaries[label], key)
            m = float(row["mean"])
            lo = float(row["ci95_lo"])
            hi = float(row["ci95_hi"])
            means.append(m)
            yerr_lo.append(m - lo)
            yerr_hi.append(hi - m)
            colors.append(color)
        x = np.arange(len(CI_RUNS))
        ax.bar(x, means, yerr=np.vstack([yerr_lo, yerr_hi]),
               capsize=6, color=colors, alpha=0.85, edgecolor="black")
        for xi, m, lo, hi in zip(x, means, np.array(means) - np.array(yerr_lo),
                                  np.array(means) + np.array(yerr_hi)):
            ax.text(xi, hi, f"{m:.2f}\n[{lo:.2f}, {hi:.2f}]",
                    ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for lbl, _, _ in CI_RUNS], fontsize=8, rotation=15)
        ax.set_title(xlabel, fontsize=10)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.suptitle("Held-out fidelity by ProSiT configuration ($N_{seeds}=10$, "
                 "95% Student-t CI)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[figures] Wrote {out_path}")


def plot_scenarios(out_path: Path) -> None:
    df_a = _load_scenario_summary("what_if_T22_closed")
    df_b = _load_scenario_summary("what_if_reduced_rmg_resources")

    kpis = [
        ("mean_turnaround_min",     "Mean turnaround (min)"),
        ("mean_rmg_service_min",    "Mean RMG service (min)"),
        ("mean_rmg_waiting_min",    "Mean RMG waiting (min)"),
    ]

    def _rows(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        # Baseline row is the one with scenario == 'baseline'; scenario row
        # is whatever else appears in the summary.
        base = df[df["scenario"] == "baseline"].iloc[0]
        pert = df[df["scenario"] != "baseline"].iloc[0]
        return base, pert

    a_base, a_pert = _rows(df_a)
    b_base, b_pert = _rows(df_b)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    for ax, (key, ylabel) in zip(axes, kpis):
        groups = ["Baseline", "T22 closed", "T22 + reduced RMG"]
        vals = [
            (a_base[key] + b_base[key]) / 2.0,  # displayed as reference average
            a_pert[key],
            b_pert[key],
        ]
        # Compute the reference values per bar (each scenario keeps its own
        # baseline internally to avoid mixing draws) — the first bar is a
        # visual anchor only, so we plot with a distinct hatch.
        vals[0] = a_base[key]
        colors = ["#7f7f7f", "#1f77b4", "#d62728"]
        x = np.arange(len(groups))
        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="black")
        bars[0].set_hatch("//")
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        # Second reference bar for the reduced-RMG baseline
        ax.axhline(b_base[key], color="grey", linewidth=0.8, linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.suptitle("What-if scenario outcomes (single-seed simulation, "
                 "rules$+$workload configuration)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[figures] Wrote {out_path}")


def main() -> int:
    args = parse_args()
    args.latex_figures.mkdir(parents=True, exist_ok=True)
    plot_three_way_ci(args.latex_figures / "three_way_ci.png")
    plot_scenarios(args.latex_figures / "scenario_deltas.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
