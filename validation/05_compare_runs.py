"""
05_compare_runs.py

Purpose:
- diff two validation runs (typically the ProSiT baseline vs
  ProSiT-with-workload-features) using the summary metrics computed by
  02_validate_simulation.py;
- produce a compact side-by-side CSV plus a bar-chart figure that makes
  the improvement (or regression) obvious for Chapter 5 of the thesis.

Usage (from the workspace root):

    python validation/05_compare_runs.py \
        --run-a prosit_train80_vs_holdout \
        --run-b prosit_train80_workload_vs_holdout \
        --label-a "ProSiT baseline" \
        --label-b "ProSiT +workload"

Inputs:
- validation/results/<run-a>/summary.json, metrics_activity.csv,
  metrics_waiting.csv, metrics_case.csv, metrics_arrival.csv;
- same for <run-b>.

Outputs (under validation/results/compare_<a>_vs_<b>/):
- summary_compare.csv                overall EMD/KS/mean-abs-error deltas
- activity_service_compare.csv       per-activity EMD/KS for service time
- activity_waiting_compare.csv       per-activity EMD/KS for waiting time
- figures/emd_headline.png           four-bar summary chart
- figures/service_ks_by_activity.png per-activity KS bars
- figures/waiting_ks_by_activity.png per-activity KS bars

Methodological note:
- deltas are computed as run_b minus run_a (positive = run_b is worse for
  distance metrics). All numbers land unchanged in the CSV — the plot is
  only a visual aid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("validation") / "results"

COLOR_A = "#1f77b4"
COLOR_B = "#d62728"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", required=True, help="Reference run label (e.g. prosit_train80_vs_holdout).")
    parser.add_argument("--run-b", required=True, help="Comparison run label (e.g. prosit_train80_workload_vs_holdout).")
    parser.add_argument("--label-a", default=None)
    parser.add_argument("--label-b", default=None)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _load_summary(root: Path, run: str) -> dict:
    with open(root / run / "summary.json", "r") as fh:
        return json.load(fh)


def _load_csv(root: Path, run: str, name: str) -> pd.DataFrame:
    path = root / run / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _merge_activity(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str) -> pd.DataFrame:
    keep = [
        "activity",
        "real_n", "sim_n",
        "real_mean", "sim_mean",
        "real_p90", "sim_p90",
        "wasserstein_min",
        "ks_stat",
        "abs_error_mean",
        "abs_error_p90",
    ]
    a = df_a[[c for c in keep if c in df_a.columns]].add_suffix(f"__{label_a}")
    b = df_b[[c for c in keep if c in df_b.columns]].add_suffix(f"__{label_b}")
    a = a.rename(columns={f"activity__{label_a}": "activity"})
    b = b.rename(columns={f"activity__{label_b}": "activity"})
    merged = a.merge(b, on="activity", how="outer")
    for stat in ("wasserstein_min", "ks_stat", "abs_error_mean", "abs_error_p90"):
        col_a = f"{stat}__{label_a}"
        col_b = f"{stat}__{label_b}"
        if col_a in merged.columns and col_b in merged.columns:
            merged[f"{stat}__delta_b_minus_a"] = merged[col_b] - merged[col_a]
    return merged


def _plot_headline(summary_a: dict, summary_b: dict, label_a: str, label_b: str, out_path: Path) -> None:
    metrics = [
        ("mean_service_time_emd_min", "Service time EMD (min)\navg over activities"),
        ("mean_waiting_time_emd_min", "Waiting time EMD (min)\navg over activities"),
        ("case_turnaround_emd_min", "Case turnaround EMD (min)"),
        ("inter_arrival_emd_min", "Inter-arrival EMD (min)"),
    ]
    vals_a = [summary_a.get(k, np.nan) for k, _ in metrics]
    vals_b = [summary_b.get(k, np.nan) for k, _ in metrics]

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    x = np.arange(len(metrics))
    width = 0.38
    ax.bar(x - width / 2, vals_a, width, color=COLOR_A, label=label_a)
    ax.bar(x + width / 2, vals_b, width, color=COLOR_B, label=label_b)
    for i, (va, vb) in enumerate(zip(vals_a, vals_b)):
        ax.text(i - width / 2, va, f"{va:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, vb, f"{vb:.2f}", ha="center", va="bottom", fontsize=8)
        if np.isfinite(va) and np.isfinite(vb) and va > 0:
            delta = (vb - va) / va * 100.0
            ax.text(i, max(va, vb) * 1.1, f"{delta:+.1f}%", ha="center", va="bottom",
                    fontsize=8, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], fontsize=9)
    ax.set_ylabel("EMD (minutes)")
    ax.set_title("Distributional distance vs held-out test log")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_activity_ks(activity_df: pd.DataFrame, label_a: str, label_b: str,
                      title: str, out_path: Path) -> None:
    col_a = f"ks_stat__{label_a}"
    col_b = f"ks_stat__{label_b}"
    if col_a not in activity_df.columns or col_b not in activity_df.columns:
        return
    df = activity_df[["activity", col_a, col_b]].dropna().copy()
    df = df.sort_values(col_a, ascending=False)
    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.bar(x - width / 2, df[col_a], width, color=COLOR_A, label=label_a)
    ax.bar(x + width / 2, df[col_b], width, color=COLOR_B, label=label_b)
    ax.set_xticks(x)
    ax.set_xticklabels(df["activity"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("KS statistic (lower = closer to real)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    label_a = args.label_a or args.run_a
    label_b = args.label_b or args.run_b

    root = args.out_root
    summary_a = _load_summary(root, args.run_a)
    summary_b = _load_summary(root, args.run_b)

    out_dir = root / f"compare_{args.run_a}__vs__{args.run_b}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary side-by-side
    metric_keys = [
        "real_events", "sim_events", "real_cases", "sim_cases",
        "mean_service_time_emd_min", "mean_waiting_time_emd_min",
        "case_turnaround_emd_min", "inter_arrival_emd_min",
    ]
    rows = []
    for key in metric_keys:
        va = summary_a.get(key, np.nan)
        vb = summary_b.get(key, np.nan)
        row = {"metric": key, label_a: va, label_b: vb, "delta_b_minus_a": (vb - va) if np.isfinite(vb) and np.isfinite(va) else np.nan}
        if np.isfinite(va) and va != 0:
            row["pct_change_b_vs_a"] = (vb - va) / va * 100.0
        else:
            row["pct_change_b_vs_a"] = np.nan
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "summary_compare.csv", index=False)
    print(f"[compare] Wrote {out_dir / 'summary_compare.csv'}")

    # 2. Per-activity service + waiting
    for kind, out_name, title in (
        ("metrics_activity.csv", "activity_service_compare.csv",
         "Service time — KS statistic per activity"),
        ("metrics_waiting.csv", "activity_waiting_compare.csv",
         "Waiting time — KS statistic per activity"),
    ):
        df_a = _load_csv(root, args.run_a, kind)
        df_b = _load_csv(root, args.run_b, kind)
        if df_a.empty or df_b.empty:
            print(f"[compare] Skipping {kind}: missing data")
            continue
        merged = _merge_activity(df_a, df_b, label_a, label_b)
        merged.to_csv(out_dir / out_name, index=False)
        print(f"[compare] Wrote {out_dir / out_name}")
        fig_name = ("service_ks_by_activity.png" if "activity_service" in out_name
                    else "waiting_ks_by_activity.png")
        _plot_activity_ks(merged, label_a, label_b, title, fig_dir / fig_name)

    # 3. Headline figure
    _plot_headline(summary_a, summary_b, label_a, label_b, fig_dir / "emd_headline.png")
    print(f"[compare] Wrote {fig_dir / 'emd_headline.png'}")

    # Console recap
    print("\n[compare] === Overall ===")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
