"""
03_validation_plots.py

Purpose:
- turn the metrics bundle produced by 02_validate_simulation.py into the
  publication-ready figures needed in the Results chapter;
- generate CDF overlays, box plots and QQ plots for the activity-level
  service and waiting time distributions;
- produce a case-level turnaround CDF and an inter-arrival CDF.

Inputs:
- validation/results/<label>/prepared_real.(parquet|csv)
- validation/results/<label>/prepared_sim.(parquet|csv)
- validation/results/<label>/metrics_activity.csv
- validation/results/<label>/metrics_waiting.csv

Outputs (under validation/results/<label>/figures/):
- cdf_service_<activity>.png
- cdf_waiting_<activity>.png
- box_service.png, box_waiting.png
- qq_service_<activity>.png
- cdf_turnaround.png
- cdf_inter_arrival.png

Methodological note:
- CDFs are the primary visual: they show the marginal distribution
  overlap directly and can be annotated with the KS statistic;
- box plots are the compact summary used for the Chapter 5 headline
  figure;
- QQ plots are added for the activities that dominate the process
  (LL_delivery, RMG_delivery, HO2_receive) so that reviewers can see
  whether the tail is off.
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


ACT_COL = "concept:name"
DEFAULT_OUT_ROOT = Path("validation") / "results"

FIGSIZE_CDF = (5.5, 3.8)
FIGSIZE_BOX = (10.0, 4.5)
FIGSIZE_QQ = (5.0, 5.0)
DURATION_UPPER = 60.0 * 24.0

REAL_COLOR = "#1f77b4"
SIM_COLOR = "#d62728"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", type=str, required=True,
                        help="Sub-folder under validation/results/ from step 2.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--max-activities", type=int, default=12,
                        help="Only produce per-activity plots for the top N most-frequent activities.")
    return parser.parse_args()


def _load_prepared(path_stub: Path) -> pd.DataFrame:
    parquet = path_stub.with_suffix(".parquet")
    csv = path_stub.with_suffix(".csv")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        df = pd.read_csv(csv)
        for col in ("enabled:timestamp", "start:timestamp", "time:timestamp", "_event_ts"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df
    raise FileNotFoundError(f"No prepared log at {parquet} or {csv}. Run step 2 first.")


def _clean_series(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    values = values[(values >= 0.0) & (values <= DURATION_UPPER)]
    return values.to_numpy()


def _empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(values) == 0:
        return np.array([]), np.array([])
    ordered = np.sort(values)
    y = np.arange(1, len(ordered) + 1) / len(ordered)
    return ordered, y


def _plot_cdf(ax: plt.Axes, real: np.ndarray, sim: np.ndarray, title: str,
              xlabel: str, ks_stat: float | None = None) -> None:
    rx, ry = _empirical_cdf(real)
    sx, sy = _empirical_cdf(sim)
    if len(rx):
        ax.step(rx, ry, where="post", color=REAL_COLOR, label=f"Real (n={len(real):,})", linewidth=1.6)
    if len(sx):
        ax.step(sx, sy, where="post", color=SIM_COLOR, label=f"Sim  (n={len(sim):,})",
                linewidth=1.6, linestyle="--")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical CDF")
    if ks_stat is not None and np.isfinite(ks_stat):
        title = f"{title}\nKS={ks_stat:.3f}"
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)


def _plot_qq(ax: plt.Axes, real: np.ndarray, sim: np.ndarray, title: str, unit: str) -> None:
    if len(real) < 2 or len(sim) < 2:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    qs = np.linspace(0.01, 0.99, 99)
    rq = np.quantile(real, qs)
    sq = np.quantile(sim, qs)
    lo = float(min(rq.min(), sq.min()))
    hi = float(max(rq.max(), sq.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle=":")
    ax.plot(rq, sq, marker="o", markersize=3, linestyle="none", color=SIM_COLOR)
    ax.set_xlabel(f"Real quantile ({unit})")
    ax.set_ylabel(f"Sim quantile ({unit})")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)


def _boxplot(ax: plt.Axes, real_lists: list[np.ndarray], sim_lists: list[np.ndarray],
             labels: list[str], ylabel: str, title: str) -> None:
    positions_real = np.arange(len(labels)) * 3 + 0.7
    positions_sim = positions_real + 1.0
    bp_real = ax.boxplot(real_lists, positions=positions_real, widths=0.8, patch_artist=True,
                         showfliers=False)
    bp_sim = ax.boxplot(sim_lists, positions=positions_sim, widths=0.8, patch_artist=True,
                        showfliers=False)
    for patch in bp_real["boxes"]:
        patch.set_facecolor(REAL_COLOR)
        patch.set_alpha(0.55)
    for patch in bp_sim["boxes"]:
        patch.set_facecolor(SIM_COLOR)
        patch.set_alpha(0.55)
    for artist in ("medians",):
        for line in bp_real[artist] + bp_sim[artist]:
            line.set_color("black")
    ax.set_xticks(positions_real + 0.5)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(handles=[bp_real["boxes"][0], bp_sim["boxes"][0]],
              labels=["Real", "Sim"], loc="upper right", fontsize=8)


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(name))


def _select_top_activities(real_df: pd.DataFrame, sim_df: pd.DataFrame, top_n: int) -> list[str]:
    counts = pd.concat([real_df[ACT_COL], sim_df[ACT_COL]]).value_counts()
    return counts.head(top_n).index.tolist()


def _ks_lookup(metrics_df: pd.DataFrame, activity: str) -> float | None:
    row = metrics_df[metrics_df["activity"] == activity]
    if row.empty:
        return None
    val = row.iloc[0].get("ks_stat")
    return float(val) if pd.notna(val) else None


def main() -> int:
    args = parse_args()
    run_dir = args.out_root / args.run_label
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}. Run step 2 first.")
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    real_df = _load_prepared(run_dir / "prepared_real")
    sim_df = _load_prepared(run_dir / "prepared_sim")

    activity_metrics = pd.read_csv(run_dir / "metrics_activity.csv")
    waiting_metrics = pd.read_csv(run_dir / "metrics_waiting.csv")

    top_activities = _select_top_activities(real_df, sim_df, args.max_activities)
    print(f"[plots] Top activities: {top_activities}")

    # Per-activity CDFs and QQ plots.
    for activity in top_activities:
        for col, kind, unit, metric_df in (
            ("service_time_min", "service", "min", activity_metrics),
            ("waiting_time_min", "waiting", "min", waiting_metrics),
        ):
            real_values = _clean_series(real_df.loc[real_df[ACT_COL] == activity, col])
            sim_values = _clean_series(sim_df.loc[sim_df[ACT_COL] == activity, col])
            if kind == "waiting" and len(real_values) == 0:
                # A real log without model-derived enabled:timestamp has no
                # observed ProSiT waiting-time reference.  Do not draw a
                # misleading artificial zero distribution.
                continue
            fig, ax = plt.subplots(figsize=FIGSIZE_CDF)
            _plot_cdf(ax, real_values, sim_values,
                      title=f"{activity} — {kind} time",
                      xlabel=f"{kind} time ({unit})",
                      ks_stat=_ks_lookup(metric_df, activity))
            fig.tight_layout()
            fig.savefig(figures_dir / f"cdf_{kind}_{_safe_filename(activity)}.png", dpi=160)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=FIGSIZE_QQ)
            _plot_qq(ax, real_values, sim_values,
                     title=f"QQ — {activity} — {kind} time", unit=unit)
            fig.tight_layout()
            fig.savefig(figures_dir / f"qq_{kind}_{_safe_filename(activity)}.png", dpi=160)
            plt.close(fig)

    # Compact box-plot overview.
    for col, kind in (("service_time_min", "service"), ("waiting_time_min", "waiting")):
        if kind == "waiting" and not np.isfinite(
            pd.to_numeric(real_df[col], errors="coerce")
        ).any():
            print("[plots] Skipping waiting-time plots: real holdout has no model-derived enablement.")
            continue
        real_lists, sim_lists, labels = [], [], []
        for activity in top_activities:
            r = _clean_series(real_df.loc[real_df[ACT_COL] == activity, col])
            s = _clean_series(sim_df.loc[sim_df[ACT_COL] == activity, col])
            if len(r) == 0 and len(s) == 0:
                continue
            real_lists.append(r if len(r) else np.array([0.0]))
            sim_lists.append(s if len(s) else np.array([0.0]))
            labels.append(activity)
        if not labels:
            continue
        fig, ax = plt.subplots(figsize=FIGSIZE_BOX)
        _boxplot(ax, real_lists, sim_lists, labels,
                 ylabel=f"{kind} time (min)",
                 title=f"Real vs simulated {kind} times")
        fig.tight_layout()
        fig.savefig(figures_dir / f"box_{kind}.png", dpi=160)
        plt.close(fig)

    # Case-level turnaround CDF.
    def _case_turnaround(df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in ("start:timestamp", "time:timestamp", "enabled:timestamp") if c in df.columns]
        if not cols:
            return np.array([])
        grouped = df.groupby("case:concept:name")[cols]
        start = grouped.min().min(axis=1)
        end = grouped.max().max(axis=1)
        values = (end - start).dt.total_seconds().div(60.0).dropna().to_numpy()
        return values[(values >= 0.0) & (values <= DURATION_UPPER)]

    fig, ax = plt.subplots(figsize=FIGSIZE_CDF)
    _plot_cdf(ax, _case_turnaround(real_df), _case_turnaround(sim_df),
              title="Case turnaround time",
              xlabel="turnaround time (min)")
    fig.tight_layout()
    fig.savefig(figures_dir / "cdf_turnaround.png", dpi=160)
    plt.close(fig)

    # Inter-arrival CDF.
    def _inter_arrival(df: pd.DataFrame) -> np.ndarray:
        if "_event_ts" not in df.columns:
            return np.array([])
        arrival = df.groupby("case:concept:name")["_event_ts"].min().dropna().sort_values()
        if len(arrival) < 2:
            return np.array([])
        deltas = arrival.diff().dt.total_seconds().div(60.0).dropna().to_numpy()
        return deltas[np.isfinite(deltas) & (deltas >= 0.0)]

    fig, ax = plt.subplots(figsize=FIGSIZE_CDF)
    _plot_cdf(ax, _inter_arrival(real_df), _inter_arrival(sim_df),
              title="Case inter-arrival time",
              xlabel="inter-arrival (min)")
    fig.tight_layout()
    fig.savefig(figures_dir / "cdf_inter_arrival.png", dpi=160)
    plt.close(fig)

    print(f"[plots] Figures written to {figures_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
