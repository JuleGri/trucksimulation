"""
04_baseline_percentile_sensitivity.py

Status: DIAGNOSTIC / DESCRIPTIVE ONLY (2026-08 revision)
  This script is no longer part of the main simulation pipeline.
  The transition-baseline table is not used to construct enabled:timestamp
  and does not affect ProSiT's simulation-parameter discovery.
  It is retained as descriptive evidence that the model-derived
  pre-service delay includes a non-trivial physical-movement component.

Purpose:
- quantify how sensitive empirical transition durations are to the choice
  of percentile;
- provide the descriptive table and figure used in
  Section 5.6 (Descriptive Analysis of Transition Durations);
- illustrate the physical-movement component embedded in ProSiT's
  model-derived pre-service delay.

Inputs:
- data/interim/CTB/s4_its_fahrplan_with_case_features_anonymized.csv (default;
  override with --input). Any file that exposes the columns
  FAHRPLAN_UID, ANLAUFPUNKT_SEQUENZNUMMER, ANLAUFPUNKT_HALTESTELLE,
  FP_DAUER_GATEIN_ERSTER_ALP_SEK, ALP_DAUER_NAECHSTER_ALP_SEK and
  FP_DAUER_LETZT_ALP_GATEOUT_SEK works.

Outputs (under validation/results/percentile_sensitivity/):
- transition_baselines_by_percentile.csv
- waiting_time_shift_summary.csv
- figures/percentile_shift.png
- figures/percentile_shift_top_transitions.png

Methodological note:
- the current production code (010_generate_transition_baseline.ipynb)
  uses the 5th percentile; the review incorrectly attributed the choice
  to the 10th percentile. This script therefore evaluates {5, 10, 25, 50}
  as candidates and reports the shift in the reconstructed
  "waiting time = actual_transition - baseline" per transition, so the
  effect on downstream discovery is transparent.
- transitions with fewer than --min-samples observations are skipped, to
  mirror the production filter.
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


DEFAULT_INPUT = Path("data") / "interim" / "CTB" / "s4_its_fahrplan_with_case_features_anonymized.csv"
DEFAULT_OUT = Path("validation") / "results" / "percentile_sensitivity"
DEFAULT_PERCENTILES = (5, 10, 25, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--percentiles", type=int, nargs="+", default=list(DEFAULT_PERCENTILES))
    parser.add_argument("--min-samples", type=int, default=20,
                        help="Only report transitions with at least this many observations.")
    return parser.parse_args()


def load_fahrplan(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fahrplan not found at {path.resolve()}")
    print(f"[sens] Reading {path}")
    df = pd.read_csv(path, sep=";", low_memory=False)
    required = {
        "FAHRPLAN_UID",
        "ANLAUFPUNKT_HALTESTELLE",
        "FP_DAUER_GATEIN_ERSTER_ALP_SEK",
        "ALP_DAUER_NAECHSTER_ALP_SEK",
        "FP_DAUER_LETZT_ALP_GATEOUT_SEK",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for sensitivity: {sorted(missing)}")
    # The anonymised interim CSV has a broken ANLAUFPUNKT_SEQUENZNUMMER
    # (all values are the same scientific-notation string). Fall back to
    # numeric parsing when possible and otherwise trust the CSV row order,
    # which the export preserved per FAHRPLAN_UID.
    if "ANLAUFPUNKT_SEQUENZNUMMER" in df.columns:
        seq = pd.to_numeric(df["ANLAUFPUNKT_SEQUENZNUMMER"], errors="coerce")
        if seq.nunique(dropna=True) <= 1:
            print("[sens] WARNING: ANLAUFPUNKT_SEQUENZNUMMER is degenerate in the "
                  "anonymised export; falling back to CSV row order per case.")
            df = df.copy()
            df["_seq"] = np.arange(len(df))
        else:
            df = df.copy()
            df["_seq"] = seq
    else:
        df = df.copy()
        df["_seq"] = np.arange(len(df))
    return df


def build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the transition list from 010_generate_transition_baseline.ipynb.

    Vectorised implementation: sorts by (FAHRPLAN_UID, _seq) and derives
    all three transition families with column operations rather than a
    per-case Python loop.
    """
    work = df.dropna(subset=["FAHRPLAN_UID", "_seq"]).copy()
    work = work.sort_values(["FAHRPLAN_UID", "_seq"]).reset_index(drop=True)

    # Rank position within case (0-based).
    grp = work.groupby("FAHRPLAN_UID", sort=False)
    work["_pos"] = grp.cumcount()
    work["_max_pos"] = grp["_pos"].transform("max")

    frames = []

    # Gate In -> first stop
    first_rows = work.loc[work["_pos"].eq(0) &
                          work["FP_DAUER_GATEIN_ERSTER_ALP_SEK"].notna() &
                          work["FP_DAUER_GATEIN_ERSTER_ALP_SEK"].gt(0),
                          ["ANLAUFPUNKT_HALTESTELLE", "FP_DAUER_GATEIN_ERSTER_ALP_SEK"]]
    if not first_rows.empty:
        frames.append(pd.DataFrame({
            "activity": "Gate In",
            "next_activity": first_rows["ANLAUFPUNKT_HALTESTELLE"].to_numpy(),
            "duration_sec": first_rows["FP_DAUER_GATEIN_ERSTER_ALP_SEK"].astype(float).to_numpy(),
        }))

    # Stop_i -> Stop_{i+1} (only for rows that have a next-stop duration and
    # are not the last row within their case).
    intra = work.loc[work["ALP_DAUER_NAECHSTER_ALP_SEK"].notna() &
                     work["ALP_DAUER_NAECHSTER_ALP_SEK"].gt(0) &
                     work["_pos"].lt(work["_max_pos"]),
                     ["FAHRPLAN_UID", "_pos", "ANLAUFPUNKT_HALTESTELLE",
                      "ALP_DAUER_NAECHSTER_ALP_SEK"]].copy()
    if not intra.empty:
        # Next haltestelle within the same case.
        intra["next_haltestelle"] = intra["ANLAUFPUNKT_HALTESTELLE"].shift(-1)
        # Only keep rows where the shift stayed inside the same case.
        intra["_next_case"] = intra["FAHRPLAN_UID"].shift(-1)
        intra = intra[intra["FAHRPLAN_UID"] == intra["_next_case"]]
        if not intra.empty:
            frames.append(pd.DataFrame({
                "activity": intra["ANLAUFPUNKT_HALTESTELLE"].to_numpy(),
                "next_activity": intra["next_haltestelle"].to_numpy(),
                "duration_sec": intra["ALP_DAUER_NAECHSTER_ALP_SEK"].astype(float).to_numpy(),
            }))

    # Last stop -> Gate Out
    last_rows = work.loc[work["_pos"].eq(work["_max_pos"]) &
                         work["FP_DAUER_LETZT_ALP_GATEOUT_SEK"].notna() &
                         work["FP_DAUER_LETZT_ALP_GATEOUT_SEK"].gt(0),
                         ["ANLAUFPUNKT_HALTESTELLE", "FP_DAUER_LETZT_ALP_GATEOUT_SEK"]]
    if not last_rows.empty:
        frames.append(pd.DataFrame({
            "activity": last_rows["ANLAUFPUNKT_HALTESTELLE"].to_numpy(),
            "next_activity": "Gate Out",
            "duration_sec": last_rows["FP_DAUER_LETZT_ALP_GATEOUT_SEK"].astype(float).to_numpy(),
        }))

    if not frames:
        return pd.DataFrame(columns=["activity", "next_activity", "duration_sec"])
    return pd.concat(frames, ignore_index=True)


def compute_baselines(transitions: pd.DataFrame, percentiles: list[int],
                      min_samples: int) -> pd.DataFrame:
    grouped = transitions.groupby(["activity", "next_activity"])
    rows = []
    for (a, b), grp in grouped:
        if len(grp) < min_samples:
            continue
        durations = grp["duration_sec"].to_numpy()
        row = {
            "activity": a,
            "next_activity": b,
            "samples": int(len(grp)),
            "mean_sec": float(durations.mean()),
            "median_sec": float(np.median(durations)),
            "max_sec": float(durations.max()),
        }
        for p in percentiles:
            row[f"p{p}_sec"] = float(np.percentile(durations, p))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["activity", "next_activity"]).reset_index(drop=True)


def summarize_shift(baselines: pd.DataFrame, percentiles: list[int]) -> pd.DataFrame:
    ref = f"p{percentiles[0]}_sec"
    rows = []
    for p in percentiles:
        col = f"p{p}_sec"
        shift = baselines[col] - baselines[ref]
        row = {
            "percentile": p,
            "reference_percentile": percentiles[0],
            "n_transitions": int(len(baselines)),
            "mean_baseline_sec": float(baselines[col].mean()),
            "median_baseline_sec": float(np.median(baselines[col])),
            "mean_shift_vs_ref_sec": float(shift.mean()),
            "median_shift_vs_ref_sec": float(np.median(shift)),
            "p90_shift_vs_ref_sec": float(np.percentile(shift, 90)) if len(shift) else float("nan"),
            "corr_with_ref": float(baselines[col].corr(baselines[ref])) if len(baselines) > 1 else float("nan"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_percentile_shift(baselines: pd.DataFrame, percentiles: list[int], out_path: Path) -> None:
    ref = f"p{percentiles[0]}_sec"
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for p in percentiles:
        col = f"p{p}_sec"
        shift_min = (baselines[col] - baselines[ref]).to_numpy() / 60.0
        ordered = np.sort(shift_min)
        cdf = np.arange(1, len(ordered) + 1) / len(ordered)
        ax.step(ordered, cdf, where="post", label=f"p{p}", linewidth=1.5)
    ax.set_xlabel(f"baseline shift vs p{percentiles[0]} (min)")
    ax.set_ylabel("CDF across transitions")
    ax.set_title("Sensitivity of transition baseline to percentile choice")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_top_transitions(baselines: pd.DataFrame, percentiles: list[int], out_path: Path,
                         top_n: int = 12) -> None:
    top = baselines.sort_values("samples", ascending=False).head(top_n).copy()
    labels = [f"{r.activity} -> {r.next_activity}" for r in top.itertuples()]
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    xs = np.arange(len(labels))
    width = 0.8 / len(percentiles)
    for i, p in enumerate(percentiles):
        vals = (top[f"p{p}_sec"] / 60.0).to_numpy()
        ax.bar(xs + i * width, vals, width=width, label=f"p{p}")
    ax.set_xticks(xs + (len(percentiles) - 1) * width / 2)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("baseline duration (min)")
    ax.set_title(f"Baseline across percentiles for top {top_n} transitions")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_fahrplan(args.input)
    transitions = build_transitions(df)
    print(f"[sens] Extracted {len(transitions):,} transitions")

    baselines = compute_baselines(transitions, args.percentiles, args.min_samples)
    baselines.to_csv(args.out_dir / "transition_baselines_by_percentile.csv", index=False)
    print(f"[sens] Retained {len(baselines):,} transitions with >= {args.min_samples} samples")

    summary = summarize_shift(baselines, args.percentiles)
    summary.to_csv(args.out_dir / "waiting_time_shift_summary.csv", index=False)

    plot_percentile_shift(baselines, args.percentiles, figures_dir / "percentile_shift.png")
    plot_top_transitions(baselines, args.percentiles, figures_dir / "percentile_shift_top_transitions.png")

    print("[sens] Shift summary (values in seconds):")
    print(summary.to_string(index=False))
    print(f"[sens] Artefacts written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
