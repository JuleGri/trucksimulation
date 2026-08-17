"""
02_validate_simulation.py

Purpose:
- compare a simulated event log against the held-out real event log
  produced by 01_train_test_split.py;
- provide distributional evidence (Wasserstein / EMD and two-sample
  Kolmogorov-Smirnov) that the simulator reproduces the real process
  at the activity, resource, arrival and case level;
- write a compact metrics bundle that can be tabulated in Chapter 5 of
  the thesis and reused by 03_validation_plots.py.

Inputs:
- --real  : path to the real held-out event log (default: s6_test.csv);
- --sim   : path to the simulated event log (any CSV with the standard
            case:concept:name / concept:name / *timestamp columns).

Outputs (under validation/results/<label>/):
- metrics_activity.csv       per-activity service-time metrics
- metrics_waiting.csv        per-activity waiting-time metrics
- metrics_case.csv           case-level KPIs (turnaround, n_events, etc.)
- metrics_arrival.csv        inter-arrival time metrics
- summary.json               overall roll-up used by the plot script
- prepared_real.parquet /    per-event data cleaned and aligned so that
  prepared_sim.parquet        03_validation_plots.py can reuse it.

Methodological notes:
- EMD is computed with scipy.stats.wasserstein_distance in the natural
  unit of the variable (minutes for durations, minutes for
  inter-arrivals). It is scale-dependent and comparable only within the
  same variable across configurations.
- The KS p-value is capped at n=10,000 sub-sampled from each side to
  keep the statistic informative (with 89 000 events the null is
  rejected almost mechanically).
- Summary-statistic errors (MAE/RMSE of mean, p50, p90) provide an
  operationally interpretable secondary view.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


CASE_COL = "case:concept:name"
ACT_COL = "concept:name"
RES_COL = "org:resource"
TS_ENABLED = "enabled:timestamp"
TS_START = "start:timestamp"
TS_COMPLETE = "time:timestamp"

DEFAULT_REAL = Path("data") / "processed" / "CTB" / "s6_test.csv"
DEFAULT_OUT_ROOT = Path("validation") / "results"

# Cap on sample size fed into KS to keep the test informative.
KS_SUBSAMPLE = 10_000
# Discard non-finite / negative durations.
DURATION_LOWER = 0.0
# Cap unrealistic outliers (in minutes) before EMD to avoid single-event dominance.
DURATION_UPPER = 60.0 * 24.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, default=DEFAULT_REAL,
                        help="Real held-out event log CSV.")
    parser.add_argument("--sim", type=Path, required=True,
                        help="Simulated event log CSV.")
    parser.add_argument("--label", type=str, required=True,
                        help="Sub-folder name under validation/results/ for this run.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--seed", type=int, default=42,
                        help="Sub-sampling seed for the KS test.")
    return parser.parse_args()


def _to_ts(df: pd.DataFrame, cols: Sequence[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")


def _clip_duration(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out[(out >= DURATION_LOWER) & (out <= DURATION_UPPER)]
    return out


def load_and_prepare(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[{label}] Event log not found: {path.resolve()}")
    print(f"[{label}] Reading {path}")
    df = pd.read_csv(path)
    _to_ts(df, [TS_ENABLED, TS_START, TS_COMPLETE])
    if ACT_COL not in df.columns:
        raise KeyError(f"[{label}] Column {ACT_COL!r} missing.")
    if CASE_COL not in df.columns:
        raise KeyError(f"[{label}] Column {CASE_COL!r} missing.")

    if TS_START in df.columns and TS_COMPLETE in df.columns:
        df["service_time_min"] = (df[TS_COMPLETE] - df[TS_START]).dt.total_seconds() / 60.0
    else:
        df["service_time_min"] = np.nan

    if TS_ENABLED in df.columns and TS_START in df.columns:
        df["waiting_time_min"] = (df[TS_START] - df[TS_ENABLED]).dt.total_seconds() / 60.0
        df["waiting_time_min"] = df["waiting_time_min"].clip(lower=0.0)
    else:
        df["waiting_time_min"] = np.nan

    # Arrival timestamp per case (earliest available timestamp).
    ts_cols_available = [c for c in (TS_ENABLED, TS_START, TS_COMPLETE) if c in df.columns]
    df["_event_ts"] = df[ts_cols_available].min(axis=1)

    print(f"[{label}] events={len(df):,}  cases={df[CASE_COL].nunique():,}  "
          f"activities={df[ACT_COL].nunique()}")
    return df


def _ks_two_sample(real: np.ndarray, sim: np.ndarray, seed: int) -> tuple[float, float]:
    if len(real) < 2 or len(sim) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if len(real) > KS_SUBSAMPLE:
        real = rng.choice(real, KS_SUBSAMPLE, replace=False)
    if len(sim) > KS_SUBSAMPLE:
        sim = rng.choice(sim, KS_SUBSAMPLE, replace=False)
    result = stats.ks_2samp(real, sim, alternative="two-sided", method="asymp")
    return float(result.statistic), float(result.pvalue)


def _describe(x: np.ndarray) -> dict[str, float]:
    if len(x) == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "p50": float("nan"), "p90": float("nan"), "min": float("nan"),
                "max": float("nan")}
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def compare_distributions(real: np.ndarray, sim: np.ndarray, seed: int) -> dict[str, float]:
    row: dict[str, float] = {}
    for prefix, values in (("real", real), ("sim", sim)):
        for k, v in _describe(values).items():
            row[f"{prefix}_{k}"] = v

    if len(real) >= 2 and len(sim) >= 2:
        row["wasserstein_min"] = float(stats.wasserstein_distance(real, sim))
    else:
        row["wasserstein_min"] = float("nan")

    ks_stat, ks_p = _ks_two_sample(real, sim, seed)
    row["ks_stat"] = ks_stat
    row["ks_pvalue"] = ks_p

    for stat_name in ("mean", "p50", "p90"):
        real_val = row.get(f"real_{stat_name}", float("nan"))
        sim_val = row.get(f"sim_{stat_name}", float("nan"))
        row[f"abs_error_{stat_name}"] = abs(real_val - sim_val)
        if np.isfinite(real_val) and abs(real_val) > 1e-9:
            row[f"rel_error_{stat_name}"] = abs(real_val - sim_val) / abs(real_val)
        else:
            row[f"rel_error_{stat_name}"] = float("nan")
    return row


def compare_by_activity(real_df: pd.DataFrame, sim_df: pd.DataFrame, value_col: str,
                        seed: int) -> pd.DataFrame:
    activities = sorted(set(real_df[ACT_COL].dropna().unique())
                        | set(sim_df[ACT_COL].dropna().unique()))
    rows = []
    for act in activities:
        real_vals = _clip_duration(real_df.loc[real_df[ACT_COL] == act, value_col]).to_numpy()
        sim_vals = _clip_duration(sim_df.loc[sim_df[ACT_COL] == act, value_col]).to_numpy()
        row = {"activity": act}
        row.update(compare_distributions(real_vals, sim_vals, seed))
        rows.append(row)
    return pd.DataFrame(rows)


def case_kpis(df: pd.DataFrame) -> pd.DataFrame:
    ts_cols = [c for c in (TS_START, TS_COMPLETE, TS_ENABLED) if c in df.columns]
    if not ts_cols:
        return pd.DataFrame(columns=["case", "turnaround_min", "n_events"])
    grouped = df.groupby(CASE_COL)
    start = grouped[ts_cols].min().min(axis=1)
    end = grouped[ts_cols].max().max(axis=1)
    turnaround = (end - start).dt.total_seconds() / 60.0
    n_events = grouped.size().rename("n_events")
    result = pd.DataFrame({"turnaround_min": turnaround, "n_events": n_events}).reset_index()
    return result


def inter_arrival_series(df: pd.DataFrame) -> np.ndarray:
    arrival_per_case = df.groupby(CASE_COL)["_event_ts"].min().dropna().sort_values()
    if len(arrival_per_case) < 2:
        return np.array([], dtype=float)
    inter = arrival_per_case.diff().dt.total_seconds().div(60.0).dropna().to_numpy()
    return inter[np.isfinite(inter) & (inter >= 0.0)]


def to_parquet_or_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
        return path.with_suffix(".parquet")
    except Exception:
        alt = path.with_suffix(".csv")
        df.to_csv(alt, index=False)
        return alt


def main() -> int:
    args = parse_args()
    out_dir = args.out_root / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    real_df = load_and_prepare(args.real, label="real")
    sim_df = load_and_prepare(args.sim, label="sim")

    print("[validate] Comparing service-time distributions per activity ...")
    activity_metrics = compare_by_activity(real_df, sim_df, "service_time_min", args.seed)
    activity_metrics.insert(0, "kind", "service_time")
    activity_metrics.to_csv(out_dir / "metrics_activity.csv", index=False)

    print("[validate] Comparing waiting-time distributions per activity ...")
    waiting_metrics = compare_by_activity(real_df, sim_df, "waiting_time_min", args.seed)
    waiting_metrics.insert(0, "kind", "waiting_time")
    waiting_metrics.to_csv(out_dir / "metrics_waiting.csv", index=False)

    print("[validate] Comparing case-level KPIs ...")
    real_case = case_kpis(real_df)
    sim_case = case_kpis(sim_df)
    case_rows = []
    for col in ("turnaround_min", "n_events"):
        row = {"metric": col}
        real_vals = _clip_duration(real_case[col]) if col == "turnaround_min" else real_case[col].dropna()
        sim_vals = _clip_duration(sim_case[col]) if col == "turnaround_min" else sim_case[col].dropna()
        row.update(compare_distributions(np.asarray(real_vals), np.asarray(sim_vals), args.seed))
        case_rows.append(row)
    case_metrics = pd.DataFrame(case_rows)
    case_metrics.to_csv(out_dir / "metrics_case.csv", index=False)

    print("[validate] Comparing inter-arrival time distribution ...")
    real_iat = inter_arrival_series(real_df)
    sim_iat = inter_arrival_series(sim_df)
    arrival_row = {"metric": "inter_arrival_min"}
    arrival_row.update(compare_distributions(real_iat, sim_iat, args.seed))
    arrival_metrics = pd.DataFrame([arrival_row])
    arrival_metrics.to_csv(out_dir / "metrics_arrival.csv", index=False)

    # Persist prepared logs so the plot script does not repeat the cleaning.
    keep_cols = [c for c in (CASE_COL, ACT_COL, RES_COL, TS_ENABLED, TS_START,
                             TS_COMPLETE, "service_time_min", "waiting_time_min",
                             "_event_ts") if c in real_df.columns]
    prepared_real = to_parquet_or_csv(real_df[keep_cols].copy(), out_dir / "prepared_real")
    prepared_sim = to_parquet_or_csv(sim_df[keep_cols].copy(), out_dir / "prepared_sim")

    def _agg_emd(df: pd.DataFrame) -> float:
        vals = df["wasserstein_min"].replace([np.inf, -np.inf], np.nan).dropna()
        return float(vals.mean()) if not vals.empty else float("nan")

    summary = {
        "label": args.label,
        "real_path": str(args.real),
        "sim_path": str(args.sim),
        "real_events": int(len(real_df)),
        "sim_events": int(len(sim_df)),
        "real_cases": int(real_df[CASE_COL].nunique()),
        "sim_cases": int(sim_df[CASE_COL].nunique()),
        "n_activities_compared": int(activity_metrics.shape[0]),
        "mean_service_time_emd_min": _agg_emd(activity_metrics),
        "mean_waiting_time_emd_min": _agg_emd(waiting_metrics),
        "case_turnaround_emd_min": float(case_metrics.loc[case_metrics["metric"] == "turnaround_min",
                                                          "wasserstein_min"].iloc[0]),
        "inter_arrival_emd_min": float(arrival_metrics.loc[0, "wasserstein_min"]),
        "prepared_real_path": str(prepared_real),
        "prepared_sim_path": str(prepared_sim),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[validate] Summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:>32}: {v:,.3f}")
        else:
            print(f"  {k:>32}: {v}")
    print(f"[validate] All artefacts written under {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
