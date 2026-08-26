"""
06_multi_seed_ci.py

Purpose:
- estimate the Monte-Carlo uncertainty of the ProSiT simulator by running
  the same discovered SimulatorParameters bundle N times with different
  random seeds and computing distributional-distance metrics against the
  held-out test log for each run;
- produce 95 % confidence intervals for the four headline KPIs
  (service-time EMD, waiting-time EMD, case turnaround EMD, inter-arrival
  EMD), plus replication-wise checks of the calibrated CTB fidelity
  constraints (no artificial >24 h durations, simultaneous-arrival share,
  and yard-activity-rate error), so that Chapter 5 can report point
  estimates with error bars instead of single Monte-Carlo draws.

Usage (from the workspace root):

    python validation/06_multi_seed_ci.py \
        --params baseline/discovery_params/params_20260816_214403_train80/prosit_discovery_workload/prosit_params.pkl \
        --real   data/processed/CTB/s6_test.csv \
        --manifest validation/results/split_manifest.json \
        --n-seeds 10 \
        --label prosit_train80_workload_ci

Inputs:
- --params    : path to a pickled SimulatorParameters instance;
- --real      : held-out real event log CSV;
- --manifest  : split_manifest.json (used for t_start and default n_traces);
- --n-seeds   : number of independent replications (10 is enough for a
                Student-t 95 % CI on 4 KPIs).

Outputs (under validation/results/<label>/):
- mc_replications.csv        one row per seed, all metrics.
- mc_summary.csv             one row per metric, with mean/std/95 % CI.
- yard_activity_emd_replications.csv
                            one row per seed and yard activity; gates excluded.
- yard_activity_emd_summary.csv
                            per-yard-activity EMD means and 95 % CIs.
- figures/mc_violin.png      violin plot of the four EMD metrics across
                             seeds.
- figures/mc_kpi_ci.png      point-plus-CI bar for the four EMD metrics.

Methodological note:
- ProSiT's SimulatorEngine.apply() does not expose an explicit seed, so
  we seed Python's random and numpy.random before each call. The measured
  spread therefore reflects the sampling variance of the fitted
  distributions, not variance across discoveries. That is the appropriate
  Monte-Carlo interval to report next to a single-discovery point
  estimate.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# Make sibling validation modules importable when run from repo root.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
# Re-use the metrics implementation from step 2 so the numbers stay
# comparable between single-shot and multi-seed runs.
_step2 = __import__("02_validate_simulation")
from _eventlog_contract import (  # noqa: E402
    ORDER_COL,
    eventlog_contract_report,
)
from _prosit_ctb_calibration import simulate_ctb  # noqa: E402

CASE_COL = _step2.CASE_COL
ACT_COL = _step2.ACT_COL

DEFAULT_ROOT = Path("validation") / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True,
                        help="Path to a pickled SimulatorParameters.")
    parser.add_argument("--real", type=Path,
                        default=Path("data") / "processed" / "CTB" / "s6_test.csv",
                        help="Held-out real event log.")
    parser.add_argument("--manifest", type=Path,
                        default=Path("validation") / "results" / "split_manifest.json",
                        help="split_manifest.json (for t_start and default n_traces).")
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Number of independent simulation seeds.")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Seed for the first replication; seed[i] = base + i.")
    parser.add_argument("--n-traces", type=int, default=None,
                        help="Cases per replication. Defaults to the test-set size.")
    parser.add_argument("--t-start", type=str, default=None,
                        help=("ISO simulation start timestamp. If provided, this overrides "
                              "the cutoff stored in the split manifest."))
    parser.add_argument("--label", type=str, required=True,
                        help="Sub-folder under validation/results/ for this run.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--fail-on-contract-violations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop if any replication produces a structurally/temporally invalid case.",
    )
    parser.add_argument(
        "--timestamp-resolution",
        choices=("minute", "native"),
        default="minute",
        help="Use the CTB log's minute observation resolution (default) or native ProSiT timestamps.",
    )
    return parser.parse_args()


def _load_simulation_window(manifest_path: Path, n_traces_override: int | None,
                            real_df: pd.DataFrame,
                            t_start_override: str | None = None) -> tuple[int, datetime | None]:
    n_traces = n_traces_override
    t_start: datetime | None = None
    if manifest_path.exists():
        with open(manifest_path, "r") as fh:
            manifest = json.load(fh)
        if n_traces is None:
            n_traces = int(manifest.get("test", {}).get("n_cases", real_df[CASE_COL].nunique()))
        cutoff = manifest.get("cutoff_arrival_ts")
        if cutoff:
            t_start = datetime.fromisoformat(cutoff)
    if t_start_override:
        t_start = datetime.fromisoformat(t_start_override.replace("Z", "+00:00"))
    if n_traces is None:
        n_traces = int(real_df[CASE_COL].nunique())
    return int(n_traces), t_start


def _load_params(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Params pickle not found at {path}.")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    # ProSiT 1.0.3 sometimes emits the same column name twice in the
    # simulated log (event vs case attribute). Keep first occurrence.
    return df.loc[:, ~df.columns.duplicated()].copy()


def _run_one_seed(
    params,
    seed: int,
    n_traces: int,
    t_start: datetime | None,
    timestamp_resolution: str = "minute",
) -> pd.DataFrame:
    sim_log = simulate_ctb(
        params,
        n_traces=n_traces,
        t_start=t_start,
        seed=seed,
        timestamp_resolution=("min" if timestamp_resolution == "minute" else None),
    )
    return _dedupe_columns(sim_log)


def _metrics_for_seed(seed: int, sim_df: pd.DataFrame, real_df: pd.DataFrame,
                       activities: list[str]) -> dict[str, float]:
    # Reuse step-2 helpers so the numbers stay identical to the single-shot
    # validation.
    # step 2 exposes load_and_prepare via file only, so we prep in-memory
    # to avoid disk round-trips.
    sim = sim_df.copy()
    _step2._to_ts(sim, [_step2.TS_ENABLED, _step2.TS_START, _step2.TS_COMPLETE])
    if _step2.TS_START in sim.columns and _step2.TS_COMPLETE in sim.columns:
        sim["service_time_min"] = (sim[_step2.TS_COMPLETE] - sim[_step2.TS_START]).dt.total_seconds() / 60.0
    else:
        sim["service_time_min"] = np.nan
    if _step2.TS_ENABLED in sim.columns and _step2.TS_START in sim.columns:
        sim["waiting_time_min"] = (sim[_step2.TS_START] - sim[_step2.TS_ENABLED]).dt.total_seconds() / 60.0
        sim["waiting_time_min"] = sim["waiting_time_min"].clip(lower=0.0)
    else:
        sim["waiting_time_min"] = np.nan
    ts_cols_available = [c for c in (_step2.TS_ENABLED, _step2.TS_START, _step2.TS_COMPLETE)
                        if c in sim.columns]
    sim["_event_ts"] = sim[ts_cols_available].min(axis=1)

    # Per-activity service + waiting EMD (averaged over activities present
    # in both logs).
    svc_rows = _step2.compare_by_activity(real_df, sim, "service_time_min", seed)
    wait_rows = _step2.compare_by_activity(real_df, sim, "waiting_time_min", seed)
    yard_robust = _step2.yard_service_emd_aggregates(svc_rows, "wasserstein_min")
    yard_raw = _step2.yard_service_emd_aggregates(svc_rows, "wasserstein_min_raw")

    def _mean_emd(df: pd.DataFrame) -> float:
        vals = df["wasserstein_min"].replace([np.inf, -np.inf], np.nan).dropna()
        return float(vals.mean()) if not vals.empty else float("nan")

    # Case-level turnaround.
    real_case = _step2.case_kpis(real_df)
    sim_case = _step2.case_kpis(sim)
    real_ta = _step2._clip_duration(real_case["turnaround_min"]).to_numpy()
    sim_ta = _step2._clip_duration(sim_case["turnaround_min"]).to_numpy()
    turnaround_stats = _step2.compare_distributions(real_ta, sim_ta, seed)
    turnaround_raw_stats = _step2.compare_distributions(
        _step2._raw_duration(real_case["turnaround_min"]).to_numpy(),
        _step2._raw_duration(sim_case["turnaround_min"]).to_numpy(),
        seed,
    )

    # Inter-arrival.
    real_iat = _step2.inter_arrival_series(real_df)
    sim_iat = _step2.inter_arrival_series(sim)
    iat_stats = _step2.compare_distributions(real_iat, sim_iat, seed)

    # Calibration-specific fidelity diagnostics.  Durations are evaluated
    # on the raw (unclipped) values so a long artificial tail cannot be
    # hidden by the plotting/EMD clipping used in the headline metrics.
    sim_service_raw = _step2._raw_duration(sim["service_time_min"])
    sim_turnaround_raw = _step2._raw_duration(sim_case["turnaround_min"])

    yard_activities = [
        activity for activity in activities
        if activity not in {"Gate In", "Gate Out"}
    ]
    real_activity_rate = (
        real_df[ACT_COL].value_counts().reindex(yard_activities, fill_value=0)
        / max(int(real_df[CASE_COL].nunique()), 1)
    )
    sim_activity_rate = (
        sim[ACT_COL].value_counts().reindex(yard_activities, fill_value=0)
        / max(int(sim[CASE_COL].nunique()), 1)
    )
    activity_rate_abs_error = (real_activity_rate - sim_activity_rate).abs()

    sim_contract_input = sim.copy()
    sim_contract_input[ORDER_COL] = sim_contract_input.groupby(
        CASE_COL, sort=False
    ).cumcount()
    contract = eventlog_contract_report(
        sim_contract_input, _already_ordered=True
    )

    result = {
        "seed": int(seed),
        "sim_events": int(len(sim)),
        "sim_cases": int(sim[CASE_COL].nunique()),
        "mean_service_time_emd_min": _mean_emd(svc_rows),
        "mean_service_time_emd_min_raw": float(
            svc_rows["wasserstein_min_raw"].replace([np.inf, -np.inf], np.nan).dropna().mean()
        ),
        "yard_service_time_emd_unweighted_min": yard_robust["unweighted_mean_min"],
        "yard_service_time_emd_frequency_weighted_min": yard_robust[
            "frequency_weighted_mean_min"
        ],
        "yard_service_time_emd_unweighted_min_raw": yard_raw["unweighted_mean_min"],
        "yard_service_time_emd_frequency_weighted_min_raw": yard_raw[
            "frequency_weighted_mean_min"
        ],
        "mean_waiting_time_emd_min": _mean_emd(wait_rows),
        "case_turnaround_emd_min": float(turnaround_stats.get("wasserstein_min", float("nan"))),
        "case_turnaround_emd_min_raw": float(
            turnaround_raw_stats.get("wasserstein_min", float("nan"))
        ),
        "case_turnaround_ks": float(turnaround_stats.get("ks_stat", float("nan"))),
        "case_turnaround_sim_mean": float(turnaround_stats.get("sim_mean", float("nan"))),
        "case_turnaround_sim_p90": float(turnaround_stats.get("sim_p90", float("nan"))),
        "inter_arrival_emd_min": float(iat_stats.get("wasserstein_min", float("nan"))),
        "inter_arrival_ks": float(iat_stats.get("ks_stat", float("nan"))),
        "sim_zero_inter_arrival_share": (
            float(np.mean(np.asarray(sim_iat) == 0.0)) if len(sim_iat) else float("nan")
        ),
        "sim_service_events_above_24h": int((sim_service_raw > 24.0 * 60.0).sum()),
        "sim_turnaround_cases_above_24h": int((sim_turnaround_raw > 24.0 * 60.0).sum()),
        "yard_activity_rate_l1_error": float(activity_rate_abs_error.sum()),
        "yard_activity_rate_max_abs_error": (
            float(activity_rate_abs_error.max())
            if not activity_rate_abs_error.empty else float("nan")
        ),
        "gate_only_cases": int(contract["gate_only_cases"]),
        "wrong_case_boundary_cases": int(contract["wrong_case_boundary_cases"]),
        "within_case_overlap_cases": int(contract.get("within_case_overlap_cases", 0)),
        "decreasing_completion_cases": int(contract.get("decreasing_completion_cases", 0)),
        "gate_out_before_final_yard_cases": int(
            contract.get("gate_out_before_final_yard_cases", 0)
        ),
    }
    yard_rows = svc_rows.loc[
        ~svc_rows["activity"].isin(["Gate In", "Gate Out"])
    ]
    for _, activity_row in yard_rows.iterrows():
        activity = str(activity_row["activity"])
        result[f"yard_service_emd__{activity}"] = float(activity_row["wasserstein_min"])
        result[f"yard_service_emd_raw__{activity}"] = float(
            activity_row["wasserstein_min_raw"]
        )
    return result


def _ci_from_samples(samples: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float, float]:
    samples = samples[np.isfinite(samples)]
    n = len(samples)
    if n < 2:
        m = float(np.mean(samples)) if n else float("nan")
        return m, float("nan"), m, m
    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1))
    t = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    half = t * std / np.sqrt(n)
    return mean, std, mean - half, mean + half


def _plot_violin(replications: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("yard_service_time_emd_unweighted_min", "Yard service EMD (min)"),
        ("mean_waiting_time_emd_min", "Waiting EMD (min)"),
        ("case_turnaround_emd_min", "Turnaround EMD (min)"),
        ("inter_arrival_emd_min", "IAT EMD (min)"),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    data = []
    labels = []
    for key, label in metrics:
        vals = replications[key].dropna().to_numpy()
        if len(vals) >= 2:
            data.append(vals)
            labels.append(label)
    if not data:
        plt.close(fig)
        return
    positions = np.arange(len(data)) + 1
    parts = ax.violinplot(data, positions=positions, widths=0.7, showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#2c7fb8")
        pc.set_alpha(0.55)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("EMD (min)")
    ax.set_title(f"Monte-Carlo spread across {len(replications)} seeds")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_ci_bar(summary: pd.DataFrame, out_path: Path) -> None:
    df = summary[summary["metric"].isin([
        "yard_service_time_emd_unweighted_min",
        "mean_waiting_time_emd_min",
        "case_turnaround_emd_min",
        "inter_arrival_emd_min",
    ])].copy()
    if df.empty:
        return
    df["label"] = df["metric"].map({
        "yard_service_time_emd_unweighted_min": "Yard service EMD",
        "mean_waiting_time_emd_min": "Waiting EMD",
        "case_turnaround_emd_min": "Turnaround EMD",
        "inter_arrival_emd_min": "IAT EMD",
    })
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    x = np.arange(len(df))
    means = df["mean"].to_numpy()
    lo = df["ci95_lo"].to_numpy()
    hi = df["ci95_hi"].to_numpy()
    yerr = np.vstack([means - lo, hi - means])
    ax.bar(x, means, yerr=yerr, capsize=6, color="#2c7fb8", alpha=0.8, edgecolor="black")
    for i, (m, l, h) in enumerate(zip(means, lo, hi)):
        ax.text(i, h, f"{m:.2f}\n[{l:.2f}, {h:.2f}]", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=9)
    ax.set_ylabel("EMD (min)")
    ax.set_title(f"Point estimate + 95 % Student-t CI (n = {int(df['n'].iloc[0])})")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = args.out_root / args.label
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    print(f"[mc] Loading params from {args.params}")
    params = _load_params(args.params)

    print(f"[mc] Loading real log from {args.real}")
    real_df = _step2.load_and_prepare(args.real, label="real")

    n_traces, t_start = _load_simulation_window(
        args.manifest, args.n_traces, real_df, args.t_start
    )
    activities = sorted(set(real_df[ACT_COL].dropna().unique()))

    print(f"[mc] n_seeds={args.n_seeds}  n_traces={n_traces:,}  t_start={t_start}")

    rows: list[dict] = []
    for i in range(args.n_seeds):
        seed = args.base_seed + i
        print(f"[mc]   seed={seed} ...", end=" ", flush=True)
        sim_df = _run_one_seed(
            params,
            seed,
            n_traces,
            t_start,
            args.timestamp_resolution,
        )
        row = _metrics_for_seed(seed, sim_df, real_df, activities)
        rows.append(row)
        contract_violations = sum(
            row[key]
            for key in (
                "wrong_case_boundary_cases",
                "within_case_overlap_cases",
                "decreasing_completion_cases",
                "gate_out_before_final_yard_cases",
            )
        )
        print(
            f"turnaround_emd={row['case_turnaround_emd_min']:.3f} min  "
            f"raw={row['case_turnaround_emd_min_raw']:.3f} min  "
            f"iat_zero={row['sim_zero_inter_arrival_share']:.3%}  "
            f">24h(service/case)={row['sim_service_events_above_24h']}/"
            f"{row['sim_turnaround_cases_above_24h']}  "
            f"gate_only={row['gate_only_cases']}  "
            f"hard_contract_violations={contract_violations}"
        )
        if contract_violations and args.fail_on_contract_violations:
            raise RuntimeError(
                f"Seed {seed} violated the sequential CTB case contract: "
                f"wrong_boundary={row['wrong_case_boundary_cases']}, "
                f"overlap={row['within_case_overlap_cases']}, "
                f"decreasing_completion={row['decreasing_completion_cases']}, "
                f"gate_out_before_yard={row['gate_out_before_final_yard_cases']}"
            )

    replications = pd.DataFrame(rows)
    replications_path = out_dir / "mc_replications.csv"
    replications.to_csv(replications_path, index=False)
    print(f"[mc] Wrote {replications_path}")

    # Aggregate.
    summary_rows = []
    summary_columns = [
                "mean_service_time_emd_min", "mean_service_time_emd_min_raw",
                "yard_service_time_emd_unweighted_min",
                "yard_service_time_emd_frequency_weighted_min",
                "yard_service_time_emd_unweighted_min_raw",
                "yard_service_time_emd_frequency_weighted_min_raw",
                "mean_waiting_time_emd_min", "case_turnaround_emd_min",
                "case_turnaround_emd_min_raw", "case_turnaround_ks",
                "case_turnaround_sim_mean", "case_turnaround_sim_p90",
                "inter_arrival_emd_min", "inter_arrival_ks",
                "sim_zero_inter_arrival_share",
                "sim_service_events_above_24h",
                "sim_turnaround_cases_above_24h",
                "yard_activity_rate_l1_error",
                "yard_activity_rate_max_abs_error",
                "gate_only_cases", "wrong_case_boundary_cases",
                "within_case_overlap_cases", "decreasing_completion_cases",
                "gate_out_before_final_yard_cases",
    ]
    summary_columns.extend(sorted(
        column for column in replications.columns
        if column.startswith("yard_service_emd__")
        or column.startswith("yard_service_emd_raw__")
    ))
    for col in summary_columns:
        m, s, lo, hi = _ci_from_samples(replications[col].to_numpy())
        summary_rows.append({
            "metric": col,
            "n": int(replications[col].dropna().shape[0]),
            "mean": m,
            "std": s,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "half_width": (hi - lo) / 2.0 if np.isfinite(hi) and np.isfinite(lo) else float("nan"),
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "mc_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[mc] Wrote {summary_path}")

    yard_replication_columns = [
        column for column in replications.columns
        if column.startswith("yard_service_emd__")
    ]
    yard_replications = replications[["seed", *yard_replication_columns]].melt(
        id_vars="seed", var_name="metric", value_name="wasserstein_min"
    )
    yard_replications["activity"] = yard_replications["metric"].str.replace(
        "yard_service_emd__", "", regex=False
    )
    yard_replications[["seed", "activity", "wasserstein_min"]].to_csv(
        out_dir / "yard_activity_emd_replications.csv", index=False
    )
    yard_summary = summary[
        summary["metric"].str.startswith("yard_service_emd__")
    ].copy()
    yard_summary["activity"] = yard_summary["metric"].str.replace(
        "yard_service_emd__", "", regex=False
    )
    yard_summary[["activity", "n", "mean", "std", "ci95_lo", "ci95_hi", "half_width"]].to_csv(
        out_dir / "yard_activity_emd_summary.csv", index=False
    )

    _plot_violin(replications, out_dir / "figures" / "mc_violin.png")
    _plot_ci_bar(summary, out_dir / "figures" / "mc_kpi_ci.png")

    # Console recap.
    print("\n[mc] === Summary ===")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
