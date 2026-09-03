"""Create a paired comparison of the executed CTB demand experiments."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent / "experimental_results"
CLEAN = ROOT / "calendar_preserving_10seeds_base2000_final"
COMPRESSED = ROOT / "saturation_full"
OUT = ROOT / "paired_comparison_10seeds_base2000"
FACTORS = (1.0, 1.2, 1.5, 2.0, 2.4, 3.0)
METRICS = (
    "mean_turnaround_min",
    "p90_turnaround_min",
    "mean_rmg_service_min",
    "mean_rmg_pre_service_min",
    "arrival_rate_per_elapsed_hour",
    "arrival_span_min",
)


def ci(values: pd.Series):
    x = pd.to_numeric(values, errors="coerce").dropna()
    n = len(x)
    mean = float(x.mean())
    if n < 2:
        return n, mean, np.nan, mean, mean
    sd = float(x.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return n, mean, sd, mean - half, mean + half


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(CLEAN / "calendar_preserving_replications.csv")
    compressed = pd.read_csv(COMPRESSED / "saturation_replications.csv")
    clean["experiment"] = "calendar_preserving"
    compressed["experiment"] = "arrival_horizon_compression"
    clean["demand_multiplier"] = clean["demand_multiplier"].astype(float)
    compressed["demand_multiplier"] = compressed["demand_multiplier"].astype(float)
    clean.to_csv(OUT / "calendar_preserving_replications.csv", index=False)
    compressed.to_csv(OUT / "compression_replications.csv", index=False)

    rows = []
    for factor in FACTORS:
        a = clean[clean.demand_multiplier.eq(factor)].set_index("seed")
        b = compressed[compressed.demand_multiplier.eq(factor)].set_index("seed")
        for metric in METRICS:
            values = a[metric] - b[metric]
            n, mean, sd, lo, hi = ci(values)
            rows.append({
                "demand_multiplier": factor,
                "metric": metric,
                "n": n,
                "calendar_preserving_mean": float(a[metric].mean()),
                "compression_mean": float(b[metric].mean()),
                "mean_difference_clean_minus_compression": mean,
                "std_difference": sd,
                "ci95_lo": lo,
                "ci95_hi": hi,
            })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "paired_metric_comparison.csv", index=False)

    clean_means = clean.groupby("demand_multiplier")[list(METRICS)].mean()
    compressed_means = compressed.groupby("demand_multiplier")[list(METRICS)].mean()
    audit = {
        "calendar_preserving_closed_period_arrivals": int(clean.closed_period_gate_arrivals.sum()),
        "calendar_preserving_sunday_arrivals": int(clean.sunday_gate_arrivals.sum()),
        "calendar_preserving_calendar_compliance_rate": float(clean.arrival_calendar_compliance.mean()),
        "calendar_preserving_cases": int(clean.sim_cases.sum()),
        "compression_cases": int(compressed.sim_cases.sum()),
        "calendar_preserving_mean_by_factor": clean_means.round(6).to_dict(orient="index"),
        "compression_mean_by_factor": compressed_means.round(6).to_dict(orient="index"),
    }
    (OUT / "paired_comparison_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"WROTE {OUT}")


if __name__ == "__main__":
    main()
