"""Audit whether the CTB +20% demand scenario approaches RMG saturation.

This is a diagnostic, not a queueing forecast. It computes block-level offered
load from held-out event rates and observed service times, then applies the
scenario's uniform arrival-rate multiplier while holding routing and service
unchanged. Minute-grid overlap statistics expose where the log itself reaches
or exceeds the stated capacity, subject to its one-minute timestamp resolution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log", type=Path, default=REPO / "data/processed/CTB/s6_test.csv"
    )
    parser.add_argument("--capacity-per-block", type=int, default=3)
    parser.add_argument("--demand-multiplier", type=float, default=1.2)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "validation/results/rmg_capacity_pressure_audit",
    )
    return parser.parse_args()


def squared_cv(values: pd.Series) -> float:
    sample = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    mean = float(sample.mean())
    if len(sample) < 2 or mean <= 0:
        return float("nan")
    return float((sample.std(ddof=1) / mean) ** 2)


def main() -> None:
    args = parse_args()
    if args.capacity_per_block <= 0:
        raise ValueError("capacity-per-block must be positive")
    if args.demand_multiplier <= 0:
        raise ValueError("demand-multiplier must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    log = pd.read_csv(
        args.log,
        parse_dates=["start:timestamp", "time:timestamp"],
        low_memory=False,
    )
    rmg = log[
        log["concept:name"].astype(str).str.startswith("RMG_")
        & log["org:resource"].astype(str).str.fullmatch(r"T\d{2}")
    ].copy()
    rmg["service_min"] = (
        rmg["time:timestamp"] - rmg["start:timestamp"]
    ).dt.total_seconds() / 60.0
    if (rmg["service_min"] < 0).any():
        raise ValueError("Negative RMG service duration found.")

    horizon_start = log["start:timestamp"].min().floor("min")
    horizon_end = log["time:timestamp"].max().ceil("min")
    horizon_hours = (horizon_end - horizon_start).total_seconds() / 3600.0
    minute_grid = pd.date_range(
        horizon_start, horizon_end, freq="min", inclusive="left"
    )

    rows: list[dict[str, float | int | str]] = []
    for block, events in rmg.groupby("org:resource", sort=True):
        service = events["service_min"]
        event_rate = len(events) / horizon_hours
        offered_load = event_rate * float(service.mean()) / 60.0
        arrivals = events["start:timestamp"].sort_values()
        inter_arrival = arrivals.diff().dt.total_seconds().div(60.0).dropna()

        changes = np.zeros(len(minute_grid) + 1, dtype=int)
        for start, complete in zip(
            events["start:timestamp"], events["time:timestamp"]
        ):
            first = max(
                0, int(np.floor((start - horizon_start).total_seconds() / 60.0))
            )
            last = int(
                np.ceil((complete - horizon_start).total_seconds() / 60.0)
            )
            # A zero-minute event still occupies its observed minute for the
            # overlap diagnostic; it contributes zero to offered load.
            last = max(first + 1, min(len(minute_grid), last))
            changes[first] += 1
            changes[last] -= 1
        concurrency = np.cumsum(changes[:-1])

        rho = offered_load / args.capacity_per_block
        scenario_rho = rho * args.demand_multiplier
        rows.append(
            {
                "block": block,
                "events": len(events),
                "event_rate_per_hour": event_rate,
                "mean_service_min": float(service.mean()),
                "p90_service_min": float(service.quantile(0.9)),
                "service_cv2": squared_cv(service),
                "inter_arrival_cv2": squared_cv(inter_arrival),
                "offered_load_erlangs": offered_load,
                "nominal_utilization_c3": rho,
                "nominal_utilization_after_demand_change": scenario_rho,
                "demand_multiplier_to_mean_saturation": (
                    1.0 / rho if rho > 0 else float("inf")
                ),
                "observed_mean_concurrency": float(concurrency.mean()),
                "observed_p95_concurrency": float(np.quantile(concurrency, 0.95)),
                "observed_max_concurrency": int(concurrency.max()),
                "share_minutes_at_or_above_capacity": float(
                    np.mean(concurrency >= args.capacity_per_block)
                ),
                "share_minutes_above_capacity": float(
                    np.mean(concurrency > args.capacity_per_block)
                ),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        "nominal_utilization_after_demand_change", ascending=False
    )
    result.to_csv(args.output / "rmg_block_capacity_pressure.csv", index=False)

    top = result.iloc[0]
    summary = {
        "design": (
            "held-out block-level offered-load diagnostic; scenario utilization "
            "holds routing and service distributions fixed and scales arrivals only"
        ),
        "log": str(args.log),
        "horizon_start": str(horizon_start),
        "horizon_end": str(horizon_end),
        "horizon_hours": horizon_hours,
        "capacity_per_block": args.capacity_per_block,
        "demand_multiplier": args.demand_multiplier,
        "rmg_events": int(len(rmg)),
        "blocks": int(len(result)),
        "highest_pressure_block": str(top["block"]),
        "highest_baseline_nominal_utilization": float(
            top["nominal_utilization_c3"]
        ),
        "highest_scenario_nominal_utilization": float(
            top["nominal_utilization_after_demand_change"]
        ),
        "smallest_multiplier_to_mean_saturation": float(
            result["demand_multiplier_to_mean_saturation"].min()
        ),
        "blocks_with_observed_minutes_above_capacity": int(
            (result["share_minutes_above_capacity"] > 0).sum()
        ),
        "limitations": [
            "The calculation is a utilization diagnostic, not a queueing-time estimate.",
            "The demand multiplier assumes unchanged block routing and service times.",
            "Minute-resolved timestamps can create or hide short overlaps.",
            "Recorded org:resource values are block-level pools, not crane trajectories.",
        ],
    }
    with (args.output / "rmg_capacity_pressure_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    plot = result.sort_values("nominal_utilization_c3", ascending=True)
    positions = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    ax.barh(
        positions - 0.18,
        plot["nominal_utilization_c3"],
        height=0.34,
        label="Held-out offered load",
        color="#355C7D",
    )
    ax.barh(
        positions + 0.18,
        plot["nominal_utilization_after_demand_change"],
        height=0.34,
        label=f"After {args.demand_multiplier:.1f}x arrival rate",
        color="#C06C84",
    )
    ax.axvline(
        1.0,
        color="#333333",
        linewidth=1.0,
        linestyle="--",
        label="Mean saturation",
    )
    ax.set_yticks(positions, plot["block"])
    ax.set_xlabel(
        f"Nominal utilization with capacity {args.capacity_per_block} per block"
    )
    ax.set_ylabel("RMG block")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(args.output / "rmg_capacity_pressure.png", dpi=220)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(result.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
