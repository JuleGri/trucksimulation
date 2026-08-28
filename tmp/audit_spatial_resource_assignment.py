from pathlib import Path
import argparse
import pickle
import sys

import pandas as pd


REPO = Path(r"C:\Users\Jule\Documents\Master\Masterthesis\trucksimulation")
DEFAULT_PARAMS = REPO / "validation" / "results" / "prosit_inductive_calibrated_scenarios_rmg_cap3_ci" / "params_baseline_rmg_max_concurrency_3.pkl"
sys.path.insert(0, str(REPO))

from _prosit_ctb_calibration import simulate_ctb  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--n-traces", type=int, default=5_000)
    args = parser.parse_args()
    with args.params.open("rb") as handle:
        params = pickle.load(handle)
    simulation = simulate_ctb(
        params,
        n_traces=args.n_traces,
        t_start=pd.Timestamp("2026-04-20 18:30:00").to_pydatetime(),
        seed=42,
        timestamp_resolution=None,
    )
    activity = simulation["concept:name"].astype(str)
    yard = simulation[activity.str.startswith(("RMG_", "HO2_", "LL_"))].copy()
    yard["resource"] = yard["org:resource"].astype(str)
    yard["target"] = yard["target_area"].astype(str)
    yard["expected_resource"] = yard["target"].replace({"VC": "HO2", "MT": "LL"})
    yard["matches_target"] = yard["resource"].eq(yard["expected_resource"])
    yard["case_id"] = yard["case:concept:name"].astype(str)
    yard["start"] = pd.to_datetime(yard["start:timestamp"])
    yard = yard.sort_values(["case_id", "start"], kind="stable")
    yard["previous_resource"] = yard.groupby("case_id")["resource"].shift()
    yard["resource_changed"] = (
        yard["previous_resource"].notna()
        & yard["resource"].ne(yard["previous_resource"])
    )

    print(f"yard_events={len(yard)}")
    print(f"target_area_match_share={yard['matches_target'].mean():.6f}")
    print(f"target_area_mismatches={(~yard['matches_target']).sum()}")
    print(f"successive_yard_legs={yard['previous_resource'].notna().sum()}")
    print(f"successive_resource_changes={yard['resource_changed'].sum()}")
    print("\nMost common mismatches")
    print(
        yard.loc[~yard["matches_target"]]
        .groupby(["target", "expected_resource", "resource"])
        .size()
        .sort_values(ascending=False)
        .head(20)
        .to_string()
    )


if __name__ == "__main__":
    main()
