from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(
    r"C:\Users\Jule\Documents\Master\Masterthesis\trucksimulation"
    r"\data\raw\CTB\ctb_fahrplan_lkw_0304.csv"
)


def main() -> None:
    columns = [
        "FAHRPLAN_UID",
        "ANLAUFPUNKT_SEQUENZNUMMER",
        "ANLAUFPUNKT_HALTESTELLE",
        "FP_DAUER_GATEIN_ERSTER_ALP_SEK",
        "ALP_DAUER_NAECHSTER_ALP_SEK",
        "FP_DAUER_LETZT_ALP_GATEOUT_SEK",
    ]
    frame = pd.read_csv(SOURCE, sep=";", usecols=columns)
    frame["sequence"] = pd.to_numeric(
        frame["ANLAUFPUNKT_SEQUENZNUMMER"].astype(str).str.replace(",", "."),
        errors="coerce",
    )
    frame = frame.sort_values(
        ["FAHRPLAN_UID", "sequence", "ANLAUFPUNKT_SEQUENZNUMMER"],
        kind="stable",
    )
    grouped_cases = frame.groupby("FAHRPLAN_UID", sort=False)
    first = grouped_cases.head(1).copy()
    last = grouped_cases.tail(1).copy()
    frame["next_stop"] = grouped_cases["ANLAUFPUNKT_HALTESTELLE"].shift(-1)

    inbound = pd.DataFrame(
        {
            "origin": "Gate In",
            "destination": first["ANLAUFPUNKT_HALTESTELLE"].astype(str),
            "duration_sec": pd.to_numeric(
                first["FP_DAUER_GATEIN_ERSTER_ALP_SEK"], errors="coerce"
            ),
            "leg_type": "inbound",
        }
    )
    internal = pd.DataFrame(
        {
            "origin": frame["ANLAUFPUNKT_HALTESTELLE"].astype(str),
            "destination": frame["next_stop"].astype("string"),
            "duration_sec": pd.to_numeric(
                frame["ALP_DAUER_NAECHSTER_ALP_SEK"], errors="coerce"
            ),
            "leg_type": "yard_to_yard",
        }
    )
    outbound = pd.DataFrame(
        {
            "origin": last["ANLAUFPUNKT_HALTESTELLE"].astype(str),
            "destination": "Gate Out",
            "duration_sec": pd.to_numeric(
                last["FP_DAUER_LETZT_ALP_GATEOUT_SEK"], errors="coerce"
            ),
            "leg_type": "outbound",
        }
    )
    transitions = pd.concat([inbound, internal, outbound], ignore_index=True)
    transitions = transitions[
        transitions["destination"].notna()
        & transitions["duration_sec"].notna()
        & transitions["duration_sec"].gt(0)
    ].copy()
    grouped = (
        transitions.groupby(["leg_type", "origin", "destination"], as_index=False)
        .agg(
            samples=("duration_sec", "size"),
            p05_sec=("duration_sec", lambda values: np.percentile(values, 5)),
            median_sec=("duration_sec", "median"),
            mean_sec=("duration_sec", "mean"),
        )
    )
    eligible = grouped[grouped["samples"] >= 20].copy()

    print("Raw positive transit observations by leg type")
    print(transitions.groupby("leg_type")["duration_sec"].agg(["size", "min", "median", "mean", "max"]).to_string())
    print("\nPooled duration percentiles by leg type")
    print(
        transitions.groupby("leg_type")["duration_sec"]
        .quantile([0.05, 0.10, 0.25, 0.50])
        .unstack()
        .rename(columns={0.05: "p05", 0.10: "p10", 0.25: "p25", 0.50: "p50"})
        .to_string()
    )
    print("\nRoute pairs with at least 20 observations")
    print(eligible.groupby("leg_type").agg(pairs=("origin", "size"), observations=("samples", "sum")).to_string())
    print("\nP05 seconds across eligible route pairs")
    print(eligible.groupby("leg_type")["p05_sec"].agg(["min", "median", "mean", "max"]).to_string())
    print("\nYard-to-yard eligible route pairs")
    print(eligible[eligible["leg_type"] == "yard_to_yard"].sort_values("samples", ascending=False).to_string(index=False))
    print("\nMost-observed yard-to-yard pairs (including sparse pairs)")
    print(
        grouped[grouped["leg_type"] == "yard_to_yard"]
        .sort_values("samples", ascending=False)
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
