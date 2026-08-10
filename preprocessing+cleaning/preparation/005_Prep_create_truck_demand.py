import pandas as pd


# ==========================================================
# AREA MAPPING
# ==========================================================

def determine_area(stop):

    if pd.isna(stop):
        return None

    stop = str(stop).strip()

    if stop == "LL":
        return "MT"

    if stop == "HO2":
        return "VC"

    if stop.startswith("T"):
        return stop

    return None


# ==========================================================
# BUILD TRUCK DEMAND TABLE
# ==========================================================

def create_truck_demand_s4(
    fahrplan_path: str,
    output_path: str = "truck_demand_s4.csv",
    frequency: str = "1h"
):

    print("Loading Fahrplan...")

    df = pd.read_csv(
        fahrplan_path,
        sep=";"
    )

    # ======================================================
    # DATETIME
    # ======================================================

    df["FP_ZEITPUNKT_ERSTELLUNG"] = pd.to_datetime(
        df["FP_ZEITPUNKT_ERSTELLUNG"],
        format="%d.%m.%Y %H:%M",
        errors="coerce"
    )

    df["ALP_ZEITPUNKT_BEREITMELDUNG"] = pd.to_datetime(
        df["ALP_ZEITPUNKT_BEREITMELDUNG"],
        format="%d.%m.%Y %H:%M",
        errors="coerce"
    )

    results = []

    # ======================================================
    # TOTAL TERMINAL DEMAND
    # based on FP creation timestamp
    # ======================================================

    total = (
        df[
            ["FAHRPLAN_UID",
             "FP_ZEITPUNKT_ERSTELLUNG"]
        ]
        .drop_duplicates(
            subset=["FAHRPLAN_UID"]
        )
        .copy()
    )

    total["cnt"] = 1

    total = (
        total
        .set_index("FP_ZEITPUNKT_ERSTELLUNG")
        .resample(frequency)
        .sum(numeric_only=True)
    )

    total["TRUCK_DEMAND_LAST_1H"] = (
        total["cnt"]
        .rolling("1h")
        .sum()
    )

    total = (
    total[["TRUCK_DEMAND_LAST_1H"]]
    .reset_index()
    )

    total = total.rename(
        columns={
            "FP_ZEITPUNKT_ERSTELLUNG":
                "ZEITPUNKT"
        }
    )

    total["AREA"] = "total"

    results.append(total)

    # ======================================================
    # AREA DEMAND
    # based on area arrival time
    # ======================================================

    area_df = df.copy()

    area_df["AREA"] = (
        area_df["ANLAUFPUNKT_HALTESTELLE"]
        .apply(determine_area)
    )

    area_df = area_df.dropna(
        subset=[
            "AREA",
            "ALP_ZEITPUNKT_BEREITMELDUNG"
        ]
    )

    area_df["cnt"] = 1

    all_areas = sorted(
        area_df["AREA"].unique()
    )

    print("Areas found:")
    print(all_areas)

    for area in all_areas:

        tmp = area_df[
            area_df["AREA"] == area
        ].copy()

        tmp = (
            tmp
            .set_index(
                "ALP_ZEITPUNKT_BEREITMELDUNG"
            )
            .resample(frequency)
            .sum(numeric_only=True)
        )

        tmp["TRUCK_DEMAND_LAST_1H"] = (
            tmp["cnt"]
            .rolling("1h")
            .sum()
        )

        tmp = (
        tmp[["TRUCK_DEMAND_LAST_1H"]]
        .reset_index()
    )

        tmp = tmp.rename(
            columns={
                "ALP_ZEITPUNKT_BEREITMELDUNG":
                    "ZEITPUNKT"
            }
        )

        tmp["AREA"] = area

        results.append(tmp)

    # ======================================================
    # RMG AGGREGATE
    # all T-blocks together
    # ======================================================

    rmg = area_df[
        area_df["AREA"]
        .str.startswith("T")
    ].copy()

    rmg = (
        rmg
        .set_index(
            "ALP_ZEITPUNKT_BEREITMELDUNG"
        )
        .resample(frequency)
        .sum(numeric_only=True)
    )

    rmg["TRUCK_DEMAND_LAST_1H"] = (
        rmg["cnt"]
        .rolling("1h")
        .sum()
    )

    rmg = (
    rmg[["TRUCK_DEMAND_LAST_1H"]]
    .reset_index()
    )

    rmg = rmg.rename(
        columns={
            "ALP_ZEITPUNKT_BEREITMELDUNG":
                "ZEITPUNKT"
        }
    )

    rmg["AREA"] = "RMG"

    results.append(rmg)

    # ======================================================
    # COMBINE
    # ======================================================

    final = pd.concat(
        results,
        ignore_index=True
    )

    final = final.rename(
        columns={
            "FP_ZEITPUNKT_ERSTELLUNG":
                "ZEITPUNKT",
            "ALP_ZEITPUNKT_BEREITMELDUNG":
                "ZEITPUNKT"
        }
    )

    final["TRUCK_DEMAND_LAST_1H"] = (
        final["TRUCK_DEMAND_LAST_1H"]
        .fillna(0)
        .round(2)
    )

    final = final.sort_values(
        ["ZEITPUNKT", "AREA"]
    )

    final.to_csv(
        output_path,
        index=False
    )

    print(
        f"Created: {output_path}"
    )

    print(
        f"Rows: {len(final):,}"
    )

    return final


# ==========================================================
# EXECUTION
# ==========================================================

if __name__ == "__main__":

    create_truck_demand_s4(
        fahrplan_path="data/raw/CTB/ctb_fahrplan_lkw_0304.csv",
        output_path="data/interim/CTB/truck_demand_aggregated.csv"
    )