import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================

INPUT_FILE = (
    "data/interim/CTB/"
    "s3_its_fahrplan_container_details.csv"
)

OUTPUT_FILE = (
    "data/interim/CTB/"
    "s4_its_fahrplan_with_case_features.csv"
)

# ==========================================================
# LOAD
# ==========================================================

print("Loading schedule...")

df = pd.read_csv(
    INPUT_FILE,
    sep=";"
)

df.columns = df.columns.str.strip()

# ==========================================================
# CLEAN TYPES
# ==========================================================

for col in [
    "is_hazardous",
    "is_reefer",
    "is_full",
    "ANLAUFPUNKT_ANZAHL_CONTAINER"
]:
    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

""" # ==========================================================
# BUILD STOP LEVEL TABLE
# ==========================================================

print("Building stop-level table...")

stop_rows = []

for (
    fahrplan_uid,
    anlaufpunkt_uid
), grp in df.groupby(
    [
        "FAHRPLAN_UID",
        "ANLAUFPUNKT_UID"
    ],
    dropna=False
):

    base = grp.iloc[0].copy()

    flows = set(
        grp["flow_type_master"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
    )

    stop_name = str(
        base["ANLAUFPUNKT_HALTESTELLE"]
    )

    if flows == {"DELIVERY"}:

        stop_flow = (
            stop_name
            + "_delivery"
        )

    elif flows == {"RECEIVE"}:

        stop_flow = (
            stop_name
            + "_receive"
        )

    elif (
        "DELIVERY" in flows
        and "RECEIVE" in flows
    ):

        stop_flow = (
            stop_name
            + "_mixed"
        )

    else:

        stop_flow = (
            stop_name
            + "_unknown"
        )

    base["stop_flow"] = stop_flow

    base["is_hazardous"] = (
        grp["is_hazardous"]
        .fillna(0)
        .max()
    )

    base["is_reefer"] = (
        grp["is_reefer"]
        .fillna(0)
        .max()
    )

    base["is_full"] = (
        grp["is_full"]
        .fillna(0)
        .mean()
    )

    base["n_containers_stop"] = (
        grp["container_id"]
        .dropna()
        .nunique()
    )

    stop_rows.append(base)

stop_df = pd.DataFrame(stop_rows) """

# ==========================================================
# BUILD STOP LEVEL TABLE (VECTORIZED)
# ==========================================================

print("Building stop-level table...")

# ----------------------------------------------------------
# STOP FLOW PER GROUP
# ----------------------------------------------------------

flow_per_stop = (
    df.groupby(
        ["FAHRPLAN_UID", "ANLAUFPUNKT_UID"]
    )["flow_type_master"]
    .apply(
        lambda x: set(
            x.dropna()
             .astype(str)
             .str.upper()
        )
    )
)

def derive_stop_flow(flows, stop_name):

    if flows == {"DELIVERY"}:
        return f"{stop_name}_delivery"

    elif flows == {"RECEIVE"}:
        return f"{stop_name}_receive"

    elif (
        "DELIVERY" in flows
        and "RECEIVE" in flows
    ):
        return f"{stop_name}_mixed"

    else:
        return f"{stop_name}_unknown"

# ----------------------------------------------------------
# AGGREGATION
# ----------------------------------------------------------

stop_df = (
    df.groupby(
        [
            "FAHRPLAN_UID",
            "ANLAUFPUNKT_UID"
        ],
        dropna=False,
        as_index=False
    )
    .agg({

        # erste Fahrplaninfos behalten

        "FAHRPLAN_ANLAUFPUNKT_UID": "first",
        "FAHRPLAN_ID": "first",
        "LKW_KENNZEICHEN": "first",

        "ANLAUFPUNKT_SEQUENZNUMMER": "first",
        "ALP_IST_ERSTER_ALP": "first",
        "ALP_IST_LETZTER_ALP": "first",

        "ANLAUFPUNKT_NAME": "first",
        "ANLAUFPUNKT_HALTESTELLE": "first",
        "ANLAUFPUNKT_SPUR": "first",
        "ANLAUFPUNKT_LAGERBEREICH": "first",

        "FP_ZEITPUNKT_ERSTELLUNG": "first",
        "ALP_ZEITPUNKT_BEREITMELDUNG": "first",
        "ALP_ZEITPUNKT_ERFUELLUNG": "first",
        "FP_ZEITPUNKT_GATEOUT": "first",

        "FP_DAUER_GATEIN_ERSTER_ALP_SEK": "first",
        "ALP_DAUER_ABFERTIGUNG_SEK": "first",
        "ALP_DAUER_NAECHSTER_ALP_SEK": "first",
        "FP_DAUER_LETZT_ALP_GATEOUT_SEK": "first",
        "FP_DAUER_GATEIN_GATEOUT_SEK": "first",

        "LETZTE_AKTUALISIERUNG": "first",

        "FAHRPLAN_HALTESTELLEN_LISTE": "first",
        "ANLAUFPUNKT_AUFTRAG_LISTE": "first",
        "ANLAUFPUNKT_ANZAHL_CONTAINER": "first",

        "TruckVisitGkey": "first",

        # Stop-Level Aggregationen

        "is_hazardous": "max",
        "is_reefer": "max",
        "is_full": "mean",

        "container_id": "nunique"
    })
)

stop_df = stop_df.rename(
    columns={
        "container_id":
            "n_containers_stop"
    }
)

# ----------------------------------------------------------
# STOP FLOW
# ----------------------------------------------------------
print(df["ANLAUFPUNKT_UID"].isna().sum())
print(stop_df["ANLAUFPUNKT_UID"].isna().sum())

stop_df["flow_set"] = (
    stop_df[
        [
            "FAHRPLAN_UID",
            "ANLAUFPUNKT_UID"
        ]
    ]
    .apply(
        lambda r: flow_per_stop.get(
            (
                r["FAHRPLAN_UID"],
                r["ANLAUFPUNKT_UID"]
            ),
            set()
        ),
        axis=1
    )
)

stop_df["stop_flow"] = stop_df.apply(
    lambda r: derive_stop_flow(
        r["flow_set"],
        str(
            r["ANLAUFPUNKT_HALTESTELLE"]
        )
    ),
    axis=1
)

stop_df = stop_df.drop(
    columns="flow_set"
)

print(
    f"Stop-level rows: "
    f"{len(stop_df):,}"
)
################################################# Bis hier alles löschen und das Kommentar wieder einbauen
print(
    f"Stop-level rows: "
    f"{len(stop_df):,}"
)

# ==========================================================
# PRECOMPUTE CASE STATS
# ==========================================================

print("Precomputing case statistics...")

case_stats = {}

for uid, grp in df.groupby("FAHRPLAN_UID"):

    n_deliveries = (
        grp["flow_type_master"]
        .astype(str)
        .str.upper()
        .eq("DELIVERY")
        .sum()
    )

    n_receives = (
        grp["flow_type_master"]
        .astype(str)
        .str.upper()
        .eq("RECEIVE")
        .sum()
    )

    unique_container_ids = (
        grp["container_id"]
        .dropna()
        .nunique()
    )

    declared_container_count = (
        grp["ANLAUFPUNKT_ANZAHL_CONTAINER"]
        .fillna(0)
        .sum()
    )

    has_hazardous = int(
        grp["is_hazardous"]
        .fillna(0)
        .max()
    )

    has_reefer = int(
        grp["is_reefer"]
        .fillna(0)
        .max()
    )

    full_ratio = float(
        grp["is_full"]
        .fillna(0)
        .mean()
    )

    container_complexity_sum = (
    grp["container_complexity"]
    .fillna(0)
    .sum()
    )

    container_complexity_mean = (
        grp["container_complexity"]
        .fillna(0)
        .mean()
    )


    case_stats[uid] = {

        "n_deliveries":
            int(n_deliveries),

        "n_receives":
            int(n_receives),

        "unique_container_ids":
            int(unique_container_ids),

        "declared_container_count":
            float(declared_container_count),

        "has_hazardous":
            has_hazardous,

        "has_reefer":
            has_reefer,

        "full_ratio":
            full_ratio,
        
        "container_complexity_sum":
            float(container_complexity_sum),

        "container_complexity_mean":
            float(container_complexity_mean),
    }

# ==========================================================
# CASE FEATURES
# ==========================================================

print("Computing case features...")

features = []

container_count_mismatch = 0
cases_without_container_ids = 0

for uid, grp in stop_df.groupby(
    "FAHRPLAN_UID"
):

    stats = case_stats[uid]

    n_deliveries = (
        stats["n_deliveries"]
    )

    n_receives = (
        stats["n_receives"]
    )

    if (
        n_deliveries > 0
        and n_receives > 0
    ):

        process_flow_type = "mixed"

    elif n_deliveries > 0:

        process_flow_type = "delivery"

    elif n_receives > 0:

        process_flow_type = "receive"

    else:

        process_flow_type = "unknown"

    unique_container_ids = (
        stats["unique_container_ids"]
    )

    declared_container_count = (
        stats["declared_container_count"]
    )

    if unique_container_ids == 0:

        cases_without_container_ids += 1

        n_containers = int(
            declared_container_count
        )

    else:

        n_containers = int(
            unique_container_ids
        )

    if (
        unique_container_ids > 0
        and unique_container_ids
        != declared_container_count
    ):
        container_count_mismatch += 1

    n_stops = (
        grp["ANLAUFPUNKT_HALTESTELLE"]
        .nunique()
    )

    has_hazardous = (
        stats["has_hazardous"]
    )

    has_reefer = (
        stats["has_reefer"]
    )

    full_ratio = (
        stats["full_ratio"]
    )

    container_complexity_sum = (
        stats["container_complexity_sum"]
    )

    container_complexity_mean = (
        stats["container_complexity_mean"]
    )

    n_mixed_stops = (
        grp["stop_flow"]
        .astype(str)
        .str.endswith("_mixed")
        .sum()
    )
    is_mixed = int(
        process_flow_type == "mixed"
    )

    # -------------------------------------
    # VISIT SIZE
    # -------------------------------------

    visit_size_score = (
        n_stops
        + n_containers
    )

    # -------------------------------------
    # CONTAINER COMPLEXITY
    # -------------------------------------

    visit_container_score = (
        container_complexity_sum
    )

    # -------------------------------------
    # OPERATIONAL FLOW COMPLEXITY
    # -------------------------------------

    visit_flow_score = (

        2 * is_mixed

        +

        3 * n_mixed_stops
    )

    # -------------------------------------
    # TOTAL VISIT COMPLEXITY
    # -------------------------------------

    visit_complexity = (

        visit_size_score

        +

        visit_container_score

        +

        visit_flow_score
    )

    features.append({

        "FAHRPLAN_UID":
            uid,

        "process_flow_type":
            process_flow_type,

        "n_containers":
            n_containers,

        "n_stops":
            n_stops,

        "n_deliveries":
            n_deliveries,

        "n_receives":
            n_receives,

        "has_hazardous":
            has_hazardous,

        "has_reefer":
            has_reefer,

        "full_ratio":
            round(full_ratio, 3),

        "container_complexity_mean":
            round(
                container_complexity_mean,
                2
            ),

        "visit_size_score":
            int(visit_size_score),

        "visit_container_score":
            round(
                visit_container_score,
                2
            ),

        "visit_flow_score":
            int(visit_flow_score),
        
        "visit_complexity":
            int(visit_complexity),
        
        "container_complexity_sum":
            round(
                container_complexity_sum,
                2
            )
    })

# ==========================================================
# MERGE BACK
# ==========================================================

feature_df = pd.DataFrame(features)

stop_df = stop_df.merge(
    feature_df,
    on="FAHRPLAN_UID",
    how="left"
)

# ==========================================================
# QUALITY REPORT
# ==========================================================

print("\n" + "=" * 70)
print("S3 -> S4 AGGREGATION REPORT")
print("=" * 70)

print(
    f"Original rows: {len(df):,}"
)

print(
    f"Stop-level rows: {len(stop_df):,}"
)

print(
    f"Truck visits: "
    f"{stop_df['FAHRPLAN_UID'].nunique():,}"
)

print()

print(
    f"Cases without container ids: "
    f"{cases_without_container_ids:,}"
)

print(
    f"Container count mismatches: "
    f"{container_count_mismatch:,}"
)

print()

print(
    stop_df["process_flow_type"]
    .value_counts(dropna=False)
)

print()

print(
    stop_df["stop_flow"]
    .value_counts()
    .head(20)
)

print("=" * 70)

# ==========================================================
# SAVE
# ==========================================================

stop_df.to_csv(
    OUTPUT_FILE,
    sep=";",
    index=False
)

print(
    f"\nSaved:\n{OUTPUT_FILE}"
)