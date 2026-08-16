import pandas as pd
import re

# ==========================================================
# CONFIGURATION
# ==========================================================

DELIVERIES_FILE = "data/raw/CTB/deliveries_0304.csv"
RECEIVES_FILE = "data/raw/CTB/receives_0304.csv"

OUTPUT_FILE = (
    "data/raw/CTB/container_master_0304.csv"
)

# ==========================================================
# HELPERS
# ==========================================================

def clean_columns(df):
    df.columns = [
        re.sub(r"<.*?>", "", c).strip()
        for c in df.columns
    ]
    return df


def parse_timestamp(series):
    return pd.to_datetime(
        series.astype(str).str.strip(),
        format="%d.%m.%y %H:%M",
        errors="coerce"
    )


def is_reefer(iso_code):

    if pd.isna(iso_code):
        return 0

    return int("R" in str(iso_code).upper())


def is_hazardous(imco):

    if pd.isna(imco):
        return 0

    return int(str(imco).strip() != "")


def get_container_size(iso_code):

    try:
        return int(str(iso_code)[:2])

    except:
        return None


def compute_complexity(row):

    score = 0

    # full container
    if str(row["Leer/Voll"]).strip().lower() == "voll":
        score += 1

    # flow_type
    if row["flow_type"] == "RECEIVE":
        score += 1

    # reefer
    if row["is_reefer"] == 1:
        score += 2

    # hazardous
    if row["is_hazardous"] == 1:
        score += 2

    
    """ if row["CTR_TYP_LANG"] == "Open Top":
        score += 1
    if row["CTR_TYP_LANG"] == "Platform":
        score += 1
    if row["CTR_TYP_LANG"] == "Bulk":
        score += 1 """

    #Container Typ Einordnung, falls voll - oder auch leer Plattformen etc.?    
    interaction_weights = {

        

        ("Open Top", "voll"): 3,

        ("Platform", "voll"): 3,

        
    }

    key = (
        row["CTR_TYP_LANG"],
        str(row["Leer/Voll"])
            .strip()
            .lower()
    )

    score += interaction_weights.get(
        key,
        0
    )
    # Long dwell time for full containers only - Ws sind hier die threshholds?

    if (
        str(row["Leer/Voll"]).strip().lower() == "voll"
        and
        str(row["flow_type"]).strip().lower() == "RECEIVE"
        and
        pd.notna(row["dwell_hours"])
    ):

        if row["dwell_hours"] > 72:
            score += 1

        if row["dwell_hours"] > 168:   # 7 days
            score += 2

        if row["dwell_hours"] > 336:   # 14 days
            score += 3

    # larger than 40 feet containers?
    if pd.notna(row["container_size"]):

        if row["container_size"] >= 45:
            score += 1
    

    return score


# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading deliveries...")

deliveries = pd.read_csv(
    DELIVERIES_FILE,
    sep=";",
    dtype=str
)

print("Loading receives...")

receives = pd.read_csv(
    RECEIVES_FILE,
    sep=";",
    dtype=str
)

deliveries = clean_columns(deliveries)
receives = clean_columns(receives)

# ==========================================================
# TIMESTAMP PARSING
# ==========================================================

for df in [deliveries, receives]:

    df["ZEIT_IN"] = parse_timestamp(
        df["ZEIT_IN"]
    )

    df["ZEIT_OUT"] = parse_timestamp(
        df["ZEIT_OUT"]
    )

# ==========================================================
# CONTAINER DWELL TIME
# ==========================================================

for df in [deliveries, receives]:

    df["dwell_hours"] = (
        (
            df["ZEIT_OUT"]
            -
            df["ZEIT_IN"]
        ).dt.total_seconds() / 3600
    ).round(2)

    df["dwell_days"] = (
        (
            df["ZEIT_OUT"]
            -
            df["ZEIT_IN"]
        ).dt.total_seconds() / (24 * 3600)
    ).round(2)
# ==========================================================
# FLOW TYPE
# ==========================================================

deliveries["flow_type"] = "DELIVERY"
receives["flow_type"] = "RECEIVE"

# ==========================================================
# MATCHING TIMESTAMP
# ==========================================================

deliveries["reference_timestamp"] = (
    deliveries["ZEIT_IN"]
)

receives["reference_timestamp"] = (
    receives["ZEIT_OUT"]
)

# ==========================================================
# COMBINE
# ==========================================================

containers = pd.concat(
    [deliveries, receives],
    ignore_index=True
)

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

containers["is_reefer"] = (
    containers["ISC"]
    .apply(is_reefer)
)

containers["is_hazardous"] = (
    containers["IMCO"]
    .apply(is_hazardous)
)

containers["is_full"] = (
    containers["Leer/Voll"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("voll")
    .astype(int)
)

containers["container_size"] = (
    containers["ISC"]
    .apply(get_container_size)
)

# ==========================================================
# CONTAINER COMPLEXITY
# ==========================================================

containers["container_complexity"] = (
    containers.apply(
        compute_complexity,
        axis=1
    )
)

# ==========================================================
# FINAL DATASET
# ==========================================================

containers_master = containers[
    [
        "EICO",
        "flow_type",

        "reference_timestamp",

        "ZEIT_IN",
        "ZEIT_OUT",
        
        "dwell_hours",
        "dwell_days",

        "ISC",
        "container_size",

        "Leer/Voll",
        "is_full",

        "IMCO",
        "is_hazardous",
        "is_reefer",

        "Inbound_Carrier",
        "Outbound_Carrier",

        "CTR_TYP_LANG",
        "ORT",
        "REED",

        "container_complexity"
    ]
].copy()

containers_master = containers_master.rename(
    columns={
        "EICO": "container_id",
        "ISC": "container_type",
        "Leer/Voll": "full_empty",
        "IMCO": "imco"
    }
)

# ==========================================================
# SORT
# ==========================================================

containers_master = containers_master.sort_values(
    "reference_timestamp"
)

# ==========================================================
# SAVE
# ==========================================================

containers_master.to_csv(
    OUTPUT_FILE,
    sep=";",
    index=False
)

# ==========================================================
# REPORT
# ==========================================================

print("\n" + "=" * 60)
print("CONTAINER MASTER DATASET REPORT")
print("=" * 60)

print(f"Total containers:        {len(containers_master):,}")

print(
    f"Deliveries:              "
    f"{(containers_master['flow_type'] == 'DELIVERY').sum():,}"
)

print(
    f"Receives:                "
    f"{(containers_master['flow_type'] == 'RECEIVE').sum():,}"
)

print(
    f"Full containers:         "
    f"{containers_master['is_full'].sum():,}"
)

print(
    f"Reefers:                "
    f"{containers_master['is_reefer'].sum():,}"
)

print(
    f"Hazardous:              "
    f"{containers_master['is_hazardous'].sum():,}"
)

print("\nContainer Complexity Distribution")

print(
    containers_master["container_complexity"]
    .value_counts()
    .sort_index()
)

print("\nOutput File:")
print(OUTPUT_FILE)

print(
    containers_master
    .duplicated(
        subset=[
            "container_id",
            "flow_type"
        ]
    )
    .sum()
)

print("=" * 60)
print("Container master dataset created successfully.")
print("=" * 60)