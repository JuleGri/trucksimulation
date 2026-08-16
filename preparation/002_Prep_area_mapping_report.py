import pandas as pd

# ==========================================================
# CONFIGURATION
# ==========================================================

INPUT_FILE = "data/raw/CTB/Report_0304.csv"

OUTPUT_FILE = (
    "data/interim/CTB/Report_0304_mapped.csv"
)

# ==========================================================
# LOAD
# ==========================================================

print("Loading report...")

df = pd.read_csv(
    INPUT_FILE,
    sep=";"
)

# ==========================================================
# MAP EXCHANGE AREA
# ==========================================================

df["MappedExchangeArea"] = df[
    "TruckExchangeArea"
].astype(str)

df.loc[
    df["TruckExchangeArea"]
    .astype(str)
    .str.startswith("HP"),
    "MappedExchangeArea"
] = "LL"

# ==========================================================
# DROP UNUSED COLUMNS
# ==========================================================

columns_to_drop = [
    "TransactionGate",
    "TransactionNumber",
    "StageCreated",
    "StageChanged",
    "StageOrder",
    "StageEndTime",
    "TranNextStageId",
    "TranHadTrouble"
]

df = df.drop(
    columns=columns_to_drop,
    errors="ignore"
)

# ==========================================================
# REPORT
# ==========================================================

print("\n" + "=" * 60)
print("REPORT PREPARATION SUMMARY")
print("=" * 60)

print(
    f"Records: {len(df):,}"
)

print(
    "\nMapped Exchange Areas:"
)

print(
    df["MappedExchangeArea"]
    .value_counts()
    .head(20)
)

print("=" * 60)

# ==========================================================
# SAVE
# ==========================================================

df.to_csv(
    OUTPUT_FILE,
    sep=";",
    index=False
)

print(f"\nSaved to:")
print(OUTPUT_FILE)