import pandas as pd

INPUT_FILE = (
    "data/interim/CTB/Report_week_0304_mapped.csv"
)

OUTPUT_FILE = (
    "data/interim/CTB/Report_week_0304_mapped_flowtype.csv"
)

print("Loading report...")

df = pd.read_csv(
    INPUT_FILE,
    sep=";"
)

df.columns = df.columns.str.strip()

# ======================================================
# FLOW TYPE
# ======================================================

def derive_flow_type(row):

    inbound = str(
        row["UnitInboundCarrierId"]
    ).strip()

    outbound = str(
        row["UnitOutboundCarrierId"]
    ).strip()

    inbound_missing = (
        inbound in ["", "nan", "None"]
    )

    outbound_missing = (
        outbound in ["", "nan", "None"]
    )

    if (
        not inbound_missing
        and outbound_missing
    ):
        return "delivery"

    if (
        inbound_missing
        and not outbound_missing
    ):
        return "receive"

    if (
        not inbound_missing
        and not outbound_missing
    ):
        return "dual"

    return "unknown"


df["FlowType"] = (
    df.apply(
        derive_flow_type,
        axis=1
    )
)

print(
    df["FlowType"]
    .value_counts(dropna=False)
)

df.to_csv(
    OUTPUT_FILE,
    sep=";",
    index=False
)

print(
    f"Saved to: {OUTPUT_FILE}"
)