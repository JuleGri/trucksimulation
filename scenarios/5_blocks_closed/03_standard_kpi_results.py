import pandas as pd

# =====================================================
# CONFIG
# =====================================================

SCENARIO_NAME = "5 Blocks Closed"

INPUT_FILE = (
    "scenarios/5_blocks_closed/scenario_B_5_blocks_closed_sim_log.csv"
)

OUTPUT_FILE = (
    "scenarios/5_blocks_closed/Scenario_B_kpis.csv"
)

# =====================================================
# LOAD
# =====================================================

sim_log = pd.read_csv(
    INPUT_FILE
)

# =====================================================
# TIMESTAMPS
# =====================================================

for col in [

    "enabled:timestamp",

    "start:timestamp",

    "time:timestamp"

]:

    sim_log[col] = pd.to_datetime(
        sim_log[col],
        errors="coerce"
    )
print("scen A - Cases:", sim_log["case:concept:name"] .nunique() )
print("scen A - Rows:",len(sim_log))
# =====================================================
# RMG EVENTS
# =====================================================

rmg_log = sim_log[

    sim_log["concept:name"].isin(

        [

            "RMG_receive",

            "RMG_delivery",

            "RMG_mixed"

        ]

    )

].copy()

print("rmg log activity counts:", rmg_log["concept:name"].value_counts())
# =====================================================
# WAITING TIME (RMG ONLY)
# =====================================================

rmg_waiting_time = (

    rmg_log["start:timestamp"]

    -

    rmg_log["enabled:timestamp"]

).dt.total_seconds() / 60

print("RMG waiting time:", rmg_waiting_time.describe())
# =====================================================
# SERVICE TIME (RMG ONLY)
# =====================================================

rmg_service_time = (

    rmg_log["time:timestamp"]

    -

    rmg_log["start:timestamp"]

).dt.total_seconds() / 60

print("RMG Service Time:", rmg_service_time.describe())
# =====================================================
# TURNAROUND TIME (COMPLETE PROCESS)
# =====================================================

turnaround_time = (

    sim_log

    .groupby(
        "case:concept:name"
    )

    .agg(

        start=(
            "start:timestamp",
            "min"
        ),

        end=(
            "time:timestamp",
            "max"
        )

    )

)

turnaround_time = (

    turnaround_time["end"]

    -

    turnaround_time["start"]

).dt.total_seconds() / 60

# =====================================================
# KPI TABLE
# =====================================================

kpis = {

    "scenario":
        SCENARIO_NAME,

    "cases":
        sim_log[
            "case:concept:name"
        ].nunique(),

    "waiting_mean":
        rmg_waiting_time.mean(),

    "service_mean":
        rmg_service_time.mean(),

    "turnaround_mean":
        turnaround_time.mean(),

    "waiting_median":
        rmg_waiting_time.median(),

    "service_median":
        rmg_service_time.median(),

    "turnaround_median":
        turnaround_time.median()

}

kpis_df = pd.DataFrame(
    [kpis]
)

print(kpis_df)

# =====================================================
# SAVE
# =====================================================

kpis_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"Saved: {OUTPUT_FILE}"
)