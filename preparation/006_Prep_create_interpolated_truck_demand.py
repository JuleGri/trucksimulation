import pandas as pd


def create_interpolated_truck_demand(
    input_file,
    output_file="truck_demand_s4_interpolated_1min.csv"
):

    df = pd.read_csv(input_file)

    df["ZEITPUNKT"] = pd.to_datetime(
        df["ZEITPUNKT"]
    )

    result = []

    for area in sorted(df["AREA"].unique()):

        tmp = (
            df[df["AREA"] == area]
            .copy()
            .sort_values("ZEITPUNKT")
        )

        tmp = (
            tmp.set_index("ZEITPUNKT")
            .resample("1min")
            .asfreq()
        )

        tmp["TRUCK_DEMAND_LAST_1H"] = (
            tmp["TRUCK_DEMAND_LAST_1H"]
            .interpolate(method="time")
        )

        tmp["AREA"] = area

        tmp = tmp.reset_index()

        result.append(tmp)

    final = pd.concat(
        result,
        ignore_index=True
    )

    final["TRUCK_DEMAND_LAST_1H"] = (
        final["TRUCK_DEMAND_LAST_1H"]
        .round(2)
    )

    final.to_csv(
        output_file,
        index=False
    )

    print(
        f"Created {output_file}"
    )

    print(
        f"Rows: {len(final):,}"
    )

    return final


if __name__ == "__main__":

    create_interpolated_truck_demand(
        "data/interim/CTB/truck_demand_aggregated.csv",
        "data/interim/CTB/truck_demand_interpolated_1min.csv"
    )