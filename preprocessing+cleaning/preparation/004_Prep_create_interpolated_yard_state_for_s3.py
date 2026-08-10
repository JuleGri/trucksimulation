import pandas as pd


def create_interpolated_yard_state(
    input_file,
    output_file="data/interim/CTB/yard_aggregated_s2_interpolated_1min.csv"
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

        numeric_cols = [
            "KAP_TEU_THEORETISCH",
            "BEL_TEU",
            "AUSLASTUNG_%"
        ]

        tmp[numeric_cols] = (
            tmp[numeric_cols]
            .interpolate(method="time")
        )

        tmp["AREA"] = area

        tmp = tmp.reset_index()

        result.append(tmp)

    final = pd.concat(
        result,
        ignore_index=True
    )

    final["AUSLASTUNG_%"] = (
        final["AUSLASTUNG_%"]
        .round(4)
    )

    final["BEL_TEU"] = (
        final["BEL_TEU"]
        .round(2)
    )

    final["KAP_TEU_THEORETISCH"] = (
        final["KAP_TEU_THEORETISCH"]
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

    create_interpolated_yard_state(
        "data/raw/CTB/yard_aggregated_s2.csv",
        "data/interim/CTB/yard_aggregated_s2_interpolated_1min.csv"
    )
