import pandas as pd


def create_yard_aggregated_s2(
    input_file: str,
    output_file: str = "yard_aggregated_s2.csv"
):

    # =====================================================
    # LOAD
    # =====================================================

    yard = pd.read_csv(
        input_file,
        sep=";"
    )

    yard["ZEITPUNKT"] = pd.to_datetime(
        yard["ZEITPUNKT"]
    )

    numeric_cols = [
        "KAP_TEU_THEORETISCH",
        "BEL_TEU"
    ]

    for col in numeric_cols:
        yard[col] = (
            yard[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )

    result = []

    # =====================================================
    # TOTAL TERMINAL
    # =====================================================

    total = (
        yard.groupby("ZEITPUNKT")
        .agg(
            KAP_TEU_THEORETISCH=("KAP_TEU_THEORETISCH", "sum"),
            BEL_TEU=("BEL_TEU", "sum")
        )
        .reset_index()
    )

    total["AREA"] = "total"

    result.append(total)

    # =====================================================
    # BLOCK TYPES
    # =====================================================

    for area in ["VC", "MT", "RMG"]:

        tmp = (
            yard[
                yard["BLOCKTYP"] == area
            ]
            .groupby("ZEITPUNKT")
            .agg(
                KAP_TEU_THEORETISCH=("KAP_TEU_THEORETISCH", "sum"),
                BEL_TEU=("BEL_TEU", "sum")
            )
            .reset_index()
        )

        tmp["AREA"] = area

        result.append(tmp)

    # =====================================================
    # INDIVIDUAL BLOCKS
    # B06 → T06
    # ...
    # B27 → T27
    # =====================================================

    for i in range(6, 28):

        block = f"B{i:02d}"

        tmp = yard[
            yard["BLOCKNAME"] == block
        ].copy()

        if len(tmp) == 0:
            continue

        tmp = (
            tmp.groupby("ZEITPUNKT")
            .agg(
                KAP_TEU_THEORETISCH=("KAP_TEU_THEORETISCH", "sum"),
                BEL_TEU=("BEL_TEU", "sum")
            )
            .reset_index()
        )

        tmp["AREA"] = f"T{i:02d}"

        result.append(tmp)

    # =====================================================
    # COMBINE
    # =====================================================

    final = pd.concat(
        result,
        ignore_index=True
    )

    final["AUSLASTUNG_%"] = (
        final["BEL_TEU"]
        / final["KAP_TEU_THEORETISCH"]
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

    final = final[
        [
            "ZEITPUNKT",
            "AREA",
            "KAP_TEU_THEORETISCH",
            "BEL_TEU",
            "AUSLASTUNG_%"
        ]
    ]

    final = final.sort_values(
        ["ZEITPUNKT", "AREA"]
    )

    final.to_csv(
        output_file,
        sep=";",
        index=False
    )

    print(
        f"Created: {output_file}"
    )

    print(
        f"Rows: {len(final):,}"
    )

    return final



yard_s2 = create_yard_aggregated_s2(
    "data/raw/CTB/yard_aggregated.csv"
)
yard_s2 = yard_s2.round(2)

yard_s2.to_csv("data/interim/CTB/yard_aggregated_s2.csv", index=False)