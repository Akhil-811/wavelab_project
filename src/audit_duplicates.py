
from pathlib import Path

import pandas as pd

from src.data_utils import find_coordinate_columns, find_target_column

DATA_PATH = Path("data/raw/WEC_Perth_49.csv")


def main():
    df = pd.read_csv(DATA_PATH)

    coordinate_columns = find_coordinate_columns(df.columns)
    target_column = find_target_column(df)

    # Audit identical coordinate layouts.
    coordinates = df[coordinate_columns]

    exact_duplicates = int(df.duplicated().sum())
    repeated_layout_rows = int(coordinates.duplicated().sum())

    layout_target_counts = (
        df.groupby(coordinate_columns, dropna=False)[target_column]
        .nunique()
    )

    conflicting_layouts = layout_target_counts[layout_target_counts > 1]

    print("\nDUPLICATE LAYOUT AUDIT")
    print("=" * 45)
    print(f"Dataset: {DATA_PATH}")
    print(f"Total rows: {len(df):,}")
    print(f"Exact duplicate rows: {exact_duplicates:,}")
    print(f"Rows repeating an earlier layout: {repeated_layout_rows:,}")
    print(f"Unique coordinate layouts: {len(layout_target_counts):,}")
    print(f"Layouts with conflicting targets: {len(conflicting_layouts):,}")

    # Find the layout with the most distinct target values.
    if len(conflicting_layouts) > 0:
        worst_layout = conflicting_layouts.idxmax()

        mask = pd.Series(True, index=df.index)
        for column, value in zip(coordinate_columns, worst_layout):
            mask &= df[column].eq(value)

        matching = df.loc[mask].copy()

        print("\nMOST CONFLICTING LAYOUT")
        print("=" * 45)
        print(f"Records: {len(matching)}")
        print(f"Distinct targets: {matching[target_column].nunique()}")

        print("\nQW AND TARGET RELATIONSHIP")
        print("=" * 45)
        print(f"Distinct qW values: {matching['qW'].nunique()}")
        print(
            matching.groupby("qW")[target_column]
            .agg(["count", "mean", "std", "min", "max"])
            .round(2)
            .to_string()
        )
        print(
            "\nCorrelation between qW and Total_Power:",
            matching[["qW", target_column]].corr().iloc[0, 1],
        )

    # Check whether identical coordinates AND qW have conflicting targets.
    group_columns = coordinate_columns + ["qW"]
    grouped = df.groupby(group_columns, dropna=False)[target_column]

    target_counts = grouped.nunique()
    target_ranges = grouped.agg(lambda values: values.max() - values.min())
    conflicting_groups = target_counts > 1

    print("\nLAYOUT + qW CONSISTENCY AUDIT")
    print("=" * 45)
    print(f"Unique layout/qW combinations: {len(target_counts):,}")
    print(f"Groups with conflicting targets: {conflicting_groups.sum():,}")

    if conflicting_groups.any():
        print("\nTarget-range statistics for conflicting groups:")
        print(
            target_ranges[conflicting_groups]
            .describe()
            .round(2)
            .to_string()
        )
    else:
        print("All identical layout/qW groups have consistent targets.")


if __name__ == "__main__":
    main()
