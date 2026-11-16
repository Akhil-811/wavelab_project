
from pathlib import Path

import pandas as pd

from src.data_utils import find_coordinate_columns, find_target_column

DATA_PATH = Path("data/raw/WEC_Perth_49.csv")


def main():
    df = pd.read_csv(DATA_PATH)
    coords = find_coordinate_columns(df.columns)
    target = find_target_column(df)

    # Group rows by coordinates and qW.
    group_columns = coords + ["qW"]
    group_sizes = df.groupby(group_columns, dropna=False)[target].transform("size")
    target_counts = df.groupby(group_columns, dropna=False)[target].transform("nunique")

    conflicts = df.loc[(group_sizes > 1) & (target_counts > 1)].copy()

    print("CONFLICTING RECORDS")
    print("=" * 50)
    print(f"Rows in conflicting groups: {len(conflicts):,}")

    # Compare every non-coordinate, non-target column.
    excluded = set(coords + [target])
    other_columns = [c for c in df.columns if c not in excluded]

    variation = (
        conflicts[other_columns]
        .nunique(dropna=False)
        .sort_values(ascending=False)
    )

    print("\nColumns with the most distinct values in conflicting records:")
    print(variation.head(20).to_string())

    # Display a small sample of records from one conflicting group.
    first_group = conflicts.iloc[0]
    mask = pd.Series(True, index=df.index)

    for col in group_columns:
        mask &= df[col].eq(first_group[col])

    sample = df.loc[mask]

    print("\nSAMPLE: SAME COORDINATES AND qW")
    print("=" * 50)
    print(f"Records: {len(sample)}")
    print(
        sample[["qW", target] + [
            c for c in other_columns
            if c.startswith("Power") and c[5:].isdigit()
        ][:5]]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
