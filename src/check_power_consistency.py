
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_utils import find_coordinate_columns, find_target_column

DATA_PATH = Path("data/raw/WEC_Perth_49.csv")


def main():
    df = pd.read_csv(DATA_PATH)

    coords = find_coordinate_columns(df.columns)
    target = find_target_column(df)

    power_columns = [
        col for col in df.columns
        if col.startswith("Power") and col[5:].isdigit()
    ]

    group_columns = coords + ["qW"]
    group_sizes = df.groupby(
        group_columns, dropna=False
    )[target].transform("size")

    target_counts = df.groupby(
        group_columns, dropna=False
    )[target].transform("nunique")

    conflicts = df.loc[
        (group_sizes > 1) & (target_counts > 1)
    ].copy()

    # Select one group with the same coordinates and qW.
    first = conflicts.iloc[0]
    mask = pd.Series(True, index=df.index)

    for col in group_columns:
        mask &= df[col].eq(first[col])

    sample = df.loc[mask].copy()

    # Compare the target with the sum of individual WEC powers.
    sample["Power_Sum"] = sample[power_columns].sum(axis=1)
    sample["Target_Minus_Power_Sum"] = (
        sample[target] - sample["Power_Sum"]
    )

    print("POWER CONSISTENCY CHECK")
    print("=" * 60)
    print(f"Sample group records: {len(sample)}")
    print(f"Number of WEC power columns: {len(power_columns)}")

    print("\nTarget versus sum of individual powers:")
    print(
        sample[["qW", target, "Power_Sum",
                "Target_Minus_Power_Sum"]].to_string(index=False)
    )

    print("\nColumns that vary within this group:")
    varying = [
        col for col in df.columns
        if sample[col].nunique(dropna=False) > 1
    ]
    print(varying)

    print("\nSample records:")
    print(
        sample[["qW", target] + power_columns[:5]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
