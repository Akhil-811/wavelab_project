
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data_utils import find_coordinate_columns, find_target_column

DATA_PATH = Path("data/raw/WEC_Perth_49.csv")


def main():
    df = pd.read_csv(DATA_PATH)

    coordinate_columns = find_coordinate_columns(df.columns)
    target = find_target_column(df)

    # Every identical coordinate layout receives the same group ID.
    groups = pd.util.hash_pandas_object(
        df[coordinate_columns],
        index=False
    )

    # Split off approximately 15% of layouts for the test set.
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=42
    )

    train_val_idx, test_idx = next(
        first_split.split(df, df[target], groups=groups)
    )

    # Split the remaining layouts into training and validation.
    remaining = df.iloc[train_val_idx]
    remaining_groups = groups.iloc[train_val_idx]

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.1765,
        random_state=42
    )

    train_rel_idx, val_rel_idx = next(
        second_split.split(
            remaining,
            remaining[target],
            groups=remaining_groups
        )
    )

    train = remaining.iloc[train_rel_idx]
    val = remaining.iloc[val_rel_idx]
    test = df.iloc[test_idx]

    train_groups = set(groups.iloc[train.index].tolist())
    val_groups = set(groups.iloc[val.index].tolist())
    test_groups = set(groups.iloc[test.index].tolist())

    print("LAYOUT-GROUPED SPLIT")
    print("=" * 50)
    print(f"Total rows: {len(df):,}")
    print(f"Unique layouts: {groups.nunique():,}")
    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(val):,}")
    print(f"Test rows: {len(test):,}")

    print("\nUnique layouts per split:")
    print(f"Training: {len(train_groups):,}")
    print(f"Validation: {len(val_groups):,}")
    print(f"Test: {len(test_groups):,}")

    print("\nOverlap checks:")
    print(f"Train / validation: {bool(train_groups & val_groups)}")
    print(f"Train / test: {bool(train_groups & test_groups)}")
    print(f"Validation / test: {bool(val_groups & test_groups)}")


if __name__ == "__main__":
    main()
