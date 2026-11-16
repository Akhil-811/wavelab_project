from pathlib import Path

import pandas as pd

from src.data_utils import find_coordinate_columns, normalise_name

RAW_DIR = Path("data/raw")


def main():
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        print("No CSV files found in data/raw/.")
        print("1. Open the official UCI dataset page:")
        print("   https://archive.ics.uci.edu/dataset/882/large-scale+wave+energy+farm")
        print("2. Download WEC.zip and extract the CSV files into data/raw/.")
        return

    print(f"Found {len(files)} CSV file(s) in {RAW_DIR.resolve()}\n")
    for path in files:
        print("=" * 88)
        print(f"FILE: {path.name}")
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"Could not read CSV: {exc}")
            continue

        print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
        print(f"Duplicate complete rows: {int(df.duplicated().sum()):,}")
        print(f"Missing cells: {int(df.isna().sum().sum()):,}")
        print("Columns:")
        print(", ".join(map(str, df.columns)))

        try:
            coords = find_coordinate_columns(df.columns)
            print(f"Detected coordinate columns: {len(coords)} ({len(coords)//2} WECs)")
            print(f"Coordinate sample: {coords[:min(8, len(coords))]}")
        except ValueError as exc:
            print(f"Coordinate detection: {exc}")

        target_candidates = [
            str(c) for c in df.columns
            if "total" in normalise_name(c) and "power" in normalise_name(c)
        ]
        print(f"Candidate total-power columns: {target_candidates or 'NONE FOUND'}")

        if target_candidates:
            for target in target_candidates:
                values = pd.to_numeric(df[target], errors="coerce")
                print(
                    f"Target '{target}': min={values.min()}, max={values.max()}, "
                    f"mean={values.mean()}, std={values.std()}"
                )
        print()


if __name__ == "__main__":
    main()
