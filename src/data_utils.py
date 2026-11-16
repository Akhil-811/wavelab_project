from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_CANDIDATES = [
    "total_farm_power_output",
    "total_power",
    "totalpower",
    "total_power_output",
    "farm_total_power",
    "total_farm_power",
]


def normalise_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def find_coordinate_columns(columns: Iterable[object]) -> list[str]:
    """Return X1,Y1,X2,Y2... coordinate columns in numeric device order."""
    found: dict[int, dict[str, str]] = {}
    for col in columns:
        s = str(col).strip()
        match = re.fullmatch(r"([xy])\s*_?\s*(\d+)", s, flags=re.IGNORECASE)
        if match:
            axis = match.group(1).lower()
            idx = int(match.group(2))
            found.setdefault(idx, {})[axis] = s

    complete = [
        (idx, pair["x"], pair["y"])
        for idx, pair in found.items()
        if "x" in pair and "y" in pair
    ]
    complete.sort(key=lambda item: item[0])

    if not complete:
        raise ValueError(
            "Could not detect coordinate columns. Expected names like X1,Y1,X2,Y2. "
            "Run `python -m src.inspect_dataset` and inspect the printed column names."
        )

    # Require contiguous device numbering to avoid silently missing a device.
    indices = [idx for idx, _, _ in complete]
    expected = list(range(indices[0], indices[-1] + 1))
    if indices != expected:
        raise ValueError(f"Coordinate numbering has gaps: found {indices}")

    ordered: list[str] = []
    for _, x_col, y_col in complete:
        ordered.extend([x_col, y_col])
    return ordered


def find_target_column(df: pd.DataFrame, requested: str | None = None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(
                f"Target '{requested}' not found. Available columns: {list(df.columns)}"
            )
        return requested

    by_normalised = {normalise_name(c): str(c) for c in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in by_normalised:
            return by_normalised[candidate]

    # Conservative fallback: only accept a single column containing both
    # 'total' and 'power'; never silently select an individual Power_i column.
    matches = [
        str(c) for c in df.columns
        if "total" in normalise_name(c) and "power" in normalise_name(c)
    ]
    if len(matches) == 1:
        return matches[0]

    raise ValueError(
        "Could not uniquely identify total farm power target. "
        "Run `python -m src.inspect_dataset`, then pass `--target EXACT_COLUMN_NAME`."
    )


def load_wave_data(path: str | Path, target: str | None = None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Download WEC.zip from the official UCI dataset page, extract its CSV files "
            "into data/raw/, then rerun."
        )

    df = pd.read_csv(path)
    coord_cols = find_coordinate_columns(df.columns)
    target_col = find_target_column(df, target)

    X_df = df[coord_cols].apply(pd.to_numeric, errors="coerce")
    y_series = pd.to_numeric(df[target_col], errors="coerce")

    if X_df.isna().any().any() or y_series.isna().any():
        bad_x = int(X_df.isna().sum().sum())
        bad_y = int(y_series.isna().sum())
        raise ValueError(
            f"NaNs/non-numeric values found after conversion (coordinate cells={bad_x}, "
            f"target cells={bad_y}). Inspect the source CSV before proceeding."
        )

    X = X_df.to_numpy(dtype=np.float32)
    y = y_series.to_numpy(dtype=np.float32).reshape(-1, 1)

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite values (inf/-inf) found in features or target.")

    return df, X, y, coord_cols, target_col
