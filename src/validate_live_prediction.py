from __future__ import annotations

import csv
import re
from pathlib import Path

from src.live_predict import DATASETS, predict_total_power

ROOT = Path(__file__).resolve().parents[1]


def read_first_layout(csv_path: Path, expected_n: int) -> list[list[float]]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        found = {}
        for column in columns:
            match = re.fullmatch(r"([xy])\s*_?\s*(\d+)", column.strip(), re.I)
            if match:
                found.setdefault(int(match.group(2)), {})[match.group(1).lower()] = column
        pairs = [(i, found[i]["x"], found[i]["y"]) for i in sorted(found)
                 if "x" in found[i] and "y" in found[i]]
        if len(pairs) != expected_n:
            raise RuntimeError(
                f"{csv_path.name}: expected {expected_n} coordinate pairs, found {len(pairs)}"
            )
        row = next(reader)
        return [[float(row[x]), float(row[y])] for _, x, y in pairs]


def main() -> int:
    checked = 0
    for dataset, (stem, count) in DATASETS.items():
        csv_path = ROOT / "data" / "raw" / f"{stem}.csv"
        coords = read_first_layout(csv_path, count)
        for model in ("Coordinate MLP", "Spatial-feature MLP", "GNN"):
            result = predict_total_power(dataset, model, 21, coords)
            value = result["prediction_total_power"]
            if not isinstance(value, float) or not (-float("inf") < value < float("inf")):
                raise RuntimeError(f"Non-finite prediction: {dataset} / {model}")
            checked += 1
            print(
                f"[PASS] {dataset} | {model} | seed 21 | "
                f"Total_Power={value:.3f}"
            )
    print(f"\nLive prediction smoke test: {checked}/12 passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
