#!/usr/bin/env bash
set -euo pipefail

# Run from the project root after copying the three updated scripts into src/.
# Split seed remains fixed at 42; training seed changes across runs.
SEEDS=(21 42 123)
DATASETS=(
  data/raw/WEC_Perth_49.csv
  data/raw/WEC_Perth_100.csv
  data/raw/WEC_Sydney_49.csv
  data/raw/WEC_Sydney_100.csv
)

for seed in "${SEEDS[@]}"; do
  for data in "${DATASETS[@]}"; do
    echo "===== Coordinate MLP | seed=${seed} | ${data} ====="
    python -m src.train_mlp_grouped --data "$data" --seed "$seed" --split-seed 42

    echo "===== Spatial-feature MLP | seed=${seed} | ${data} ====="
    python -m src.train_spatial_mlp --data "$data" --seed "$seed" --split-seed 42

    echo "===== GNN | seed=${seed} | ${data} ====="
    python -m src.train_gnn --data "$data" --seed "$seed" --split-seed 42
  done
done

echo "All 36 runs completed."
