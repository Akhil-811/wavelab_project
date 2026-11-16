from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_utils import load_wave_data
from src.train_gnn import SpatialGNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fixed_group_split(X: np.ndarray, y: np.ndarray, groups: np.ndarray, split_seed: int):
    first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=split_seed)
    trainval_idx, test_idx = next(first.split(X, y.ravel(), groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=split_seed)
    train_rel, val_rel = next(
        second.split(X[trainval_idx], y[trainval_idx].ravel(), groups=groups[trainval_idx])
    )
    return trainval_idx[train_rel], trainval_idx[val_rel], test_idx


def run_one(X, y, train_idx, val_idx, test_idx, *, seed, batch_size,
            epochs, patience, learning_rate, device):
    # Reset all model/dropout RNG state for each batch-size condition.
    set_seed(seed)
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y[train_idx]).astype(np.float32)

    n_wec = X.shape[1] // 2
    train_coords = X[train_idx].reshape(-1, n_wec, 2)
    coord_mean = float(train_coords.reshape(-1, 2).mean())
    coord_scale = max(float(train_coords.reshape(-1, 2).std()), 1e-8)

    def scale_coords(values):
        return ((values.reshape(-1, n_wec, 2) - coord_mean) / coord_scale).astype(np.float32)

    train_x = torch.from_numpy(scale_coords(X[train_idx]))
    val_x = torch.from_numpy(scale_coords(X[val_idx])).to(device)
    test_x = torch.from_numpy(scale_coords(X[test_idx])).to(device)
    train_y = torch.from_numpy(y_train_s)

    # Keep the same dedicated seeded DataLoader generator in both conditions.
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size,
                        shuffle=True, generator=generator)

    model = SpatialGNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_val_mae = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred_s = model(val_x).cpu().numpy()
        val_pred = y_scaler.inverse_transform(val_pred_s).ravel()
        val_mae = float(mean_absolute_error(y[val_idx].ravel(), val_pred))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    training_seconds = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError(f"No best checkpoint selected for seed={seed}, batch_size={batch_size}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_s = model(test_x).cpu().numpy()

    pred = y_scaler.inverse_transform(pred_s).ravel()
    truth = y[test_idx].ravel()
    return {
        "best_epoch": best_epoch,
        "best_validation_mae": best_val_mae,
        "test_mae": float(mean_absolute_error(truth, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(truth, pred))),
        "test_r2": float(r2_score(truth, pred)),
        "training_seconds": float(training_seconds),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Controlled Perth49 GNN batch-size comparison (16 vs 32)."
    )
    parser.add_argument("--data", default="data/raw/WEC_Perth_49.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[21, 42, 123])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="reports/gnn_batch_comparison")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if len(set(args.batch_sizes)) != len(args.batch_sizes) or any(x < 1 for x in args.batch_sizes):
        raise ValueError("Batch sizes must be unique positive integers.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique.")

    df, X, y, coord_cols, target_col = load_wave_data(args.data)
    groups = pd.util.hash_pandas_object(df[coord_cols], index=False).to_numpy()
    train_idx, val_idx, test_idx = fixed_group_split(X, y, groups, args.split_seed)
    print(f"Dataset: {args.data}")
    print(f"Rows: {len(df):,}; unique layouts: {np.unique(groups).size:,}")
    print(f"Fixed split rows: train={len(train_idx):,}, validation={len(val_idx):,}, test={len(test_idx):,}")
    print(f"Fixed split layouts: train={np.unique(groups[train_idx]).size:,}, "
          f"validation={np.unique(groups[val_idx]).size:,}, test={np.unique(groups[test_idx]).size:,}")
    print(f"Planned runs: {len(args.seeds) * len(args.batch_sizes)}")
    if args.dry_run:
        print("DRY RUN complete. No models trained and no result files written.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in args.seeds:
        for batch_size in args.batch_sizes:
            print(f"\n=== Perth49 GNN | batch_size={batch_size} | seed={seed} ===", flush=True)
            result = run_one(X, y, train_idx, val_idx, test_idx, seed=seed,
                             batch_size=batch_size, epochs=args.epochs,
                             patience=args.patience, learning_rate=args.learning_rate,
                             device=device)
            record = {
                "dataset": Path(args.data).name,
                "model": "gnn",
                "seed": seed,
                "batch_size": batch_size,
                "split_seed": args.split_seed,
                "rows_total": len(df),
                "train_rows": len(train_idx),
                "validation_rows": len(val_idx),
                "test_rows": len(test_idx),
                "train_layouts": int(np.unique(groups[train_idx]).size),
                "validation_layouts": int(np.unique(groups[val_idx]).size),
                "test_layouts": int(np.unique(groups[test_idx]).size),
                "epochs_max": args.epochs,
                "patience": args.patience,
                "learning_rate": args.learning_rate,
                "device": str(device),
                **result,
            }
            records.append(record)
            pd.DataFrame(records).to_csv(out / "gnn_batch_comparison_per_seed_partial.csv", index=False)
            print(f"MAE={record['test_mae']:.3f} | RMSE={record['test_rmse']:.3f} | "
                  f"R2={record['test_r2']:.5f} | best_epoch={record['best_epoch']}", flush=True)

    per_seed = pd.DataFrame(records)
    per_seed.to_csv(out / "gnn_batch_comparison_per_seed.csv", index=False)
    summary = (per_seed.groupby("batch_size")
               .agg(runs=("seed", "count"), mae_mean=("test_mae", "mean"),
                    mae_sd=("test_mae", "std"), rmse_mean=("test_rmse", "mean"),
                    rmse_sd=("test_rmse", "std"), r2_mean=("test_r2", "mean"),
                    r2_sd=("test_r2", "std"), best_epoch_mean=("best_epoch", "mean"),
                    training_seconds_mean=("training_seconds", "mean"))
               .reset_index())
    summary.to_csv(out / "gnn_batch_comparison_mean_sd.csv", index=False)
    print("\n=== Batch-size summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {out.resolve()}")


if __name__ == "__main__":
    main()
