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
from src.train_mlp_grouped import CoordinateMLP
from src.train_spatial_mlp import SpatialFeatureMLP, build_spatial_features
from src.train_gnn import SpatialGNN

DEFAULT_DATASETS = [
    "data/raw/WEC_Perth_49.csv",
    "data/raw/WEC_Perth_100.csv",
    "data/raw/WEC_Sydney_49.csv",
    "data/raw/WEC_Sydney_100.csv",
]
MODELS = ["coordinate_mlp", "spatial_mlp", "gnn"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_layout(X, y, groups, split_seed):
    first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=split_seed)
    trainval, test = next(first.split(X, y.ravel(), groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=split_seed)
    train_rel, val_rel = next(second.split(
        X[trainval], y[trainval].ravel(), groups=groups[trainval]
    ))
    return trainval[train_rel], trainval[val_rel], test


def fit_one(model_name, X_train, y_train, X_val, y_val, X_test, y_test,
            args, seed, device, n_wec):
    set_seed(seed)
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train).astype(np.float32)
    y_val_s = y_scaler.transform(y_val).astype(np.float32)

    if model_name in ("coordinate_mlp", "spatial_mlp"):
        x_scaler = StandardScaler()
        x_train_s = x_scaler.fit_transform(X_train).astype(np.float32)
        x_val_s = x_scaler.transform(X_val).astype(np.float32)
        x_test_s = x_scaler.transform(X_test).astype(np.float32)
        model = (CoordinateMLP(x_train_s.shape[1]) if model_name == "coordinate_mlp"
                 else SpatialFeatureMLP(x_train_s.shape[1])).to(device)
        train_x = torch.from_numpy(x_train_s)
        val_x = torch.from_numpy(x_val_s).to(device)
        test_x = torch.from_numpy(x_test_s).to(device)
        batch_size = args.batch_size_mlp
    else:
        train_coords = X_train.reshape(-1, n_wec, 2)
        coord_mean = float(train_coords.reshape(-1, 2).mean())
        coord_scale = max(float(train_coords.reshape(-1, 2).std()), 1e-8)
        def scale_coords(arr):
            return ((arr.reshape(-1, n_wec, 2) - coord_mean) / coord_scale).astype(np.float32)
        train_x = torch.from_numpy(scale_coords(X_train))
        val_x = torch.from_numpy(scale_coords(X_val)).to(device)
        test_x = torch.from_numpy(scale_coords(X_test)).to(device)
        model = SpatialGNN().to(device)
        batch_size = args.batch_size_gnn

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(TensorDataset(train_x, torch.from_numpy(y_train_s)),
                        batch_size=batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    start_train = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
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
        val_mae = float(mean_absolute_error(y_val.ravel(), val_pred))
        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    train_seconds = time.perf_counter() - start_train
    if best_state is None:
        raise RuntimeError(f"No best model state for {model_name}, seed {seed}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        if device.type == "cuda": torch.cuda.synchronize()
        start_inf = time.perf_counter()
        pred_s = model(test_x).cpu().numpy()
        if device.type == "cuda": torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - start_inf

    pred = y_scaler.inverse_transform(pred_s).ravel()
    truth = y_test.ravel()
    return {
        "best_epoch": best_epoch,
        "best_validation_mae": best_val,
        "test_mae": float(mean_absolute_error(truth, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(truth, pred))),
        "test_r2": float(r2_score(truth, pred)),
        "training_seconds": float(train_seconds),
        "test_inference_seconds": float(inference_seconds),
        "inference_ms_per_sample": float(1000 * inference_seconds / len(truth)),
    }


def main():
    p = argparse.ArgumentParser(description="WaveLab 49-vs-100 WEC scalability experiment")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=[21, 42, 123])
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size-mlp", type=int, default=256)
    p.add_argument("--batch-size-gnn", type=int, default=16)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--output-dir", default="reports/scalability")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    print(f"Training device: {device}")
    print(f"Planned runs: {len(args.datasets) * len(args.models) * len(args.seeds)}")

    for data_path in args.datasets:
        df, X, y, coord_cols, target_col = load_wave_data(data_path)
        groups = pd.util.hash_pandas_object(df[coord_cols], index=False).to_numpy()
        train_idx, val_idx, test_idx = split_by_layout(X, y, groups, args.split_seed)
        n_wec = X.shape[1] // 2
        n_layouts = int(np.unique(groups).size)
        print(f"\nDATASET {Path(data_path).name}: rows={len(df):,}; WECs={n_wec}; "
              f"layouts={n_layouts:,}; train/val/test={len(train_idx):,}/{len(val_idx):,}/{len(test_idx):,}")
        if args.dry_run:
            continue

        spatial_X = None
        if "spatial_mlp" in args.models:
            print("Precomputing spatial features...")
            spatial_X = build_spatial_features(X)

        for seed in args.seeds:
            for model_name in args.models:
                features = spatial_X if model_name == "spatial_mlp" else X
                print(f"\n=== {Path(data_path).stem} | {model_name} | seed {seed} ===", flush=True)
                result = fit_one(model_name, features[train_idx], y[train_idx],
                                 features[val_idx], y[val_idx], features[test_idx], y[test_idx],
                                 args, seed, device, n_wec)
                rec = {
                    "dataset": Path(data_path).name,
                    "site": "Perth" if "Perth" in Path(data_path).name else "Sydney",
                    "wec_count": n_wec,
                    "model": model_name,
                    "seed": seed,
                    "split_seed": args.split_seed,
                    "rows_total": len(df),
                    "unique_layouts": n_layouts,
                    "train_rows": len(train_idx),
                    "validation_rows": len(val_idx),
                    "test_rows": len(test_idx),
                    "coordinate_input_dimensions": X.shape[1],
                    "model_input_dimensions": features.shape[1],
                    "device": str(device),
                    **{k: v for k, v in result.items() if k != "history"},
                }
                records.append(rec)
                pd.DataFrame(records).to_csv(out / "scalability_per_seed_partial.csv", index=False)
                print(f"MAE={rec['test_mae']:.3f} | RMSE={rec['test_rmse']:.3f} | "
                      f"R2={rec['test_r2']:.5f} | train={rec['training_seconds']:.1f}s | "
                      f"inference={rec['inference_ms_per_sample']:.4f}ms/sample", flush=True)

    if args.dry_run:
        print("DRY RUN complete. No training performed.")
        return
    per_run = pd.DataFrame(records)
    per_run.to_csv(out / "scalability_per_seed.csv", index=False)
    summary = (per_run.groupby(["site", "wec_count", "model"])
               .agg(runs=("seed", "count"), rows_total=("rows_total", "mean"),
                    unique_layouts=("unique_layouts", "mean"), train_rows=("train_rows", "mean"),
                    test_rows=("test_rows", "mean"), mae_mean=("test_mae", "mean"),
                    mae_sd=("test_mae", "std"), rmse_mean=("test_rmse", "mean"),
                    rmse_sd=("test_rmse", "std"), r2_mean=("test_r2", "mean"),
                    r2_sd=("test_r2", "std"), training_seconds_mean=("training_seconds", "mean"),
                    training_seconds_sd=("training_seconds", "std"),
                    inference_ms_per_sample_mean=("inference_ms_per_sample", "mean"),
                    inference_ms_per_sample_sd=("inference_ms_per_sample", "std"))
               .reset_index())
    summary.to_csv(out / "scalability_mean_sd.csv", index=False)
    print("\nSCALABILITY SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {out.resolve()}")


if __name__ == "__main__":
    main()
