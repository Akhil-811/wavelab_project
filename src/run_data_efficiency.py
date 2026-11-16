from __future__ import annotations

import argparse
import json
import math
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fixed_group_split(X: np.ndarray, y: np.ndarray, groups: np.ndarray, split_seed: int):
    """Match the existing two-stage GroupShuffleSplit procedure."""
    first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=split_seed)
    trainval_idx, test_idx = next(first.split(X, y.ravel(), groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=split_seed)
    train_rel, val_rel = next(
        second.split(X[trainval_idx], y[trainval_idx].ravel(), groups=groups[trainval_idx])
    )
    return trainval_idx[train_rel], trainval_idx[val_rel], test_idx


def make_nested_group_subsets(train_groups: np.ndarray, fractions: list[float], seed: int):
    """Create nested subsets of unique training layouts for fair comparisons."""
    unique_groups = np.unique(train_groups)
    shuffled = np.random.default_rng(seed).permutation(unique_groups)
    subsets = {}
    for fraction in sorted(fractions):
        n = max(1, int(math.ceil(len(shuffled) * fraction)))
        subsets[fraction] = set(shuffled[:n].tolist())
    return subsets


def fit_model(model_name, X_train, y_train, X_val, y_val, X_test, y_test, args, seed, device):
    """Train one existing WaveLab architecture and evaluate on the fixed test set."""
    set_seed(seed)
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train).astype(np.float32)
    y_val_s = y_scaler.transform(y_val).astype(np.float32)

    if model_name in ("coordinate_mlp", "spatial_mlp"):
        x_scaler = StandardScaler()
        X_train_s = x_scaler.fit_transform(X_train).astype(np.float32)
        X_val_s = x_scaler.transform(X_val).astype(np.float32)
        X_test_s = x_scaler.transform(X_test).astype(np.float32)
        model = (CoordinateMLP(X_train_s.shape[1]) if model_name == "coordinate_mlp"
                 else SpatialFeatureMLP(X_train_s.shape[1])).to(device)
        train_inputs = torch.from_numpy(X_train_s)
        val_inputs = torch.from_numpy(X_val_s).to(device)
        test_inputs = torch.from_numpy(X_test_s).to(device)
        batch_size = args.batch_size_mlp
    else:
        # Match train_gnn.py: one shared scalar mean/std for both coordinate axes.
        n_nodes = X_train.shape[1] // 2
        train_coords = X_train.reshape(-1, n_nodes, 2)
        coord_mean = float(train_coords.reshape(-1, 2).mean())
        coord_scale = max(float(train_coords.reshape(-1, 2).std()), 1e-8)

        def reshape_scale(values):
            return ((values.reshape(-1, n_nodes, 2) - coord_mean) / coord_scale).astype(np.float32)

        X_train_s = reshape_scale(X_train)
        X_val_s = reshape_scale(X_val)
        X_test_s = reshape_scale(X_test)
        model = SpatialGNN().to(device)
        train_inputs = torch.from_numpy(X_train_s)
        val_inputs = torch.from_numpy(X_val_s).to(device)
        test_inputs = torch.from_numpy(X_test_s).to(device)
        batch_size = args.batch_size_gnn

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_inputs, torch.from_numpy(y_train_s)),
        batch_size=batch_size, shuffle=True, generator=loader_generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_val_mae = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_pred_s = model(val_inputs).cpu().numpy()
        val_pred = y_scaler.inverse_transform(val_pred_s).ravel()
        val_mae = float(mean_absolute_error(y_val.ravel(), val_pred))
        history.append({"epoch": epoch, "train_mse_scaled": float(np.mean(batch_losses)),
                        "validation_mae": val_mae})

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {model_name}, seed {seed}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred_s = model(test_inputs).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - start

    y_pred = y_scaler.inverse_transform(pred_s).ravel()
    y_true = y_test.ravel()
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    nmae = float(mae / np.mean(np.abs(y_true)))
    r2 = float(r2_score(y_true, y_pred))
    return {"best_epoch": best_epoch, "best_validation_mae": best_val_mae,
            "test_mae": mae, "test_rmse": rmse, "test_nmae": nmae,
            "test_r2": r2, "inference_seconds": inference_seconds, "history": history}


def main():
    parser = argparse.ArgumentParser(description="Layout-grouped data-efficiency experiment using WaveLab models.")
    parser.add_argument("--data", default="data/raw/WEC_Perth_49.csv")
    parser.add_argument("--target", default=None)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[21, 42, 123])
    parser.add_argument("--models", nargs="+", choices=["coordinate_mlp", "spatial_mlp", "gnn"],
                        default=["coordinate_mlp", "spatial_mlp", "gnn"])
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size-mlp", type=int, default=256)
    parser.add_argument("--batch-size-gnn", type=int, default=32)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output-dir", default="reports/data_efficiency_perth49")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if any(f <= 0 or f > 1 for f in args.fractions):
        raise ValueError("Fractions must be > 0 and <= 1.")
    if len(set(args.fractions)) != len(args.fractions):
        raise ValueError("Fractions must not contain duplicates.")

    df, X, y, coord_cols, target_col = load_wave_data(args.data, args.target)
    groups = pd.util.hash_pandas_object(df[coord_cols], index=False).to_numpy()
    train_idx, val_idx, test_idx = fixed_group_split(X, y, groups, args.split_seed)
    subset_by_seed = {
        seed: make_nested_group_subsets(groups[train_idx], args.fractions, seed)
        for seed in args.seeds
    }

    print(f"Dataset: {args.data}")
    print(f"Rows: {len(df):,}; unique layouts: {len(np.unique(groups)):,}")
    print(f"Fixed split rows: train={len(train_idx):,}, validation={len(val_idx):,}, test={len(test_idx):,}")
    print(f"Fixed split layouts: train={len(np.unique(groups[train_idx])):,}, "
          f"validation={len(np.unique(groups[val_idx])):,}, test={len(np.unique(groups[test_idx])):,}")
    print("Sampling unit: unique training layouts; all rows from selected layouts are retained.")
    for seed in args.seeds:
        for fraction in sorted(args.fractions):
            chosen = subset_by_seed[seed][fraction]
            row_count = int(np.isin(groups[train_idx], list(chosen)).sum())
            print(f"seed={seed} fraction={fraction:.0%}: layouts={len(chosen):,}, rows={row_count:,}")
    if args.dry_run:
        print("DRY RUN complete. No models trained and no result files written.")
        return

    X_spatial = None
    if "spatial_mlp" in args.models:
        print("Precomputing spatial features from coordinates...")
        X_spatial = build_spatial_features(X)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records = []

    for seed in args.seeds:
        for fraction in sorted(args.fractions):
            chosen = subset_by_seed[seed][fraction]
            sampled_train_idx = train_idx[np.isin(groups[train_idx], list(chosen))]
            for model_name in args.models:
                print(f"\n=== {model_name} | {fraction:.0%} layouts | seed {seed} ===")
                features = X_spatial if model_name == "spatial_mlp" else X
                result = fit_model(
                    model_name, features[sampled_train_idx], y[sampled_train_idx],
                    features[val_idx], y[val_idx], features[test_idx], y[test_idx],
                    args, seed, device,
                )
                record = {
                    "dataset": Path(args.data).name, "model": model_name,
                    "training_fraction_of_layouts": fraction, "seed": seed,
                    "split_seed": args.split_seed, "train_layouts": int(len(chosen)),
                    "total_train_layouts": int(len(np.unique(groups[train_idx]))),
                    "train_rows": int(len(sampled_train_idx)), "validation_rows": int(len(val_idx)),
                    "test_rows": int(len(test_idx)), "best_epoch": result["best_epoch"],
                    "best_validation_mae": result["best_validation_mae"],
                    "test_mae": result["test_mae"], "test_rmse": result["test_rmse"],
                    "test_nmae": result["test_nmae"], "test_r2": result["test_r2"],
                    "inference_seconds": result["inference_seconds"], "device": str(device),
                }
                all_records.append(record)
                stem = f"{model_name}_{int(round(fraction * 100)):03d}pct_seed{seed}"
                pd.DataFrame(result["history"]).to_csv(out_dir / f"{stem}_history.csv", index=False)
                print(f"MAE={record['test_mae']:.4f} | RMSE={record['test_rmse']:.4f} | "
                      f"R2={record['test_r2']:.5f} | best epoch={record['best_epoch']}")
                pd.DataFrame(all_records).to_csv(out_dir / "data_efficiency_per_seed_partial.csv", index=False)

    per_run = pd.DataFrame(all_records)
    per_run.to_csv(out_dir / "data_efficiency_per_seed.csv", index=False)
    summary = (per_run.groupby(["model", "training_fraction_of_layouts"])
               .agg(runs=("seed", "count"), train_rows_mean=("train_rows", "mean"),
                    train_layouts=("train_layouts", "mean"), mae_mean=("test_mae", "mean"),
                    mae_sd=("test_mae", "std"), rmse_mean=("test_rmse", "mean"),
                    rmse_sd=("test_rmse", "std"), nmae_mean=("test_nmae", "mean"),
                    nmae_sd=("test_nmae", "std"), r2_mean=("test_r2", "mean"),
                    r2_sd=("test_r2", "std")).reset_index())
    summary.to_csv(out_dir / "data_efficiency_mean_sd.csv", index=False)
    metadata = {
        "dataset": str(args.data), "target": target_col, "coordinate_columns": coord_cols,
        "models": args.models, "fractions": sorted(args.fractions), "seeds": args.seeds,
        "split_seed": args.split_seed,
        "split_method": "same two-stage GroupShuffleSplit as existing training scripts",
        "sampling_unit": "unique coordinate-layout groups; all rows for selected layouts retained",
        "nested_subsets": True, "validation_and_test_fixed_across_fractions": True,
        "scalers_fit_on_sampled_training_rows_only": True, "epochs": args.epochs,
        "patience": args.patience, "device": str(device),
        "interpretation_note": "Training fraction refers to unique training layouts, not raw rows.",
    }
    (out_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("\nDATA-EFFICIENCY SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
