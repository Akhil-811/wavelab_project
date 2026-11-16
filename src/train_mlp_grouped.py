
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_utils import load_wave_data


SEED = 42


class CoordinateMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.10):
        super().__init__()
        layers = []
        previous = input_dim

        for width in hidden_dims:
            layers.extend([
                nn.Linear(previous, width),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            previous = width

        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Train coordinate-only MLP using layout-grouped splits."
    )
    parser.add_argument("--data", default="data/raw/WEC_Perth_49.csv")
    parser.add_argument("--target", default=None)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Fixed seed for layout-grouped train/validation/test splits.")
    parser.add_argument("--output-dir", default="results/grouped")
    parser.add_argument("--model-dir", default="models/grouped")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df, X, y, coord_cols, target_col = load_wave_data(
        args.data, args.target
    )

    # Hash identical coordinate layouts into groups.
    groups = pd.util.hash_pandas_object(
        df[coord_cols], index=False
    ).to_numpy()

    # Reserve 15% of unique layouts for testing.
    split1 = GroupShuffleSplit(
        n_splits=1, test_size=0.15, random_state=args.split_seed
    )
    trainval_idx, test_idx = next(
        split1.split(X, y, groups=groups)
    )

    # Reserve approximately 15% of all layouts for validation.
    remaining_groups = groups[trainval_idx]
    split2 = GroupShuffleSplit(
        n_splits=1, test_size=0.1765, random_state=args.split_seed
    )
    train_rel, val_rel = next(
        split2.split(
            X[trainval_idx],
            y[trainval_idx],
            groups=remaining_groups
        )
    )

    train_idx = trainval_idx[train_rel]
    val_idx = trainval_idx[val_rel]

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    # Fit both scalers on training data only.
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = x_scaler.transform(X_val).astype(np.float32)
    X_test_s = x_scaler.transform(X_test).astype(np.float32)

    y_train_s = y_scaler.fit_transform(y_train).astype(np.float32)
    y_val_s = y_scaler.transform(y_val).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train_s),
            torch.from_numpy(y_train_s)
        ),
        batch_size=args.batch_size,
        shuffle=True
    )

    X_val_t = torch.from_numpy(X_val_s).to(device)
    y_val_t = torch.from_numpy(y_val_s).to(device)

    model = CoordinateMLP(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4
    )
    loss_fn = nn.MSELoss()

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    tag = Path(args.data).stem
    run_tag = f"{tag}_coordinate_mlp_grouped_seed{args.seed}"
    checkpoint_path = model_dir / f"{run_tag}.pt"

    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    print(f"Dataset: {args.data}")
    print(f"Rows: {len(df):,}")
    print(f"Unique layouts: {len(np.unique(groups)):,}")
    print(f"Train rows: {len(train_idx):,}")
    print(f"Validation rows: {len(val_idx):,}")
    print(f"Test rows: {len(test_idx):,}")
    print(f"Input dimensions: {X.shape[1]}")
    print(f"Target: {target_col}")
    print("Inputs: coordinates only")

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred_s = model(X_val_t).cpu().numpy()

        val_pred = y_scaler.inverse_transform(val_pred_s).ravel()
        val_true = y_val.ravel()
        val_mae = mean_absolute_error(val_true, val_pred)

        history.append({
            "epoch": epoch,
            "train_mse_scaled": float(np.mean(batch_losses)),
            "validation_mae": float(val_mae),
        })

        if val_mae < best_val_mae:
            best_val_mae = float(val_mae)
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": X.shape[1],
                "hidden_dims": [256, 128, 64],
                "dropout": 0.10,
                "coordinate_columns": coord_cols,
                "target_column": target_col,
                "x_mean": x_scaler.mean_.tolist(),
                "x_scale": x_scaler.scale_.tolist(),
                "y_mean": y_scaler.mean_.tolist(),
                "y_scale": y_scaler.scale_.tolist(),
                "seed": args.seed,
        "split_seed": args.split_seed,
            }, checkpoint_path)
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train MSE(scaled)={np.mean(batch_losses):.5f} | "
                f"val MAE={val_mae:.6g} | "
                f"best={best_val_mae:.6g}"
            )

        if patience_counter >= args.patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch was {best_epoch}."
            )
            break

    # Load the best validation checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_test_t = torch.from_numpy(X_test_s).to(device)

    with torch.no_grad():
        _ = model(X_test_t[:min(32, len(X_test_t))])
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        pred_s = model(X_test_t).cpu().numpy()

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    y_pred = y_scaler.inverse_transform(pred_s).ravel()
    y_true = y_test.ravel()

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mean_abs_target = float(np.mean(np.abs(y_true)))

    metrics = {
        "dataset": str(args.data),
        "rows": len(df),
        "unique_layouts": int(len(np.unique(groups))),
        "wec_count": len(coord_cols) // 2,
        "input_dimensions": int(X.shape[1]),
        "target_column": target_col,
        "split_method": "layout_grouped",
        "train_rows": len(train_idx),
        "validation_rows": len(val_idx),
        "test_rows": len(test_idx),
        "train_layouts": int(len(np.unique(groups[train_idx]))),
        "validation_layouts": int(len(np.unique(groups[val_idx]))),
        "test_layouts": int(len(np.unique(groups[test_idx]))),
        "seed": args.seed,
        "split_seed": args.split_seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_mae": best_val_mae,
        "test_mae": mae,
        "test_rmse": rmse,
        "test_nmae": (
            float(mae / mean_abs_target)
            if mean_abs_target > 1e-12 else None
        ),
        "test_r2": r2,
        "test_inference_total_seconds": float(elapsed),
        "test_inference_ms_per_sample": float(
            elapsed * 1000 / len(X_test)
        ),
    }

    with open(output_dir / f"{run_tag}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(history).to_csv(
        output_dir / f"{run_tag}_history.csv",
        index=False
    )
    pd.DataFrame({
        "actual_total_power": y_true,
        "predicted_total_power": y_pred
    }).to_csv(
        output_dir / f"{run_tag}_predictions.csv",
        index=False
    )

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, s=12, alpha=0.55)
    low = min(float(y_true.min()), float(y_pred.min()))
    high = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual total farm power")
    plt.ylabel("Predicted total farm power")
    plt.title("Coordinate MLP: layout-grouped test")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{run_tag}_actual_vs_predicted.png",
        dpi=180
    )
    plt.close()

    print("\nGROUPED TEST RESULTS")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"\nResults saved to: {output_dir.resolve()}")
    print(f"Checkpoint saved to: {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
