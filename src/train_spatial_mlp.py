
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


class SpatialFeatureMLP(nn.Module):
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


def build_spatial_features(X):
    """
    Append geometry features to the raw coordinates.

    For each WEC, calculate distances to its 1st, 3rd and 5th
    nearest neighbours. Also calculate global layout statistics.

    Features are calculated once per unique layout, then mapped
    back to all rows with that layout.
    """
    X = np.asarray(X, dtype=np.float64)
    unique_layouts, inverse = np.unique(
        X, axis=0, return_inverse=True
    )

    n_layouts, input_dim = unique_layouts.shape
    if input_dim % 2 != 0:
        raise ValueError("Expected an even number of coordinate columns.")

    n_wec = input_dim // 2
    if n_wec < 6:
        raise ValueError("At least six WECs are required for these features.")

    # Each layout gets 3 distances per WEC plus 7 global features.
    geometry = np.zeros((n_layouts, 3 * n_wec + 7), dtype=np.float32)

    for i, row in enumerate(unique_layouts):
        coords = row.reshape(n_wec, 2)

        # Euclidean distance between every pair of WECs.
        delta = coords[:, None, :] - coords[None, :, :]
        distances = np.sqrt(np.sum(delta * delta, axis=2))

        # Exclude each WEC's distance to itself.
        np.fill_diagonal(distances, np.inf)
        sorted_distances = np.sort(distances, axis=1)

        # Per-WEC distances to the 1st, 3rd and 5th neighbours.
        nearest = sorted_distances[:, [0, 2, 4]].reshape(-1)

        # Unique pairwise distances for global layout statistics.
        pairwise = distances[np.triu_indices(n_wec, k=1)]

        x_span = float(np.ptp(coords[:, 0]))
        y_span = float(np.ptp(coords[:, 1]))

        global_features = np.array([
            np.mean(pairwise),
            np.std(pairwise),
            np.min(pairwise),
            np.max(pairwise),
            x_span,
            y_span,
            x_span * y_span,
        ], dtype=np.float32)

        geometry[i] = np.concatenate([
            nearest.astype(np.float32),
            global_features,
        ])

    # Identical layouts receive identical geometric features.
    X_spatial = np.concatenate([
        X.astype(np.float32),
        geometry[inverse],
    ], axis=1)

    return X_spatial


def main():
    parser = argparse.ArgumentParser(
        description="Train a coordinate plus spatial-feature MLP."
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
    parser.add_argument("--output-dir", default="results/spatial")
    parser.add_argument("--model-dir", default="models/spatial")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df, X, y, coord_cols, target_col = load_wave_data(
        args.data, args.target
    )

    # Match the grouped baseline's layout groups and split procedure.
    groups = pd.util.hash_pandas_object(
        df[coord_cols], index=False
    ).to_numpy()

    split1 = GroupShuffleSplit(
        n_splits=1, test_size=0.15, random_state=args.split_seed
    )
    trainval_idx, test_idx = next(
        split1.split(X, y, groups=groups)
    )

    split2 = GroupShuffleSplit(
        n_splits=1, test_size=0.1765, random_state=args.split_seed
    )
    train_rel, val_rel = next(
        split2.split(
            X[trainval_idx],
            y[trainval_idx],
            groups=groups[trainval_idx],
        )
    )

    train_idx = trainval_idx[train_rel]
    val_idx = trainval_idx[val_rel]

    # Geometry depends only on coordinates, never on target values.
    print("Building spatial features...")
    X_features = build_spatial_features(X)

    X_train = X_features[train_idx]
    X_val = X_features[val_idx]
    X_test = X_features[test_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    # Fit preprocessing only on the training split.
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
            torch.from_numpy(y_train_s),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    X_val_t = torch.from_numpy(X_val_s).to(device)

    model = SpatialFeatureMLP(input_dim=X_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    loss_fn = nn.MSELoss()

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    tag = Path(args.data).stem
    run_tag = f"{tag}_spatial_mlp_seed{args.seed}"
    checkpoint_path = model_dir / f"{run_tag}_best.pt"

    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    print(f"Dataset: {args.data}")
    print(f"Rows: {len(df):,}")
    print(f"Unique layouts: {len(np.unique(groups)):,}")
    print(f"WEC count: {len(coord_cols) // 2}")
    print(f"Coordinate input dimensions: {X.shape[1]}")
    print(f"Spatial-feature input dimensions: {X_features.shape[1]}")
    print(f"Target: {target_col}")
    print(f"Train/validation/test rows: "
          f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

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
        val_mae = float(mean_absolute_error(val_true, val_pred))

        history.append({
            "epoch": epoch,
            "train_mse_scaled": float(np.mean(batch_losses)),
            "validation_mae": val_mae,
        })

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": X_features.shape[1],
                "coordinate_columns": coord_cols,
                "target_column": target_col,
                "x_mean": x_scaler.mean_.tolist(),
                "x_scale": x_scaler.scale_.tolist(),
                "y_mean": y_scaler.mean_.tolist(),
                "y_scale": y_scaler.scale_.tolist(),
                "seed": args.seed,
        "split_seed": args.split_seed,
                "feature_description": (
                    "raw coordinates, per-WEC 1st/3rd/5th nearest "
                    "neighbour distances, and global layout statistics"
                ),
            }, checkpoint_path)
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train MSE(scaled)={np.mean(batch_losses):.5f} | "
                f"val MAE={val_mae:.6g} | best={best_val_mae:.6g}"
            )

        if patience_counter >= args.patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch was {best_epoch}."
            )
            break

    # Evaluate the best validation-selected checkpoint on test data.
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
    mean_abs_target = float(np.mean(np.abs(y_true)))

    metrics = {
        "dataset": str(args.data),
        "rows": int(len(df)),
        "unique_layouts": int(len(np.unique(groups))),
        "wec_count": int(len(coord_cols) // 2),
        "coordinate_input_dimensions": int(X.shape[1]),
        "spatial_feature_input_dimensions": int(X_features.shape[1]),
        "target_column": target_col,
        "split_method": "layout_grouped",
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "seed": args.seed,
        "split_seed": args.split_seed,
        "best_epoch": best_epoch,
        "best_validation_mae": best_val_mae,
        "test_mae": mae,
        "test_rmse": rmse,
        "test_nmae": (
            float(mae / mean_abs_target)
            if mean_abs_target > 1e-12 else None
        ),
        "test_r2": float(r2_score(y_true, y_pred)),
        "test_inference_total_seconds": float(elapsed),
        "test_inference_ms_per_sample": float(
            elapsed * 1000 / len(X_test)
        ),
    }

    with open(output_dir / f"{run_tag}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(history).to_csv(
        output_dir / f"{run_tag}_history.csv", index=False
    )
    pd.DataFrame({
        "actual_total_power": y_true,
        "predicted_total_power": y_pred,
    }).to_csv(
        output_dir / f"{run_tag}_predictions.csv", index=False
    )

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, s=12, alpha=0.55)
    low = min(float(y_true.min()), float(y_pred.min()))
    high = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual total farm power")
    plt.ylabel("Predicted total farm power")
    plt.title("Spatial-feature MLP: layout-grouped test")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{run_tag}_actual_vs_predicted.png",
        dpi=180,
    )
    plt.close()

    print("\nSPATIAL-FEATURE MLP TEST RESULTS")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"\nResults saved to: {output_dir.resolve()}")
    print(f"Checkpoint saved to: {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
