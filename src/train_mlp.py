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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_utils import load_wave_data


SEED = 42


class CoordinateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(256, 128, 64), dropout=0.10):
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train coordinate-only MLP baseline.")
    parser.add_argument("--data", default="data/raw/WEC_Perth_49.csv")
    parser.add_argument("--target", default=None, help="Exact total-power target column name.")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df, X, y, coord_cols, target_col = load_wave_data(args.data, args.target)
    print(f"Loaded: {args.data}")
    print(f"Rows: {len(df):,}; WEC count: {len(coord_cols)//2}; input dimensions: {X.shape[1]}")
    print(f"Target: {target_col}")
    print("Inputs: coordinates only. Individual power and q-factor are excluded.")

    # Reproducible 70/15/15 split: first reserve 15% test, then 15/85 of
    # the remaining 85% for validation (=12.75% total). Adjusted below to
    # exact 70/15/15 using 0.1764706 of the remaining data.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=args.seed
    )
    val_fraction_of_trainval = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction_of_trainval,
        random_state=args.seed
    )

    # Fit scalers on training data only to prevent leakage.
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = x_scaler.transform(X_val).astype(np.float32)
    X_test_s = x_scaler.transform(X_test).astype(np.float32)
    y_train_s = y_scaler.fit_transform(y_train).astype(np.float32)
    y_val_s = y_scaler.transform(y_val).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train_s)),
        batch_size=args.batch_size, shuffle=True
    )
    X_val_t = torch.from_numpy(X_val_s).to(device)
    y_val_t = torch.from_numpy(y_val_s).to(device)

    model = CoordinateMLP(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred_s = model(X_val_t).cpu().numpy()
        val_pred = y_scaler.inverse_transform(val_pred_s).ravel()
        val_true = y_val.ravel()
        val_mae = mean_absolute_error(val_true, val_pred)
        history.append({
            "epoch": epoch,
            "train_mse_scaled": float(np.mean(train_losses)),
            "validation_mae": float(val_mae),
        })

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            epochs_without_improvement = 0
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
            }, model_dir / "coordinate_mlp_best.pt")
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train MSE(scaled)={np.mean(train_losses):.5f} | "
                f"val MAE={val_mae:.6g} | best={best_val_mae:.6g}"
            )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    # Restore best checkpoint, then evaluate test set once.
    checkpoint = torch.load(model_dir / "coordinate_mlp_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    X_test_t = torch.from_numpy(X_test_s).to(device)

    # Warm-up before timing; synchronize GPU if relevant.
    with torch.no_grad():
        _ = model(X_test_t[:min(32, len(X_test_t))])
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        y_pred_s = model(X_test_t).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    y_pred = y_scaler.inverse_transform(y_pred_s).ravel()
    y_true = y_test.ravel()
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    target_mean = float(np.mean(y_true))
    nmae_denom = float(np.mean(np.abs(y_true)))
    nmae = float(mae / nmae_denom) if nmae_denom > 1e-12 else None
    r2 = float(r2_score(y_true, y_pred))
    metrics = {
        "dataset": str(args.data),
        "rows": int(len(df)),
        "wec_count": int(len(coord_cols) // 2),
        "input_dimensions": int(X.shape[1]),
        "target_column": target_col,
        "split": {"train": int(len(X_train)), "validation": int(len(X_val)), "test": int(len(X_test))},
        "seed": args.seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_mae": float(best_val_mae),
        "test_mae": mae,
        "test_rmse": rmse,
        "test_nmae_mae_over_mean_abs_target": nmae,
        "test_r2": r2,
        "test_inference_total_seconds": float(elapsed),
        "test_inference_ms_per_sample": float(elapsed * 1000 / len(X_test)),
        "test_target_mean": target_mean,
    }

    tag = Path(args.data).stem
    with open(output_dir / f"{tag}_coordinate_mlp_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(history).to_csv(output_dir / f"{tag}_training_history.csv", index=False)
    pd.DataFrame({"actual_total_power": y_true, "predicted_total_power": y_pred}).to_csv(
        output_dir / f"{tag}_test_predictions.csv", index=False
    )

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, s=12, alpha=0.55)
    low = min(float(np.min(y_true)), float(np.min(y_pred)))
    high = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual total farm power")
    plt.ylabel("Predicted total farm power")
    plt.title("Coordinate MLP: actual vs predicted")
    plt.tight_layout()
    plt.savefig(output_dir / f"{tag}_actual_vs_predicted.png", dpi=180)
    plt.close()

    print("\nTEST RESULTS")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"\nSaved metrics, history, predictions and plot in {output_dir.resolve()}")
    print(f"Saved model checkpoint in {(model_dir / 'coordinate_mlp_best.pt').resolve()}")


if __name__ == "__main__":
    main()
