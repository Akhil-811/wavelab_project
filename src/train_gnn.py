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

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

class GraphConv(nn.Module):

    """Mean-aggregating graph convolution for a batch of graphs."""

    def __init__(self, in_features, out_features):

        super().__init__()

        self.self_linear = nn.Linear(in_features, out_features)

        self.neighbor_linear = nn.Linear(in_features, out_features)

    def forward(self, h, adjacency):

        # h: [batch, nodes, features]

        # adjacency: [batch, nodes, nodes]

        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)

        neighbor_mean = torch.bmm(adjacency, h) / degree

        return (

            self.self_linear(h)

            + self.neighbor_linear(neighbor_mean)

        )

class SpatialGNN(nn.Module):

    def __init__(self, hidden_dim=64, dropout=0.10):

        super().__init__()

        self.conv1 = GraphConv(2, hidden_dim)

        self.conv2 = GraphConv(hidden_dim, hidden_dim)

        self.conv3 = GraphConv(hidden_dim, hidden_dim)

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        # Mean pooling + max pooling produces 2 * hidden_dim features.

        self.regressor = nn.Sequential(

            nn.Linear(hidden_dim * 2, 128),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Linear(64, 1),

        )

    def build_adjacency(self, coords, k=8):

        """

        Build a directed k-nearest-neighbour graph and symmetrize it.

        Self-edges are excluded.

        """

        batch_size, n_nodes, _ = coords.shape

        distances = torch.cdist(coords, coords, p=2)

        eye = torch.eye(

            n_nodes, dtype=torch.bool, device=coords.device

        ).unsqueeze(0)

        distances = distances.masked_fill(eye, float("inf"))

        k = min(k, n_nodes - 1)

        nearest = torch.topk(

            distances, k=k, dim=-1, largest=False

        ).indices

        adjacency = torch.zeros(

            batch_size, n_nodes, n_nodes,

            dtype=coords.dtype, device=coords.device

        )

        adjacency.scatter_(

            2,

            nearest,

            1.0

        )

        # If A connects i to j, also connect j to i.

        adjacency = torch.maximum(

            adjacency, adjacency.transpose(1, 2)

        )

        # No self-edges.

        adjacency = adjacency.masked_fill(eye, 0.0)

        return adjacency

    def forward(self, coords):

        adjacency = self.build_adjacency(coords, k=8)

        h = self.activation(self.conv1(coords, adjacency))

        h = self.dropout(h)

        h = self.activation(self.conv2(h, adjacency))

        h = self.dropout(h)

        h = self.activation(self.conv3(h, adjacency))

        # Graph-level readout.

        mean_pool = h.mean(dim=1)

        max_pool = h.max(dim=1).values

        graph_embedding = torch.cat(

            [mean_pool, max_pool], dim=1

        )

        return self.regressor(graph_embedding)

def main():

    parser = argparse.ArgumentParser(

        description="Train a coordinate-only spatial GNN."

    )

    parser.add_argument("--data", default="data/raw/WEC_Perth_49.csv")

    parser.add_argument("--target", default=None)

    parser.add_argument("--epochs", type=int, default=200)

    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--learning-rate", type=float, default=1e-3)

    parser.add_argument("--patience", type=int, default=25)

    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Fixed seed for layout-grouped train/validation/test splits.")

    parser.add_argument("--output-dir", default="results/gnn")

    parser.add_argument("--model-dir", default="models/gnn")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    df, X, y, coord_cols, target_col = load_wave_data(

        args.data, args.target

    )

    X = np.asarray(X, dtype=np.float32)

    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

    n_nodes = X.shape[1] // 2

    X_coords = X.reshape(len(X), n_nodes, 2)

    # Identical coordinate layouts must remain in one split.

    groups = pd.util.hash_pandas_object(

        df[coord_cols], index=False

    ).to_numpy()

    split1 = GroupShuffleSplit(

        n_splits=1, test_size=0.15, random_state=args.split_seed

    )

    trainval_idx, test_idx = next(

        split1.split(X, y.ravel(), groups=groups)

    )

    split2 = GroupShuffleSplit(

        n_splits=1, test_size=0.1765, random_state=args.split_seed

    )

    train_rel, val_rel = next(

        split2.split(

            X[trainval_idx],

            y[trainval_idx].ravel(),

            groups=groups[trainval_idx],

        )

    )

    train_idx = trainval_idx[train_rel]

    val_idx = trainval_idx[val_rel]

    # Use one shared scale for both coordinate axes.

    # Uniform scaling preserves Euclidean distances and neighbour rankings.

    train_coords_flat = X_coords[train_idx].reshape(-1, 2)

    coord_mean = float(train_coords_flat.mean())

    coord_scale = float(train_coords_flat.std())

    coord_scale = max(coord_scale, 1e-8)

    X_coords_scaled = (

        (X_coords - coord_mean) / coord_scale

    ).astype(np.float32)

    # Scale targets using training records only.

    y_scaler = StandardScaler()

    y_train_scaled = y_scaler.fit_transform(

        y[train_idx]

    ).astype(np.float32)

    y_val_scaled = y_scaler.transform(

        y[val_idx]

    ).astype(np.float32)

    X_train_t = torch.from_numpy(

        X_coords_scaled[train_idx]

    )

    X_val_t = torch.from_numpy(

        X_coords_scaled[val_idx]

    )

    X_test_t = torch.from_numpy(

        X_coords_scaled[test_idx]

    )

    y_train_t = torch.from_numpy(y_train_scaled)

    y_val_t = torch.from_numpy(y_val_scaled)

    train_loader = DataLoader(

        TensorDataset(X_train_t, y_train_t),

        batch_size=args.batch_size,

        shuffle=True,

    )

    X_val_t = X_val_t.to(device)

    model = SpatialGNN().to(device)

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

    run_tag = f"{tag}_gnn_seed{args.seed}"
    checkpoint_path = model_dir / f"{run_tag}_best.pt"

    best_val_mae = float("inf")

    best_epoch = 0

    patience_counter = 0

    history = []

    print(f"Dataset: {args.data}")

    print(f"Rows: {len(df):,}")

    print(f"Unique layouts: {len(np.unique(groups)):,}")

    print(f"WEC count / graph nodes: {n_nodes}")

    print("Node features: x and y coordinates")

    print("Graph: symmetrized 8-nearest-neighbour connections")

    print(f"Train/validation/test rows: "

          f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    for epoch in range(1, args.epochs + 1):

        model.train()

        batch_losses = []

        for xb, yb in train_loader:

            xb = xb.to(device)

            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            pred = model(xb)

            loss = loss_fn(pred, yb)

            loss.backward()

            optimizer.step()

            batch_losses.append(loss.item())

        model.eval()

        with torch.no_grad():

            val_pred_scaled = []

            for start in range(0, len(X_val_t), args.batch_size):

                xb = X_val_t[start:start + args.batch_size]

                val_pred_scaled.append(model(xb).cpu().numpy())

            val_pred_scaled = np.concatenate(val_pred_scaled)

        val_pred = y_scaler.inverse_transform(

            val_pred_scaled

        ).ravel()

        val_true = y[val_idx].ravel()

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

                "n_nodes": n_nodes,

                "k_nearest_neighbours": 8,

                "hidden_dim": 64,

                "coordinate_columns": coord_cols,

                "target_column": target_col,

                "coordinate_mean": coord_mean,

                "coordinate_scale": coord_scale,

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

                f"val MAE={val_mae:.6g} | best={best_val_mae:.6g}"

            )

        if patience_counter >= args.patience:

            print(

                f"Early stopping at epoch {epoch}; "

                f"best epoch was {best_epoch}."

            )

            break

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    X_test_t = X_test_t.to(device)

    test_pred_scaled = []

    with torch.no_grad():

        # Warm-up before inference timing.

        _ = model(X_test_t[:min(args.batch_size, len(X_test_t))])

        if device.type == "cuda":

            torch.cuda.synchronize()

        start = time.perf_counter()

        for start_idx in range(0, len(X_test_t), args.batch_size):

            xb = X_test_t[start_idx:start_idx + args.batch_size]

            test_pred_scaled.append(model(xb).cpu().numpy())

        if device.type == "cuda":

            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    test_pred_scaled = np.concatenate(test_pred_scaled)

    y_pred = y_scaler.inverse_transform(test_pred_scaled).ravel()

    y_true = y[test_idx].ravel()

    mae = float(mean_absolute_error(y_true, y_pred))

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    mean_abs_target = float(np.mean(np.abs(y_true)))

    metrics = {

        "dataset": str(args.data),

        "rows": int(len(df)),

        "unique_layouts": int(len(np.unique(groups))),

        "wec_count": int(n_nodes),

        "target_column": target_col,

        "model": "coordinate_graph_neural_network",

        "graph": "symmetrized_8_nearest_neighbours",

        "split_method": "layout_grouped",

        "train_rows": int(len(train_idx)),

        "validation_rows": int(len(val_idx)),

        "test_rows": int(len(test_idx)),

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

        "test_r2": float(r2_score(y_true, y_pred)),

        "test_inference_total_seconds": float(elapsed),

        "test_inference_ms_per_sample": float(

            elapsed * 1000 / len(test_idx)

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

    plt.title("GNN: layout-grouped test")

    plt.tight_layout()

    plt.savefig(

        output_dir / f"{run_tag}_actual_vs_predicted.png", dpi=180

    )

    plt.close()

    print("\nGNN TEST RESULTS")

    for key, value in metrics.items():

        print(f"{key}: {value}")

    print(f"\nResults saved to: {output_dir.resolve()}")

    print(f"Checkpoint saved to: {checkpoint_path.resolve()}")

if __name__ == "__main__":

    main()

