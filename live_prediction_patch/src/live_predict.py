from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.train_mlp_grouped import CoordinateMLP
from src.train_spatial_mlp import SpatialFeatureMLP, build_spatial_features
from src.train_gnn import SpatialGNN

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "Perth 49": ("WEC_Perth_49", 49),
    "Perth 100": ("WEC_Perth_100", 100),
    "Sydney 49": ("WEC_Sydney_49", 49),
    "Sydney 100": ("WEC_Sydney_100", 100),
}
MODEL_FOLDERS = {
    "Coordinate MLP": "grouped",
    "Spatial-feature MLP": "spatial",
    "GNN": "gnn",
}


def _checkpoint_path(dataset: str, model_name: str, seed: int) -> Path:
    stem, _ = DATASETS[dataset]
    folder = MODEL_FOLDERS[model_name]
    if model_name == "Coordinate MLP":
        filename = f"{stem}_coordinate_mlp_grouped_seed{seed}.pt"
    elif model_name == "Spatial-feature MLP":
        filename = f"{stem}_spatial_mlp_seed{seed}_best.pt"
    else:
        filename = f"{stem}_gnn_seed{seed}_best.pt"
    return ROOT / "models" / folder / filename


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint not found: {path.relative_to(ROOT)}. "
            "Choose another available training seed."
        )
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint has an unsupported or incomplete format.")
    return checkpoint


def predict_total_power(
    dataset: str,
    model_name: str,
    seed: int,
    coordinates: list[list[float]],
) -> dict[str, Any]:
    """Run one farm-level prediction using a saved training checkpoint."""
    if dataset not in DATASETS:
        raise ValueError("Unsupported dataset configuration.")
    if model_name not in MODEL_FOLDERS:
        raise ValueError("Unsupported model architecture.")
    if seed not in (21, 42, 123):
        raise ValueError("Training seed must be 21, 42 or 123.")

    _, expected_n = DATASETS[dataset]
    arr = np.asarray(coordinates, dtype=np.float64)
    if arr.shape != (expected_n, 2):
        raise ValueError(
            f"{dataset} requires exactly {expected_n} coordinate pairs "
            f"(received shape {arr.shape})."
        )
    if not np.isfinite(arr).all():
        raise ValueError("Coordinates must contain only finite numeric values.")
    if np.unique(arr, axis=0).shape[0] != expected_n:
        raise ValueError("Each WEC must have a unique coordinate pair.")

    checkpoint_path = _checkpoint_path(dataset, model_name, seed)
    ckpt = _load_checkpoint(checkpoint_path)

    if model_name == "Coordinate MLP":
        x = arr.astype(np.float32).reshape(1, -1)
        expected_dim = int(ckpt["input_dim"])
        if x.shape[1] != expected_dim:
            raise ValueError("Coordinate count does not match checkpoint input size.")
        x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
        x_scale = np.asarray(ckpt["x_scale"], dtype=np.float32)
        x_scale = np.where(np.abs(x_scale) < 1e-12, 1.0, x_scale)
        x = (x - x_mean) / x_scale
        model = CoordinateMLP(
            input_dim=expected_dim,
            hidden_dims=tuple(ckpt.get("hidden_dims", [256, 128, 64])),
            dropout=float(ckpt.get("dropout", 0.10)),
        )
        tensor = torch.from_numpy(x)

    elif model_name == "Spatial-feature MLP":
        x = np.asarray(build_spatial_features(arr.reshape(1, -1)), dtype=np.float32)
        if x.shape[1] != int(ckpt["input_dim"]):
            raise ValueError("Spatial feature count does not match checkpoint.")
        x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
        x_scale = np.asarray(ckpt["x_scale"], dtype=np.float32)
        x_scale = np.where(np.abs(x_scale) < 1e-12, 1.0, x_scale)
        x = (x - x_mean) / x_scale
        model = SpatialFeatureMLP(
            input_dim=int(ckpt["input_dim"]),
            hidden_dims=(256, 128, 64),
            dropout=0.10,
        )
        tensor = torch.from_numpy(x)

    else:
        if int(ckpt["n_nodes"]) != expected_n:
            raise ValueError("GNN checkpoint node count does not match dataset.")
        mean = float(ckpt["coordinate_mean"])
        scale = max(float(ckpt["coordinate_scale"]), 1e-8)
        x = ((arr - mean) / scale).astype(np.float32).reshape(1, expected_n, 2)
        model = SpatialGNN(hidden_dim=int(ckpt.get("hidden_dim", 64)), dropout=0.10)
        tensor = torch.from_numpy(x)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        pred_scaled = float(model(tensor).reshape(-1)[0].item())

    y_mean = float(np.asarray(ckpt["y_mean"]).reshape(-1)[0])
    y_scale = float(np.asarray(ckpt["y_scale"]).reshape(-1)[0])
    prediction = pred_scaled * y_scale + y_mean
    if not math.isfinite(prediction):
        raise ValueError("The model returned a non-finite prediction.")

    return {
        "prediction_total_power": float(prediction),
        "target": str(ckpt.get("target_column", "Total_Power")),
        "dataset": dataset,
        "model": model_name,
        "training_seed": seed,
        "wec_count": expected_n,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "coordinate_bounds": {
            "x_min": float(arr[:, 0].min()), "x_max": float(arr[:, 0].max()),
            "y_min": float(arr[:, 1].min()), "y_max": float(arr[:, 1].max()),
        },
        "interpretation": (
            "Model estimate for the supplied coordinate layout; "
            "not a measured value or a validated uncertainty interval."
        ),
    }
