"""
Training and Evaluation Utilities for Multi-Modal Solar Power Forecasting.
Includes:
- DataLoader constructor (.npz and .npy support)
- Inverse scaling to real kW units
- Multi-step evaluation metrics (RMSE, MAE, R^2, and MAPE with data-derived 1% threshold)
- ReduceLROnPlateau & EarlyStopping
- Loss curve and forecast comparison visualization
"""

import json
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    """PyTorch Dataset wrapping multi-modal sequence tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_sequence_data(tensor_dir: Path) -> Tuple[Dict[str, np.ndarray], dict]:
    """Loads precomputed sequence tensors from .npz or .npy files and scalers."""
    npz_file = tensor_dir / "sequences.npz"
    if npz_file.exists():
        with np.load(npz_file) as data:
            tensors = {k: data[k] for k in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]}
    else:
        tensors = {
            "X_train": np.load(tensor_dir / "X_train.npy"),
            "y_train": np.load(tensor_dir / "y_train.npy"),
            "X_val": np.load(tensor_dir / "X_val.npy"),
            "y_val": np.load(tensor_dir / "y_val.npy"),
            "X_test": np.load(tensor_dir / "X_test.npy"),
            "y_test": np.load(tensor_dir / "y_test.npy"),
        }

    with open(tensor_dir / "scalers.json", "r", encoding="utf-8") as f:
        scalers = json.load(f)

    return tensors, scalers


def unscale_predictions(scaled_arr: np.ndarray, target_scaler_dict: dict) -> np.ndarray:
    """Invert MinMax scaling back to physical kW generation units."""
    min_val = np.array(target_scaler_dict["min_vals"])[0]
    diff_val = np.array(target_scaler_dict["diff"])[0]
    f_min = target_scaler_dict["feature_range"][0]
    f_max = target_scaler_dict["feature_range"][1]

    unscaled = (scaled_arr - f_min) / (f_max - f_min)
    unscaled = unscaled * diff_val + min_val
    # Physical clip: generation cannot be negative
    return np.clip(unscaled, a_min=0.0, a_max=None)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes RMSE, MAE, R^2, and MAPE.
    Uses data-derived daytime threshold (y > 0.01 * y_max) for MAPE to prevent near-zero distortions.
    Reports both overall horizon-flattened metrics and per-step breakdown.
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Overall metrics
    rmse = float(np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2)))
    mae = float(np.mean(np.abs(y_true_flat - y_pred_flat)))

    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # Data-derived threshold: 1% of peak capacity
    y_max = float(np.max(y_true_flat))
    threshold = 0.01 * y_max
    daytime_mask = y_true_flat > threshold

    if np.sum(daytime_mask) > 0:
        mape = float(
            100.0
            * np.mean(
                np.abs(y_true_flat[daytime_mask] - y_pred_flat[daytime_mask])
                / y_true_flat[daytime_mask]
            )
        )
    else:
        mape = 0.0

    # Per-step breakdown across the forecast horizon (t+15m, t+30m, t+45m, t+60m)
    horizon_steps = y_true.shape[1] if len(y_true.shape) > 1 else 1
    per_step_metrics = {}

    for h in range(horizon_steps):
        yt_h = y_true[:, h]
        yp_h = y_pred[:, h]
        step_min = (h + 1) * 15

        h_rmse = float(np.sqrt(np.mean((yt_h - yp_h) ** 2)))
        h_mae = float(np.mean(np.abs(yt_h - yp_h)))
        h_ss_res = np.sum((yt_h - yp_h) ** 2)
        h_ss_tot = np.sum((yt_h - np.mean(yt_h)) ** 2)
        h_r2 = float(1.0 - (h_ss_res / h_ss_tot)) if h_ss_tot > 0 else 0.0

        h_mask = yt_h > threshold
        h_mape = (
            float(100.0 * np.mean(np.abs(yt_h[h_mask] - yp_h[h_mask]) / yt_h[h_mask]))
            if np.sum(h_mask) > 0
            else 0.0
        )

        per_step_metrics[f"t+{step_min}min"] = {
            "rmse_kw": round(h_rmse, 3),
            "mae_kw": round(h_mae, 3),
            "mape_pct": round(h_mape, 2),
            "r2_score": round(h_r2, 4),
        }

    return {
        "overall": {
            "rmse_kw": round(rmse, 3),
            "mae_kw": round(mae, 3),
            "mape_pct": round(mape, 2),
            "r2_score": round(r2, 4),
            "threshold_kw": round(threshold, 2),
            "y_max_kw": round(y_max, 2),
        },
        "per_step_horizon": per_step_metrics,
    }


def plot_loss_curves(train_losses: list, val_losses: list, save_path: Path):
    """Plot training and validation loss curves over epochs."""
    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(train_losses, label="Train Loss (MSE)", color="#2563EB", linewidth=2)
    plt.plot(val_losses, label="Validation Loss (MSE)", color="#EA580C", linewidth=2)
    plt.title("Solar LSTM — Training & Validation Loss Curve", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss (MSE)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_forecast_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    title: str = "LSTM 1-Hour Ahead Forecast vs. Actual Generation (5-Day Test Partition)",
):
    """Plots actual vs predicted power generation across the test partition."""
    # Plot first horizon step (t+15m) and fourth horizon step (t+60m)
    plt.figure(figsize=(13, 6), dpi=150)

    # Plot actual generation
    plt.plot(y_true[:, 0], label="Actual AC Generation (kW)", color="#0F172A", linewidth=2, alpha=0.9)
    # Plot 15-min ahead forecast
    plt.plot(
        y_pred[:, 0],
        label="Predicted AC Power (t+15min)",
        color="#2563EB",
        linestyle="--",
        linewidth=1.8,
    )
    # Plot 60-min ahead forecast
    if y_pred.shape[1] >= 4:
        plt.plot(
            y_pred[:, 3],
            label="Predicted AC Power (t+60min)",
            color="#DC2626",
            linestyle=":",
            linewidth=1.8,
        )

    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("15-Minute Test Intervals (June 13 to June 17, 2020)", fontsize=11)
    plt.ylabel("Power Generation (kW)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
