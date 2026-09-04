"""
Main Training & Evaluation Script for Transformer Model.
Trains a multi-layer SolarTransformer on multi-modal sequence data (D=13, H=4).
Evaluates on unseen chronological test split and generates empirical metrics & plots.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import BASE_DIR, PROCESSED_DATA_DIR
from model_transformer import SolarTransformer
from src.utils.logger import setup_logger
from train_utils import (
    SequenceDataset,
    compute_metrics,
    load_sequence_data,
    plot_forecast_comparison,
    plot_loss_curves,
    unscale_predictions,
)

logger = setup_logger("transformer_trainer")


def train_transformer(
    plant_key: str = "plant1",
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 0.001,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.2,
    patience: int = 10,
):
    logger.info("=" * 70)
    logger.info(f"Training Solar Transformer for: {plant_key.upper()}")
    logger.info(f"Configuration: D=13 features, Horizon=4 steps (1-hour ahead nowcast)")
    logger.info("=" * 70)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device}")

    # Paths
    tensor_dir = PROCESSED_DATA_DIR / "tensors" / plant_key
    models_dir = BASE_DIR / "models"
    results_dir = BASE_DIR / "results"
    figures_dir = results_dir / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    tensors, scalers = load_sequence_data(tensor_dir)
    X_train, y_train = tensors["X_train"], tensors["y_train"]
    X_val, y_val = tensors["X_val"], tensors["y_val"]
    X_test, y_test = tensors["X_test"], tensors["y_test"]

    logger.info(
        f"Data Loaded Successfully:\n"
        f"  Train: X={X_train.shape}, y={y_train.shape}\n"
        f"  Val:   X={X_val.shape}, y={y_val.shape}\n"
        f"  Test:  X={X_test.shape}, y={y_test.shape}"
    )

    train_loader = DataLoader(
        SequenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SequenceDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # 2. Instantiate Model
    input_dim = X_train.shape[2]
    output_dim = y_train.shape[1]
    model = SolarTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        output_dim=output_dim,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Architecture Initialized ({total_params:,} trainable parameters).")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 3. Training Loop with Early Stopping
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model_path = models_dir / f"transformer_{plant_key}_best.pth"
    epochs_no_improve = 0

    logger.info("\nStarting Training Loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_train_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                val_loss = criterion(preds, batch_y)
                running_val_loss += val_loss.item() * batch_x.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        if epoch % 5 == 0 or epoch == 1 or epoch_val_loss < best_val_loss:
            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | LR: {current_lr:.6f}"
            )

        # Checkpoint saving
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (best val loss: {best_val_loss:.6f}).")
                break

    # 4. Evaluation on Unseen Test Partition
    logger.info("\nLoading best model checkpoint for test evaluation...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    test_preds_list = []
    test_true_list = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            test_preds_list.append(preds.cpu().numpy())
            test_true_list.append(batch_y.numpy())

    y_test_pred_scaled = np.concatenate(test_preds_list, axis=0)
    y_test_true_scaled = np.concatenate(test_true_list, axis=0)

    # Invert scaling to physical kW units
    target_scaler = scalers["target_scaler"]
    y_test_pred_kw = unscale_predictions(y_test_pred_scaled, target_scaler)
    y_test_true_kw = unscale_predictions(y_test_true_scaled, target_scaler)

    # Compute empirical metrics
    metrics = compute_metrics(y_test_true_kw, y_test_pred_kw)
    logger.info("\n" + "=" * 70)
    logger.info("EMPIRICAL TEST SET BENCHMARKS (June 13 to June 17, 2020) — Transformer")
    logger.info("=" * 70)
    overall = metrics["overall"]
    logger.info(f"Overall Horizon RMSE:   {overall['rmse_kw']:.2f} kW")
    logger.info(f"Overall Horizon MAE:    {overall['mae_kw']:.2f} kW")
    logger.info(f"Daytime MAPE (>1% cap): {overall['mape_pct']:.2f}%")
    logger.info(f"Coefficient of Det R^2: {overall['r2_score']:.4f}")
    logger.info(f"Daytime Threshold:      {overall['threshold_kw']:.2f} kW (1% of max {overall['y_max_kw']:.2f} kW)")

    logger.info("\nPer-Step Forecast Horizon Breakdown:")
    for step_name, step_vals in metrics["per_step_horizon"].items():
        logger.info(
            f"  {step_name}: RMSE={step_vals['rmse_kw']:.2f} kW, "
            f"MAE={step_vals['mae_kw']:.2f} kW, "
            f"MAPE={step_vals['mape_pct']:.2f}%, "
            f"R^2={step_vals['r2_score']:.4f}"
        )
    logger.info("=" * 70)

    # 5. Save Plots & Metric Reports
    loss_plot_path = figures_dir / f"transformer_{plant_key}_loss.png"
    forecast_plot_path = figures_dir / f"transformer_{plant_key}_forecast.png"

    plot_loss_curves(train_losses, val_losses, loss_plot_path)
    logger.info(f"Saved loss curve plot to: {loss_plot_path.name}")

    plot_forecast_comparison(
        y_test_true_kw,
        y_test_pred_kw,
        forecast_plot_path,
        title=f"Solar Transformer Multi-Step Forecast vs. Actual — {plant_key.upper()} (June 13–17, 2020)",
    )
    logger.info(f"Saved forecast curve plot to: {forecast_plot_path.name}")

    # Export metrics JSON
    metrics_report = {
        "model": "SolarTransformer",
        "plant": plant_key,
        "parameters": {
            "input_dim": input_dim,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "output_dim": output_dim,
            "dropout": dropout,
            "epochs_trained": len(train_losses),
            "best_val_loss": best_val_loss,
            "total_params": total_params,
        },
        "metrics": metrics,
    }
    metrics_json_path = results_dir / f"transformer_{plant_key}_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)
    logger.info(f"Saved benchmark report to: {metrics_json_path.name}")

    return metrics_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Solar Transformer Model")
    parser.add_argument("--plant", type=str, default="plant1", choices=["plant1", "plant2", "combined"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    args = parser.parse_args()

    train_transformer(
        plant_key=args.plant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
    )
