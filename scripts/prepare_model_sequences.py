"""
Master Sequence Preparation Script for Model Training.
Prepares 3D sequence tensors for LSTM, GRU, and Transformer training.
Supports:
- D = 13 features
- H = 4 forecast horizon (ultra-short-term, 1 hour ahead)
- Dual format export (.npz and .npy)
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR
from src.preprocessing.feature_pipeline import (
    FEATURE_COLS,
    TARGET_COL,
    add_cyclical_time_features,
    chronological_split,
    generate_partition_windows,
    repair_anomalies,
)
from src.utils.logger import setup_logger

logger = setup_logger("sequence_prep")


def process_and_save_sequences(
    plant_key: str,
    csv_filename: str,
    lookback_window: int = 96,
    forecast_horizon: int = 4,
):
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing Sequences for: {plant_key.upper()} (D={len(FEATURE_COLS)}, H={forecast_horizon})")
    logger.info(f"{'='*70}")

    csv_path = PROCESSED_DATA_DIR / csv_filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    logger.info(f"Loaded {csv_filename}: {len(df_raw):,} records.")

    # 1. Feature Engineering: Diurnal cyclical time encodings (hour_sin, hour_cos)
    df_feat = add_cyclical_time_features(df_raw)

    # 2. Repair telemetry anomalies to keep continuous sequence
    df_clean = repair_anomalies(df_feat, FEATURE_COLS)

    # 3. Chronological Train / Val / Test Partition
    train_df, val_df, test_df = chronological_split(
        df_clean, train_days=24, val_days=5, test_days=5
    )

    # 4. Generate 3D Sequence Windows with Leak-Free Normalization
    tensors = generate_partition_windows(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        lookback_window=lookback_window,
        forecast_horizon=forecast_horizon,
    )

    # 5. Sanity Check: Inverse transform precision verification on first step of target
    y_train_sample = tensors["y_train"][:10, 0]  # First horizon step
    y_unscaled = tensors["target_scaler"].inverse_transform(y_train_sample)
    y_actual = train_df[TARGET_COL].values[lookback_window : lookback_window + 10]
    max_recon_err = np.max(np.abs(y_unscaled.flatten() - y_actual))
    logger.info(f"Inverse transform precision check: Max absolute error = {max_recon_err:.2e}")

    # 6. Save Sequence Tensors (.npy and .npz)
    target_dir = PROCESSED_DATA_DIR / "tensors" / plant_key
    target_dir.mkdir(parents=True, exist_ok=True)

    # Individual .npy files
    np.save(target_dir / "X_train.npy", tensors["X_train"])
    np.save(target_dir / "y_train.npy", tensors["y_train"])
    np.save(target_dir / "X_val.npy", tensors["X_val"])
    np.save(target_dir / "y_val.npy", tensors["y_val"])
    np.save(target_dir / "X_test.npy", tensors["X_test"])
    np.save(target_dir / "y_test.npy", tensors["y_test"])

    # Compressed single-file .npz archive
    npz_path = target_dir / "sequences.npz"
    np.savez_compressed(
        npz_path,
        X_train=tensors["X_train"],
        y_train=tensors["y_train"],
        X_val=tensors["X_val"],
        y_val=tensors["y_val"],
        X_test=tensors["X_test"],
        y_test=tensors["y_test"],
    )
    logger.info(f"Saved compressed archive: {npz_path.name} ({npz_path.stat().st_size / (1024*1024):.2f} MB)")

    # Save scalers dictionary
    scalers_dict = {
        "feature_scaler": tensors["feature_scaler"].to_dict(),
        "target_scaler": tensors["target_scaler"].to_dict(),
    }
    with open(target_dir / "scalers.json", "w", encoding="utf-8") as f:
        json.dump(scalers_dict, f, indent=2)

    # Save dataset metadata
    metadata = {
        "plant_key": plant_key,
        "source_file": csv_filename,
        "lookback_window": lookback_window,
        "lookback_hours": lookback_window * 15 / 60.0,
        "forecast_horizon": forecast_horizon,
        "forecast_minutes": forecast_horizon * 15,
        "num_features": len(FEATURE_COLS),
        "feature_columns": FEATURE_COLS,
        "target_column": TARGET_COL,
        "tensor_shapes": {
            "X_train": list(tensors["X_train"].shape),
            "y_train": list(tensors["y_train"].shape),
            "X_val": list(tensors["X_val"].shape),
            "y_val": list(tensors["y_val"].shape),
            "X_test": list(tensors["X_test"].shape),
            "y_test": list(tensors["y_test"].shape),
        },
        "chronological_partitions": {
            "train": {
                "records": len(train_df),
                "start": str(train_df["timestamp"].min()),
                "end": str(train_df["timestamp"].max()),
            },
            "val": {
                "records": len(val_df),
                "start": str(val_df["timestamp"].min()),
                "end": str(val_df["timestamp"].max()),
            },
            "test": {
                "records": len(test_df),
                "start": str(test_df["timestamp"].min()),
                "end": str(test_df["timestamp"].max()),
            },
        },
    }
    with open(target_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Successfully saved all sequence tensors & metadata in: {target_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Prepare sequence window datasets for deep learning.")
    parser.add_argument(
        "--plant",
        type=str,
        default="all",
        choices=["plant1", "plant2", "combined", "all"],
        help="Plant dataset to process.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=96,
        help="Number of 15-minute lookback intervals (default: 96 = 24 hours).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=4,
        help="Number of 15-minute steps ahead to predict (default: 4 = 1 hour ahead).",
    )
    args = parser.parse_args()

    plant_map = {
        "plant1": ("plant1", "plant1_multimodal_15min.csv"),
        "plant2": ("plant2", "plant2_multimodal_15min.csv"),
        "combined": ("combined", "combined_plants_multimodal_15min.csv"),
    }

    targets = list(plant_map.keys()) if args.plant == "all" else [args.plant]

    for t in targets:
        pkey, fname = plant_map[t]
        process_and_save_sequences(
            plant_key=pkey,
            csv_filename=fname,
            lookback_window=args.lookback,
            forecast_horizon=args.horizon,
        )

    logger.info("\nAll model training sequence datasets have been prepared successfully!")


if __name__ == "__main__":
    main()
