"""
Feature Engineering, Chronological Splitting, Leak-Free Normalization,
and 3D Sequence Windowing Pipeline.
Configured for:
- D = 13 features (11 physical/weather telemetry + 2 diurnal cyclical encodings)
- H = 4 forecast horizon (15, 30, 45, 60 minutes ahead = 1 hour ahead)
- Dual format export (.npz and .npy)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR
from src.utils.logger import setup_logger

logger = setup_logger("feature_pipeline")

# Exactly 13 input features:
# Autoregressive power + 3 ground sensors + 7 NASA feeds + 2 diurnal encodings
FEATURE_COLS = [
    "total_ac_power",          # 1: Historical generation (kW)
    "sensor_irradiation",      # 2: Ground pyranometer (kW/m^2)
    "sensor_ambient_temp",     # 3: Ground ambient temp (°C)
    "sensor_module_temp",      # 4: PV panel temperature (°C)
    "nasa_ghi",                # 5: NASA satellite GHI (W/m^2)
    "nasa_cloud_amt",          # 6: NASA Daylight Cloud Coverage Index (%)
    "nasa_ambient_temp",       # 7: NASA 2m air temp (°C)
    "nasa_rh",                 # 8: NASA Relative Humidity (%)
    "nasa_wind_speed",         # 9: NASA Wind Speed (m/s)
    "nasa_pressure",           # 10: NASA Surface Pressure (kPa)
    "nasa_clearness_index",    # 11: NASA Clearness Index (Kt)
    "hour_sin",                # 12: Diurnal cyclical sine encoding
    "hour_cos",                # 13: Diurnal cyclical cosine encoding
]

TARGET_COL = "total_ac_power"


def repair_anomalies(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Smooths anomalous telemetry spikes using rolling linear interpolation
    to preserve uniform delta t = 15 min sequence continuity.
    """
    df_clean = df.copy()
    if "is_anomaly" not in df_clean.columns:
        return df_clean

    anomaly_mask = df_clean["is_anomaly"] == 1
    num_anomalies = anomaly_mask.sum()

    if num_anomalies > 0:
        logger.info(f"Smoothing {num_anomalies} flagged anomalies across continuous series...")
        for col in feature_cols:
            if col in df_clean.columns:
                s = df_clean[col].copy()
                s[anomaly_mask] = np.nan
                df_clean[col] = s.interpolate(method="linear", limit_direction="both")

    return df_clean


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds diurnal cyclical trigonometric encodings for hour-of-day.
    Enables neural networks to comprehend 24-hour diurnal patterns continuously.
    """
    df_feat = df.copy()
    dt = pd.to_datetime(df_feat["timestamp"])

    # Continuous hour of day (0.0 to 23.75)
    hour_float = dt.dt.hour + dt.dt.minute / 60.0
    df_feat["hour_sin"] = np.sin(2.0 * np.pi * hour_float / 24.0)
    df_feat["hour_cos"] = np.cos(2.0 * np.pi * hour_float / 24.0)

    return df_feat


def chronological_split(
    df: pd.DataFrame,
    train_days: int = 24,
    val_days: int = 5,
    test_days: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strictly chronological time-series partition (34 days total):
    - Train: First 24 days (May 15 00:00 to June 7 23:45) -> 2,304 intervals
    - Val: Next 5 days (June 8 00:00 to June 12 23:45) -> 480 intervals
    - Test: Last 5 days (June 13 00:00 to June 17 23:45) -> 480 intervals
    """
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    intervals_per_day = 96  # 24 hours * 4 intervals/hour

    train_end_idx = train_days * intervals_per_day
    val_end_idx = train_end_idx + (val_days * intervals_per_day)

    train_df = df_sorted.iloc[:train_end_idx].copy()
    val_df = df_sorted.iloc[train_end_idx:val_end_idx].copy()
    test_df = df_sorted.iloc[val_end_idx:].copy()

    logger.info(
        f"Chronological split complete: "
        f"Train={len(train_df):,} ({train_df['timestamp'].min()} to {train_df['timestamp'].max()}), "
        f"Val={len(val_df):,} ({val_df['timestamp'].min()} to {val_df['timestamp'].max()}), "
        f"Test={len(test_df):,} ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})"
    )

    return train_df, val_df, test_df


class LeakFreeMinMaxScaler:
    """
    MinMax Scaler fitted strictly on the training partition to prevent data leakage.
    Provides JSON-serializable parameters for unscaling predictions in any framework.
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.min_vals = None
        self.max_vals = None
        self.columns = None
        self.diff = None

    def fit(self, df: pd.DataFrame, columns: List[str]):
        self.columns = columns
        data = df[columns].values
        self.min_vals = np.nanmin(data, axis=0)
        self.max_vals = np.nanmax(data, axis=0)
        diff = self.max_vals - self.min_vals
        diff[diff == 0.0] = 1.0
        self.diff = diff
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        data = df[self.columns].values
        scaled = (data - self.min_vals) / self.diff
        scaled = scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return scaled

    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        unscaled = (scaled_data - self.feature_range[0]) / (
            self.feature_range[1] - self.feature_range[0]
        )
        unscaled = unscaled * self.diff + self.min_vals
        return unscaled

    def to_dict(self) -> dict:
        return {
            "columns": self.columns,
            "min_vals": self.min_vals.tolist(),
            "max_vals": self.max_vals.tolist(),
            "diff": self.diff.tolist(),
            "feature_range": list(self.feature_range),
        }


def create_sliding_windows(
    X_series: np.ndarray,
    y_series: np.ndarray,
    lookback_window: int = 96,
    forecast_horizon: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts 2D continuous arrays into 3D supervised sequence tensors:
    X: (num_samples, lookback_window, num_features)
    y: (num_samples, forecast_horizon)
    """
    n_samples = len(X_series) - lookback_window - forecast_horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Series length {len(X_series)} is too short for lookback={lookback_window} "
            f"and horizon={forecast_horizon}."
        )

    X_windows = []
    y_windows = []

    for i in range(n_samples):
        X_window = X_series[i : i + lookback_window]
        y_window = y_series[i + lookback_window : i + lookback_window + forecast_horizon]
        X_windows.append(X_window)
        y_windows.append(y_window)

    return np.array(X_windows, dtype=np.float32), np.array(y_windows, dtype=np.float32)


def generate_partition_windows(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
    lookback_window: int = 96,
    forecast_horizon: int = 4,
) -> Dict[str, np.ndarray]:
    """
    Fits leak-free scalers on train split, scales all splits, and constructs 3D sequence tensors.
    Preserves trailing historical context across split boundaries so validation and test
    targets start at the exact boundary timestamps.
    """
    # 1. Fit scalers strictly on train partition
    feature_scaler = LeakFreeMinMaxScaler(feature_range=(0.0, 1.0))
    feature_scaler.fit(train_df, feature_cols)

    target_scaler = LeakFreeMinMaxScaler(feature_range=(0.0, 1.0))
    target_scaler.fit(train_df, [target_col])

    # 2. Scale train data
    X_train_scaled = feature_scaler.transform(train_df)
    y_train_scaled = target_scaler.transform(train_df).flatten()

    X_train_seq, y_train_seq = create_sliding_windows(
        X_train_scaled, y_train_scaled, lookback_window, forecast_horizon
    )

    # 3. Concatenate trailing lookback context for Val and Test
    val_context_df = pd.concat([train_df.iloc[-lookback_window:], val_df], ignore_index=True)
    X_val_scaled = feature_scaler.transform(val_context_df)
    y_val_scaled = target_scaler.transform(val_context_df).flatten()

    X_val_seq, y_val_seq = create_sliding_windows(
        X_val_scaled, y_val_scaled, lookback_window, forecast_horizon
    )

    # Test context: last lookback_window steps of Val + Test
    test_context_df = pd.concat([val_df.iloc[-lookback_window:], test_df], ignore_index=True)
    X_test_scaled = feature_scaler.transform(test_context_df)
    y_test_scaled = target_scaler.transform(test_context_df).flatten()

    X_test_seq, y_test_seq = create_sliding_windows(
        X_test_scaled, y_test_scaled, lookback_window, forecast_horizon
    )

    logger.info(
        f"Generated Sequence Tensors (features={len(feature_cols)}, lookback={lookback_window}, horizon={forecast_horizon}):\n"
        f"  Train: X={X_train_seq.shape}, y={y_train_seq.shape}\n"
        f"  Val:   X={X_val_seq.shape}, y={y_val_seq.shape}\n"
        f"  Test:  X={X_test_seq.shape}, y={y_test_seq.shape}"
    )

    return {
        "X_train": X_train_seq,
        "y_train": y_train_seq,
        "X_val": X_val_seq,
        "y_val": y_val_seq,
        "X_test": X_test_seq,
        "y_test": y_test_seq,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }
