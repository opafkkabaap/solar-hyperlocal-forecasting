"""
Data Sanitization & Anomaly Detection Pipeline.
Implements:
1. Physical range verification (eliminates non-physical readings such as negative irradiance/generation).
2. Unsupervised Isolation Forest outlier detection (flags transient telemetry noise and inverter drift).
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

from config import PHYSICAL_BOUNDS
from src.utils.logger import setup_logger

logger = setup_logger("sanitizer")

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not found; will use statistical MAD outlier detector as fallback.")


def enforce_physical_bounds(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Enforces physical lower and upper bounds on numerical columns.
    Returns cleaned DataFrame and anomaly report dictionary.
    """
    cleaned_df = df.copy()
    report = {}

    # Map column names case-insensitively
    col_mapping = {
        "dc_power": "DC_POWER",
        "ac_power": "AC_POWER",
        "daily_yield": "DAILY_YIELD",
        "irradiation": "GHI",
        "ambient_temperature": "AMBIENT_TEMP",
        "module_temperature": "MODULE_TEMP",
        "nasa_ghi": "GHI",
        "nasa_cloud_amt": "CLOUD_AMT",
        "nasa_ambient_temp": "AMBIENT_TEMP",
        "nasa_rh": "RH2M",
        "nasa_wind_speed": "WS2M",
        "nasa_pressure": "PS",
    }

    for col in cleaned_df.columns:
        norm_key = col_mapping.get(col.lower())
        if norm_key and norm_key in PHYSICAL_BOUNDS:
            low, high = PHYSICAL_BOUNDS[norm_key]
            out_of_bounds = (cleaned_df[col] < low) | (cleaned_df[col] > high)
            violations_count = int(out_of_bounds.sum())
            report[col] = violations_count

            if violations_count > 0:
                logger.info(
                    f"Column '{col}': found {violations_count} values outside [{low}, {high}]. Clipping to bounds."
                )
                cleaned_df[col] = cleaned_df[col].clip(lower=low, upper=high)

    return cleaned_df, report


def detect_anomalies_isolation_forest(
    df: pd.DataFrame,
    features: List[str],
    contamination: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Applies Isolation Forest to multi-variate telemetry features to identify sensor anomalies.
    Adds 'is_anomaly' (0 for normal, 1 for anomaly) and 'anomaly_score' columns.
    """
    sanitized_df = df.copy()
    valid_features = [f for f in features if f in sanitized_df.columns]

    if not valid_features:
        logger.warning("No matching features found for Isolation Forest detection.")
        sanitized_df["is_anomaly"] = 0
        return sanitized_df

    # Extract valid numerical matrix, filling NaNs temporarily with feature medians
    X = sanitized_df[valid_features].copy()
    X = X.fillna(X.median())

    if SKLEARN_AVAILABLE:
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        preds = iso_forest.fit_predict(X)
        scores = iso_forest.decision_function(X)
        # sklearn outputs -1 for anomaly, 1 for inlier
        sanitized_df["is_anomaly"] = (preds == -1).astype(int)
        sanitized_df["anomaly_score"] = scores
    else:
        # Fallback: Multi-variate Robust Z-score via MAD
        med = X.median()
        mad = (X - med).abs().median().replace(0, 1e-6)
        z_scores = ((X - med).abs() / (1.4826 * mad)).max(axis=1)
        sanitized_df["is_anomaly"] = (z_scores > 3.5).astype(int)
        sanitized_df["anomaly_score"] = -z_scores

    anomaly_count = int(sanitized_df["is_anomaly"].sum())
    total_count = len(sanitized_df)
    logger.info(
        f"Anomaly Detection: {anomaly_count}/{total_count} records ({anomaly_count / total_count:.2%}) flagged as anomalies."
    )

    return sanitized_df
