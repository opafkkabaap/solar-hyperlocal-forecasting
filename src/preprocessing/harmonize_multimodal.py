"""
Multi-Modal Temporal Harmonization and Fusion Engine.
Harmonizes multi-rate temporal series:
- Kaggle Ground-Truth Generation (15-minute)
- Kaggle On-Site Weather Sensor Telemetry (15-minute)
- NASA POWER API Meteorological Feeds (1-hour resampled to 15-minute using cubic spline interpolation)
Outputs unified multi-modal DataFrames for model training.
"""

from typing import Tuple, Dict
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from config import TARGET_RESAMPLE_FREQ, PHYSICAL_BOUNDS
from src.preprocessing.clean_and_sanitize import enforce_physical_bounds, detect_anomalies_isolation_forest
from src.utils.logger import setup_logger

logger = setup_logger("harmonizer")


def parse_datetime_flexible(series: pd.Series) -> pd.Series:
    """Parse dates that may have different formats (e.g. DD-MM-YYYY vs YYYY-MM-DD)."""
    try:
        return pd.to_datetime(series, format="%d-%m-%Y %H:%M")
    except Exception:
        try:
            return pd.to_datetime(series, format="%Y-%m-%d %H:%M:%S")
        except Exception:
            return pd.to_datetime(series, infer_datetime_format=True)


def resample_nasa_to_15min(df_nasa: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Interpolates 1-hour NASA weather records to target 15-minute intervals using cubic spline.
    Guarantees physical bounds (e.g. non-negative nocturnal irradiance).
    """
    df = df_nasa.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).set_index("timestamp")

    # Reindex to uniform 15-minute grid encompassing the whole range
    full_15m_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max() + pd.Timedelta(hours=1),
        freq=TARGET_RESAMPLE_FREQ,
    )
    df_reindexed = df.reindex(full_15m_index)

    # Numerical feature columns to interpolate
    feature_cols = [
        "nasa_ghi",
        "nasa_cloud_amt",
        "nasa_ambient_temp",
        "nasa_rh",
        "nasa_wind_speed",
        "nasa_pressure",
        "nasa_clearness_index",
    ]
    interp_cols = [c for c in feature_cols if c in df_reindexed.columns]

    # Perform cubic spline interpolation on each feature
    # x points: seconds from start for observed hourly points
    t_orig = (df.index - df.index[0]).total_seconds().values
    t_target = (full_15m_index - df.index[0]).total_seconds().values

    for col in interp_cols:
        y_orig = df[col].values
        # Filter NaNs if any in original
        valid_mask = ~np.isnan(y_orig)
        if valid_mask.sum() > 3:
            cs = CubicSpline(t_orig[valid_mask], y_orig[valid_mask], bc_type="natural")
            interp_vals = cs(t_target)
            df_reindexed[col] = interp_vals
        else:
            df_reindexed[col] = df_reindexed[col].interpolate(method="time")

    # Re-enforce physical constraints:
    # 1. GHI cannot be negative (spline overshoot prevention)
    if "nasa_ghi" in df_reindexed.columns:
        df_reindexed["nasa_ghi"] = df_reindexed["nasa_ghi"].clip(lower=0.0)
    if "nasa_cloud_amt" in df_reindexed.columns:
        df_reindexed["nasa_cloud_amt"] = df_reindexed["nasa_cloud_amt"].clip(lower=0.0, upper=100.0)
    if "nasa_rh" in df_reindexed.columns:
        df_reindexed["nasa_rh"] = df_reindexed["nasa_rh"].clip(lower=0.0, upper=100.0)
    if "nasa_wind_speed" in df_reindexed.columns:
        df_reindexed["nasa_wind_speed"] = df_reindexed["nasa_wind_speed"].clip(lower=0.0)

    # Forward-fill metadata columns
    for col in ["plant_id", "latitude", "longitude"]:
        if col in df_reindexed.columns:
            df_reindexed[col] = df_reindexed[col].ffill().bfill()

    df_reindexed = df_reindexed.reindex(target_index).reset_index()
    df_reindexed = df_reindexed.rename(columns={"index": "timestamp"})
    return df_reindexed


def harmonize_plant_data(
    df_gen: pd.DataFrame,
    df_weather: pd.DataFrame,
    df_nasa: pd.DataFrame,
    plant_id: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fuses Generation, On-site Weather Sensors, and Resampled NASA POWER weather.
    Returns:
    1. Plant-level aggregated multi-modal dataset (1 row per 15-minute interval)
    2. Inverter-level multi-modal dataset
    """
    logger.info(f"Harmonizing data streams for Plant ID {plant_id}...")

    # 1. Standardize Generation timestamps
    df_g = df_gen.copy()
    df_g["timestamp"] = parse_datetime_flexible(df_g["DATE_TIME"])
    df_g = df_g.sort_values("timestamp")

    # 2. Standardize Weather Sensor timestamps
    df_w = df_weather.copy()
    df_w["timestamp"] = parse_datetime_flexible(df_w["DATE_TIME"])
    df_w = df_w.sort_values("timestamp")

    # Group weather sensors by timestamp (typically 1 sensor per plant, or take mean)
    w_grouped = df_w.groupby("timestamp").agg(
        sensor_ambient_temp=("AMBIENT_TEMPERATURE", "mean"),
        sensor_module_temp=("MODULE_TEMPERATURE", "mean"),
        sensor_irradiation=("IRRADIATION", "mean"),
    ).reset_index()

    # 3. Create Plant-Level Aggregated Generation
    g_agg = df_g.groupby("timestamp").agg(
        total_dc_power=("DC_POWER", "sum"),
        mean_dc_power=("DC_POWER", "mean"),
        total_ac_power=("AC_POWER", "sum"),
        mean_ac_power=("AC_POWER", "mean"),
        total_daily_yield=("DAILY_YIELD", "sum"),
        inverter_count=("SOURCE_KEY", "nunique"),
    ).reset_index()

    # Merge Ground-Truth Generation and Sensor Weather
    plant_merged = pd.merge(g_agg, w_grouped, on="timestamp", how="outer").sort_values("timestamp")
    plant_merged["plant_id"] = plant_id

    # Create uniform 15-minute timeline
    plant_15m_index = pd.date_range(
        start=plant_merged["timestamp"].min(),
        end=plant_merged["timestamp"].max(),
        freq=TARGET_RESAMPLE_FREQ,
    )
    plant_merged = plant_merged.set_index("timestamp").reindex(plant_15m_index).reset_index()
    plant_merged = plant_merged.rename(columns={"index": "timestamp"})

    # Forward fill or interpolate small gaps in sensor telemetry (< 1 hour), then bfill edge values
    sensor_cols = ["sensor_ambient_temp", "sensor_module_temp", "sensor_irradiation"]
    plant_merged[sensor_cols] = plant_merged[sensor_cols].interpolate(method="linear", limit_direction="both")
    # Night hours: generation is naturally 0
    plant_merged["total_dc_power"] = plant_merged["total_dc_power"].fillna(0.0)
    plant_merged["mean_dc_power"] = plant_merged["mean_dc_power"].fillna(0.0)
    plant_merged["total_ac_power"] = plant_merged["total_ac_power"].fillna(0.0)
    plant_merged["mean_ac_power"] = plant_merged["mean_ac_power"].fillna(0.0)
    plant_merged["inverter_count"] = plant_merged["inverter_count"].fillna(0).astype(int)
    plant_merged["total_daily_yield"] = plant_merged["total_daily_yield"].ffill().bfill().fillna(0.0)
    plant_merged["plant_id"] = plant_id

    # 4. Resample NASA POWER data to exact target timestamps
    nasa_15m = resample_nasa_to_15min(df_nasa, target_index=plant_15m_index)

    # 5. Multimodal Fusion
    fused_plant = pd.merge(
        plant_merged,
        nasa_15m[
            [
                "timestamp",
                "nasa_ghi",
                "nasa_cloud_amt",
                "nasa_ambient_temp",
                "nasa_rh",
                "nasa_wind_speed",
                "nasa_pressure",
                "nasa_clearness_index",
            ]
        ],
        on="timestamp",
        how="left",
    )

    # 6. Apply Sanitization Pipeline: Bounds + Isolation Forest
    fused_plant, bounds_report = enforce_physical_bounds(fused_plant)
    anomaly_features = [
        "total_ac_power",
        "total_dc_power",
        "sensor_irradiation",
        "sensor_ambient_temp",
        "nasa_ghi",
        "nasa_ambient_temp",
    ]
    fused_plant = detect_anomalies_isolation_forest(
        fused_plant,
        features=anomaly_features,
        contamination=0.01,
    )

    # 7. Inverter-level Dataset Construction
    fused_inverter = pd.merge(
        df_g,
        w_grouped,
        on="timestamp",
        how="left",
    )
    fused_inverter = pd.merge(
        fused_inverter,
        nasa_15m[
            [
                "timestamp",
                "nasa_ghi",
                "nasa_cloud_amt",
                "nasa_ambient_temp",
                "nasa_rh",
                "nasa_wind_speed",
                "nasa_pressure",
            ]
        ],
        on="timestamp",
        how="left",
    )

    logger.info(
        f"Plant {plant_id} harmonization complete: {len(fused_plant):,} timestamps "
        f"({fused_plant['timestamp'].min()} to {fused_plant['timestamp'].max()})."
    )

    return fused_plant, fused_inverter
