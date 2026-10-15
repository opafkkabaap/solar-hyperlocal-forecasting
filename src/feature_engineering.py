import numpy as np
import pandas as pd

def create_cyclical_features(df):
    """
    Computes sine and cosine cyclic transformations for hour-of-day and day-of-year.
    """
    df = df.copy()
    hour = df["timestamp"].dt.hour
    doy = df["timestamp"].dt.dayofyear
    
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["hour"] = hour
    df["month"] = df["timestamp"].dt.month
    return df

def create_lag_and_rolling_features(df):
    """
    Creates historical lag features (1h, 2h, 24h) and rolling window metrics
    (3h, 6h rolling mean/std) for GHI, CloudOpacity, and generation.
    """
    df = df.copy()
    
    # Lagged features
    lag_cols = {
        "Ghi": [1, 2, 24],
        "clearsky_index": [1, 2, 24],
        "CloudOpacity": [1, 2, 24],
        "AirTemperature": [1, 24],
        "RelativeHumidity": [1],
        "WindSpeed": [1],
        "energy_kwh": [1, 2, 24]
    }
    
    for col, lags in lag_cols.items():
        if col in df.columns:
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
                
    # Rolling features over the past windows (shifted by 1 so no current leakage)
    shifted_ghi = df["Ghi"].shift(1)
    df["ghi_rolling_mean_3h"] = shifted_ghi.rolling(3, min_periods=1).mean()
    df["ghi_rolling_std_3h"] = shifted_ghi.rolling(3, min_periods=1).std().fillna(0.0)
    
    shifted_cloud = df["CloudOpacity"].shift(1)
    df["cloud_rolling_mean_3h"] = shifted_cloud.rolling(3, min_periods=1).mean()
    df["cloud_rolling_mean_6h"] = shifted_cloud.rolling(6, min_periods=1).mean()
    
    shifted_kt = df["clearsky_index"].shift(1)
    df["kt_rolling_mean_3h"] = shifted_kt.rolling(3, min_periods=1).mean()
    
    # Drop initial rows with NaNs from longest lag (24 rows)
    df = df.dropna().reset_index(drop=True)
    return df

def prepare_feature_sets(df, test_ratio=0.20):
    """
    Builds full feature sets, selects appropriate feature columns for weather and generation models,
    and performs a chronological train/test split.
    """
    df_feat = create_cyclical_features(df)
    df_feat = create_lag_and_rolling_features(df_feat)
    
    # Weather forecasting features (predicting future GHI / CloudOpacity)
    weather_feature_cols = [
        "solar_zenith", "solar_elevation", "solar_azimuth",
        "clearsky_ghi", "clearsky_dni", "clearsky_dhi",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos", "hour", "month",
        "Ghi_lag_1", "Ghi_lag_2", "Ghi_lag_24",
        "clearsky_index_lag_1", "clearsky_index_lag_2", "clearsky_index_lag_24",
        "CloudOpacity_lag_1", "CloudOpacity_lag_2", "CloudOpacity_lag_24",
        "AirTemperature_lag_1", "AirTemperature_lag_24",
        "RelativeHumidity_lag_1", "WindSpeed_lag_1",
        "ghi_rolling_mean_3h", "ghi_rolling_std_3h",
        "cloud_rolling_mean_3h", "cloud_rolling_mean_6h",
        "kt_rolling_mean_3h"
    ]
    
    # Solar generation forecasting features (predicting energy_kwh)
    generation_feature_cols = weather_feature_cols + [
        "theoretical_dc_power", "cell_temperature",
        "energy_kwh_lag_1", "energy_kwh_lag_2", "energy_kwh_lag_24"
    ]
    
    # Chronological split to prevent temporal leakage
    n = len(df_feat)
    split_idx = int(n * (1.0 - test_ratio))
    
    train_df = df_feat.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df_feat.iloc[split_idx:].copy().reset_index(drop=True)
    
    print(f"Features engineered: {len(df_feat)} rows.")
    print(f"Train split: {len(train_df)} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"Test split:  {len(test_df)} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    
    return {
        "full_df": df_feat,
        "train_df": train_df,
        "test_df": test_df,
        "weather_feature_cols": weather_feature_cols,
        "generation_feature_cols": generation_feature_cols
    }
