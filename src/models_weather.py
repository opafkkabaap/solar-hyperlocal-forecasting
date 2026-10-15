import numpy as np
import pandas as pd
from xgboost import XGBRegressor

class WeatherForecaster:
    """
    Forecasting model for solar weather variables (Global Horizontal Irradiance GHI and Cloud Opacity)
    combining astronomical clear-sky trajectories and historical weather lags with XGBoost.
    """
    def __init__(self, n_estimators=250, max_depth=6, learning_rate=0.05):
        self.ghi_model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.cloud_model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.feature_cols = None

    def train(self, train_df, feature_cols):
        self.feature_cols = feature_cols
        X_train = train_df[feature_cols]
        y_ghi = train_df["Ghi"]
        y_cloud = train_df["CloudOpacity"]
        
        print(f"Training Weather Forecaster on {len(X_train)} samples with {len(feature_cols)} features...")
        self.ghi_model.fit(X_train, y_ghi)
        self.cloud_model.fit(X_train, y_cloud)
        print("Weather forecaster trained successfully.")

    def predict(self, df):
        X = df[self.feature_cols]
        pred_ghi = self.ghi_model.predict(X)
        pred_cloud = self.cloud_model.predict(X)
        
        # Physics-based post-processing
        # 1. Solar elevation <= 0 means night -> GHI is strictly 0.0
        night_mask = df["solar_elevation"] <= 0.0
        pred_ghi = np.where(night_mask, 0.0, pred_ghi)
        
        # 2. GHI non-negative and capped by physical extraterrestrial/clear-sky upper envelope
        pred_ghi = np.clip(pred_ghi, 0.0, df["clearsky_ghi"].values * 1.3)
        pred_cloud = np.clip(pred_cloud, 0.0, 100.0)
        
        return {
            "pred_ghi": pred_ghi,
            "pred_cloud": pred_cloud
        }

    def evaluate_persistence(self, df):
        """
        Calculates 24-hour persistence baseline (GHI today = GHI same time yesterday).
        """
        ghi_persist = df["Ghi_lag_24"].values
        cloud_persist = df["CloudOpacity_lag_24"].values
        return {
            "ghi_persist": ghi_persist,
            "cloud_persist": cloud_persist
        }
