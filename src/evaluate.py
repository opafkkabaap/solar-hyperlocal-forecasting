import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import FIGURES_DIR

# Set style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 300

def compute_metrics(y_true, y_pred, capacity=None):
    """
    Computes regression metrics: MAE, RMSE, nRMSE, and R2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    norm_base = capacity if capacity is not None and capacity > 0 else (np.mean(y_true) + 1e-6)
    nrmse = (rmse / norm_base) * 100.0
    
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "nRMSE_pct": float(nrmse),
        "R2": float(r2)
    }

def compute_skill_score(rmse_model, rmse_baseline):
    """
    Calculates forecast skill score relative to a baseline (e.g. persistence):
    SS = 1 - (RMSE_model / RMSE_baseline)
    """
    if rmse_baseline <= 0:
        return 0.0
    return float((1.0 - (rmse_model / rmse_baseline)) * 100.0)

def plot_forecast_timeseries(timestamps, y_true, y_pred, y_clearsky=None, title="Solar Energy Generation Forecast", save_name="forecast_timeseries.png"):
    """
    Plots actual vs predicted time series with clear-sky envelope.
    """
    plt.figure(figsize=(14, 5.5))
    plt.plot(timestamps, y_true, label="Actual Generation (kWh)", color="#1f77b4", linewidth=1.8, alpha=0.9)
    plt.plot(timestamps, y_pred, label="Physics-Augmented XGBoost (kWh)", color="#ff7f0e", linewidth=1.8, linestyle="--")
    
    if y_clearsky is not None:
        plt.plot(timestamps, y_clearsky, label="Theoretical Clear-Sky STC (kWh)", color="#2ca02c", linewidth=1.2, linestyle=":", alpha=0.7)
        
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Date & Time (Australian Eastern Time)", fontsize=11)
    plt.ylabel("Hourly Energy Output (kWh)", fontsize=11)
    plt.legend(frameon=True, facecolor="white", loc="upper right")
    plt.tight_layout()
    
    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved figure to {save_path}")
    return save_path

def plot_diurnal_profiles(df_eval, save_name="diurnal_profiles.png"):
    """
    Plots mean diurnal profile by hour of day for Actual, Predicted, and Clear-sky.
    """
    df_eval = df_eval.copy()
    df_eval["hour"] = df_eval["timestamp"].dt.hour
    
    hourly = df_eval.groupby("hour")[["actual", "predicted", "clearsky"]].mean()
    
    plt.figure(figsize=(10, 5))
    plt.plot(hourly.index, hourly["actual"], marker="o", label="Actual Generation", color="#1f77b4", linewidth=2.2)
    plt.plot(hourly.index, hourly["predicted"], marker="s", label="Forecasted Generation", color="#ff7f0e", linewidth=2.2, linestyle="--")
    plt.plot(hourly.index, hourly["clearsky"], marker="^", label="Theoretical Clear-Sky", color="#2ca02c", linewidth=1.5, linestyle=":")
    
    plt.title("Diurnal Solar Generation Profile (Average by Hour of Day)", fontsize=13, fontweight="bold")
    plt.xlabel("Hour of Day (00:00 - 23:00)", fontsize=11)
    plt.ylabel("Average Energy Output (kWh)", fontsize=11)
    plt.xticks(range(0, 24, 2))
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    
    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved figure to {save_path}")
    return save_path

def plot_parity_scatter(y_true, y_pred, metrics, save_name="parity_scatter.png"):
    """
    Plots parity scatter plot (Actual vs Predicted) with 1:1 reference line.
    """
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.35, color="#1f77b4", edgecolors="none", s=20)
    
    max_val = max(np.max(y_true), np.max(y_pred)) * 1.05
    plt.plot([0, max_val], [0, max_val], color="red", linestyle="--", linewidth=1.5, label="1:1 Parity Line")
    
    banner = f"R² = {metrics['R2']:.3f}\nRMSE = {metrics['RMSE']:.2f} kWh\nMAE = {metrics['MAE']:.2f} kWh\nnRMSE = {metrics['nRMSE_pct']:.1f}%"
    plt.text(0.05, 0.92, banner, transform=plt.gca().transAxes, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    
    plt.title("Actual vs. Predicted Solar Generation", fontsize=13, fontweight="bold")
    plt.xlabel("Actual Generation (kWh)", fontsize=11)
    plt.ylabel("Forecasted Generation (kWh)", fontsize=11)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved figure to {save_path}")
    return save_path

def plot_optimal_tilt_curve(df_tilt, optimal_tilt, gain_pct, save_name="optimal_tilt_curve.png"):
    """
    Plots cumulative solar irradiation as a function of panel tilt angle.
    """
    plt.figure(figsize=(9, 5))
    plt.plot(df_tilt["tilt_deg"], df_tilt["cumulative_poa_kwh_m2"], marker="o", color="#2ca02c", linewidth=2.2)
    
    # Highlight optimal point
    best_row = df_tilt[df_tilt["tilt_deg"] == optimal_tilt].iloc[0]
    plt.scatter([optimal_tilt], [best_row["cumulative_poa_kwh_m2"]], color="red", s=120, zorder=5, label=f"Optimal Tilt: {optimal_tilt}° (+{gain_pct:.1f}% vs Flat)")
    
    plt.axvline(optimal_tilt, color="red", linestyle=":", alpha=0.7)
    plt.title("Solar Energy Output Optimization vs. Panel Tilt Angle", fontsize=13, fontweight="bold")
    plt.xlabel("Surface Tilt Angle (Degrees from Horizontal)", fontsize=11)
    plt.ylabel("Annual POA Solar Irradiation (kWh/m²)", fontsize=11)
    plt.legend(frameon=True, facecolor="white", loc="lower center")
    plt.tight_layout()
    
    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved figure to {save_path}")
    return save_path

def plot_battery_dispatch(dispatch_df, metrics, save_name="battery_dispatch.png"):
    """
    Plots solar forecast, building load, battery SoC, and grid import shaving.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    
    hours = dispatch_df["hour_step"]
    
    # Top plot: Power flows
    ax1.plot(hours, dispatch_df["solar_forecast_kwh"], label="Forecasted Solar (kWh)", color="#ff7f0e", linewidth=2.2)
    ax1.plot(hours, dispatch_df["load_kwh"], label="Campus Building Load (kWh)", color="#333333", linewidth=1.8, linestyle="--")
    ax1.plot(hours, dispatch_df["grid_import_before"], label="Grid Import (Without Battery)", color="#d62728", linewidth=1.5, alpha=0.7)
    ax1.plot(hours, dispatch_df["grid_import_after"], label="Grid Import (With Predictive Battery)", color="#2ca02c", linewidth=2.2)
    ax1.fill_between(hours, dispatch_df["grid_import_after"], dispatch_df["grid_import_before"], color="#2ca02c", alpha=0.2, label=f"Peak Shaved: {metrics['peak_shaved_kw']:.1f} kW ({metrics['peak_shaved_pct']:.1f}%)")
    
    ax1.set_title("24-Hour Predictive Solar Dispatch & Peak Shaving", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Power / Hourly Energy (kWh)", fontsize=11)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9)
    
    # Bottom plot: Battery SoC and Charge/Discharge
    ax2.plot(hours, dispatch_df["battery_soc_kwh"], label="Battery State-of-Charge (kWh)", color="#9467bd", linewidth=2.2)
    ax2.bar(hours, dispatch_df["battery_charge_kwh"], width=0.6, label="Solar Charging (kWh)", color="#1f77b4", alpha=0.7)
    ax2.bar(hours, -dispatch_df["battery_discharge_kwh"], width=0.6, label="Peak Discharging (kWh)", color="#e377c2", alpha=0.7)
    
    ax2.set_title("Battery State-of-Charge & Charge/Discharge Schedule", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Hour Ahead (t=0 to t=24)", fontsize=11)
    ax2.set_ylabel("Battery Energy (kWh)", fontsize=11)
    ax2.set_xticks(range(0, len(hours), 2))
    ax2.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved figure to {save_path}")
    return save_path

def plot_feature_importance(importance_series, top_n=15, save_name="feature_importance.png"):
    """
    Plots horizontal bar chart of top feature importances.
    """
    top_feats = importance_series.head(top_n).iloc[::-1]
    
    plt.figure(figsize=(9, 6))
    plt.barh(top_feats.index, top_feats.values, color="#1f77b4", edgecolor="#0f4c81")
    plt.title(f"Top {top_n} Feature Importances (XGBoost Solar Model)", fontsize=13, fontweight="bold")
    plt.xlabel("Normalized Gain / Relative Importance", fontsize=11)
    plt.tight_layout()
    
    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved figure to {save_path}")
    return save_path
