import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_SITE, FIGURES_DIR, OUTPUT_DIR, BATTERY_SPECS
from src.data_pipeline import build_harmonized_dataset
from src.feature_engineering import prepare_feature_sets
from src.models_weather import WeatherForecaster
from src.models_solar import SolarGenerationForecasterXGB, SolarGenerationForecasterDeep
from src.optimizer import find_optimal_tilt_angle, analyze_performance_and_losses, simulate_predictive_battery_dispatch
from src.evaluate import (
    compute_metrics,
    compute_skill_score,
    plot_forecast_timeseries,
    plot_diurnal_profiles,
    plot_parity_scatter,
    plot_optimal_tilt_curve,
    plot_battery_dispatch,
    plot_feature_importance
)

def run(site_key=DEFAULT_SITE, train_deep=True, sample_days=14):
    print("=" * 80)
    print(f"  SOLAR PANEL WEATHER FORECASTING & OPTIMAL ENERGY OUTPUT PIPELINE")
    print(f"  Target: Site {site_key}")
    print("=" * 80)

    # 1. Load & Harmonize Data
    df_raw, site_meta = build_harmonized_dataset(site_key=site_key, use_cache=True)
    print(f"Campus: {site_meta['campus_name']} | Capacity: {site_meta['kwp']} kWp | Inverter: {site_meta['inverter']}")

    # 2. Feature Engineering & Chronological Split
    split_data = prepare_feature_sets(df_raw, test_ratio=0.20)
    train_df = split_data["train_df"]
    test_df = split_data["test_df"]
    weather_feats = split_data["weather_feature_cols"]
    generation_feats = split_data["generation_feature_cols"]

    # 3. Stage 1: Solar Weather Forecaster (GHI & Cloud Opacity)
    print("\n--- STAGE 1: Solar Weather & Irradiance Forecasting ---")
    weather_model = WeatherForecaster(n_estimators=200, max_depth=6, learning_rate=0.06)
    weather_model.train(train_df, weather_feats)
    weather_preds = weather_model.predict(test_df)
    persist_weather = weather_model.evaluate_persistence(test_df)

    # Evaluate GHI
    ghi_metrics_model = compute_metrics(test_df["Ghi"].values, weather_preds["pred_ghi"])
    ghi_metrics_persist = compute_metrics(test_df["Ghi"].values, persist_weather["ghi_persist"])
    ghi_skill = compute_skill_score(ghi_metrics_model["RMSE"], ghi_metrics_persist["RMSE"])

    print(f"  GHI Forecast (XGBoost) -> RMSE: {ghi_metrics_model['RMSE']:.1f} W/m², MAE: {ghi_metrics_model['MAE']:.1f} W/m², R²: {ghi_metrics_model['R2']:.3f}")
    print(f"  GHI Persistence        -> RMSE: {ghi_metrics_persist['RMSE']:.1f} W/m², R²: {ghi_metrics_persist['R2']:.3f}")
    print(f"  GHI Skill Score        -> {ghi_skill:+.1f}% improvement over persistence")

    # 4. Stage 2: Solar Generation Forecaster (Physics-Augmented XGBoost)
    print("\n--- STAGE 2: Solar Energy Generation Forecasting ---")
    # Inverter rating estimate
    inverter_kw = site_meta["kwp"] * 0.85
    solar_xgb = SolarGenerationForecasterXGB(
        rated_kwp=site_meta["kwp"],
        inverter_limit_kw=inverter_kw,
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05
    )
    solar_xgb.train(train_df, generation_feats)
    pred_kwh_xgb = solar_xgb.predict(test_df)

    # Persistence baseline for generation
    persist_gen = test_df["energy_kwh_lag_24"].values
    gen_metrics_xgb = compute_metrics(test_df["energy_kwh"].values, pred_kwh_xgb, capacity=site_meta["kwp"])
    gen_metrics_persist = compute_metrics(test_df["energy_kwh"].values, persist_gen, capacity=site_meta["kwp"])
    gen_skill_xgb = compute_skill_score(gen_metrics_xgb["RMSE"], gen_metrics_persist["RMSE"])

    print(f"  Solar XGBoost Forecaster -> RMSE: {gen_metrics_xgb['RMSE']:.2f} kWh, MAE: {gen_metrics_xgb['MAE']:.2f} kWh, nRMSE: {gen_metrics_xgb['nRMSE_pct']:.2f}%, R²: {gen_metrics_xgb['R2']:.3f}")
    print(f"  24h Persistence Baseline -> RMSE: {gen_metrics_persist['RMSE']:.2f} kWh, R²: {gen_metrics_persist['R2']:.3f}")
    print(f"  Skill Score over Base    -> {gen_skill_xgb:+.1f}% improvement")

    # Optional Deep Learning comparison
    deep_metrics = None
    pred_kwh_deep = None
    if train_deep:
        print("\n--- BENCHMARK: Deep Learning Bi-GRU Sequence Forecaster ---")
        deep_model = SolarGenerationForecasterDeep(seq_len=24, hidden_dim=64, num_layers=2, epochs=12, batch_size=128)
        deep_model.train(train_df, generation_feats)
        pred_kwh_deep = deep_model.predict(test_df)
        deep_metrics = compute_metrics(test_df["energy_kwh"].values, pred_kwh_deep, capacity=site_meta["kwp"])
        deep_skill = compute_skill_score(deep_metrics["RMSE"], gen_metrics_persist["RMSE"])
        print(f"  Deep Bi-GRU Forecaster   -> RMSE: {deep_metrics['RMSE']:.2f} kWh, MAE: {deep_metrics['MAE']:.2f} kWh, nRMSE: {deep_metrics['nRMSE_pct']:.2f}%, R²: {deep_metrics['R2']:.3f}")
        print(f"  Deep Skill Score         -> {deep_skill:+.1f}% improvement")

    # 5. Optimization Engine
    print("\n--- STAGE 3: Energy Optimization & Decision Engine ---")
    
    # 5a. Optimal Tilt Analysis
    print("Computing optimal panel tilt angle...")
    tilt_analysis = find_optimal_tilt_angle(
        timestamps=test_df["timestamp"],
        ghi=test_df["Ghi"],
        latitude=site_meta["lat"],
        longitude=site_meta["lon"],
        elevation=site_meta["elevation_m"]
    )
    print(f"  Mathematically Optimal Tilt Angle: {tilt_analysis['optimal_tilt_deg']}° (True North)")
    print(f"  Solar Yield with Optimal Tilt:     {tilt_analysis['max_poa_kwh_m2']:.1f} kWh/m² vs Flat: {tilt_analysis['flat_poa_kwh_m2']:.1f} kWh/m²")
    print(f"  Annual Solar Energy Yield Gain:    +{tilt_analysis['optimal_tilt_gain_pct']:.2f}%")

    # 5b. Performance Ratio & Loss Diagnostics
    print("\nCalculating Performance Ratio (PR) & Loss Attribution...")
    loss_analysis = analyze_performance_and_losses(test_df, site_meta["kwp"], inverter_kw)
    print(f"  System Performance Ratio (PR):     {loss_analysis['performance_ratio_pct']:.1f}%")
    print(f"  Thermal Derating Loss:             {loss_analysis['total_temp_loss_kwh']:.1f} kWh")
    print(f"  Cloud / Atmospheric Loss:          {loss_analysis['total_cloud_loss_kwh']:.1f} kWh")
    print(f"  Inverter Clipping Loss:            {loss_analysis['total_clipping_loss_kwh']:.1f} kWh")

    # 5c. Predictive Battery Storage Dispatch
    print("\nSimulating 24-Hour Predictive Battery Dispatch...")
    sample_window = 24 * 3  # 3 days window
    forecast_slice = pred_kwh_xgb[-sample_window:]
    dispatch_results = simulate_predictive_battery_dispatch(forecast_slice, battery_specs=BATTERY_SPECS)
    print(f"  Peak Grid Demand Shaved:           {dispatch_results['peak_shaved_kw']:.1f} kW ({dispatch_results['peak_shaved_pct']:.1f}%)")
    print(f"  Grid Import Reduction:             {dispatch_results['grid_import_reduction_pct']:.1f}%")
    print(f"  Total Clean Solar Dispatched:      {dispatch_results['total_battery_discharged_kwh']:.1f} kWh")

    # 6. Generate Publication-Quality Visualizations
    print("\n--- STAGE 4: Generating Visualizations & Reports ---")
    slice_hours = 24 * sample_days
    eval_slice = test_df.iloc[-slice_hours:].copy()
    eval_slice["actual"] = test_df["energy_kwh"].iloc[-slice_hours:].values
    eval_slice["predicted"] = pred_kwh_xgb[-slice_hours:]
    eval_slice["clearsky"] = test_df["theoretical_dc_power"].iloc[-slice_hours:].values

    fig_ts = plot_forecast_timeseries(
        eval_slice["timestamp"],
        eval_slice["actual"],
        eval_slice["predicted"],
        eval_slice["clearsky"],
        title=f"Solar Energy Forecast vs Actual ({site_meta['campus_name']} - Site {site_key}, {site_meta['kwp']} kWp)"
    )

    fig_diurnal = plot_diurnal_profiles(eval_slice)
    fig_parity = plot_parity_scatter(test_df["energy_kwh"].values, pred_kwh_xgb, gen_metrics_xgb)
    fig_tilt = plot_optimal_tilt_curve(tilt_analysis["tilt_curve"], tilt_analysis["optimal_tilt_deg"], tilt_analysis["optimal_tilt_gain_pct"])
    fig_dispatch = plot_battery_dispatch(dispatch_results["dispatch_df"].iloc[:48], dispatch_results)
    
    importances = solar_xgb.get_feature_importances()
    fig_feat = plot_feature_importance(importances)

    # 7. Summary Report File
    summary_md = os.path.join(OUTPUT_DIR, "summary_report.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(f"# Solar Forecasting & Energy Optimization System Report\n\n")
        f.write(f"**Site:** {site_key} ({site_meta['campus_name']}, Lat: {site_meta['lat']:.2f}, Lon: {site_meta['lon']:.2f})\n")
        f.write(f"**Rated DC Capacity:** {site_meta['kwp']} kWp | **Panel:** {site_meta['panel']} | **Inverter:** {site_meta['inverter']}\n\n")
        f.write(f"## 1. Forecasting Performance Metrics\n\n")
        f.write(f"| Model | RMSE (kWh) | MAE (kWh) | nRMSE (%) | R² Score | Skill Score vs Baseline |\n")
        f.write(f"| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Physics-Augmented XGBoost** | **{gen_metrics_xgb['RMSE']:.2f}** | **{gen_metrics_xgb['MAE']:.2f}** | **{gen_metrics_xgb['nRMSE_pct']:.2f}%** | **{gen_metrics_xgb['R2']:.3f}** | **{gen_skill_xgb:+.1f}%** |\n")
        if deep_metrics:
            f.write(f"| Deep Learning Bi-GRU | {deep_metrics['RMSE']:.2f} | {deep_metrics['MAE']:.2f} | {deep_metrics['nRMSE_pct']:.2f}% | {deep_metrics['R2']:.3f} | {deep_skill:+.1f}% |\n")
        f.write(f"| 24h Persistence Baseline | {gen_metrics_persist['RMSE']:.2f} | {gen_metrics_persist['MAE']:.2f} | {gen_metrics_persist['nRMSE_pct']:.2f}% | {gen_metrics_persist['R2']:.3f} | 0.0% |\n\n")
        f.write(f"## 2. Energy Optimization Insights\n\n")
        f.write(f"- **Optimal Tilt Angle:** `{tilt_analysis['optimal_tilt_deg']}°` True North (Yield Gain: **+{tilt_analysis['optimal_tilt_gain_pct']:.1f}%** over horizontal).\n")
        f.write(f"- **System Performance Ratio (PR):** `{loss_analysis['performance_ratio_pct']:.1f}%`.\n")
        f.write(f"- **Loss Attribution:** Thermal Derating: `{loss_analysis['total_temp_loss_kwh']:.1f} kWh` | Cloud Attenuation: `{loss_analysis['total_cloud_loss_kwh']:.1f} kWh` | Clipping: `{loss_analysis['total_clipping_loss_kwh']:.1f} kWh`.\n")
        f.write(f"- **Predictive Battery Storage:** Peak demand shaved by **{dispatch_results['peak_shaved_kw']:.1f} kW ({dispatch_results['peak_shaved_pct']:.1f}%)**, reducing net grid import by **{dispatch_results['grid_import_reduction_pct']:.1f}%**.\n")
    print(f"Summary report written to {summary_md}")

    print("\n" + "=" * 80)
    print("  PIPELINE EXECUTION COMPLETE SUCCESSFULLY")
    print("=" * 80)
    return {
        "metrics_xgb": gen_metrics_xgb,
        "metrics_deep": deep_metrics,
        "tilt_analysis": tilt_analysis,
        "loss_analysis": loss_analysis,
        "dispatch_results": dispatch_results
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Solar Weather Forecasting & Optimization Pipeline")
    parser.add_argument("--site", type=int, default=DEFAULT_SITE, help="Target SiteKey (e.g. 14, 27, 1)")
    parser.add_argument("--deep", action="store_true", default=True, help="Train Deep Learning Bi-GRU comparison")
    parser.add_argument("--no-deep", dest="deep", action="store_false", help="Skip Deep Learning Bi-GRU")
    parser.add_argument("--days", type=int, default=14, help="Days to visualize in time-series plot")
    args = parser.parse_args()
    
    run(site_key=args.site, train_deep=args.deep, sample_days=args.days)
