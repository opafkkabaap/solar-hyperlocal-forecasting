import numpy as np
import pandas as pd
from src.physics_engine import compute_poa_irradiance

def find_optimal_tilt_angle(timestamps, ghi, latitude, longitude, elevation, tilt_range=range(0, 65, 5)):
    """
    Simulates cumulative solar irradiation (kWh/m2) across tilt angles (0 to 60 deg)
    facing True North (azimuth 0 deg) to determine the mathematically optimal tilt.
    """
    results = []
    print(f"Scanning tilt angles from {min(tilt_range)} deg to {max(tilt_range)} deg for lat {latitude:.2f}...")
    
    for tilt in tilt_range:
        poa = compute_poa_irradiance(
            timestamps=timestamps,
            ghi=ghi,
            latitude=latitude,
            longitude=longitude,
            elevation=elevation,
            surface_tilt=tilt,
            surface_azimuth=0.0  # True North for Southern Hemisphere
        )
        total_kwh_m2 = (poa.sum()) / 1000.0  # Wh/m2 to kWh/m2
        results.append({"tilt_deg": tilt, "cumulative_poa_kwh_m2": total_kwh_m2})
        
    df_results = pd.DataFrame(results)
    best_row = df_results.loc[df_results["cumulative_poa_kwh_m2"].idxmax()]
    flat_row = df_results.loc[df_results["tilt_deg"] == 0].iloc[0]
    
    gain_pct = ((best_row["cumulative_poa_kwh_m2"] - flat_row["cumulative_poa_kwh_m2"]) / flat_row["cumulative_poa_kwh_m2"]) * 100.0
    
    return {
        "tilt_curve": df_results,
        "optimal_tilt_deg": int(best_row["tilt_deg"]),
        "max_poa_kwh_m2": float(best_row["cumulative_poa_kwh_m2"]),
        "flat_poa_kwh_m2": float(flat_row["cumulative_poa_kwh_m2"]),
        "optimal_tilt_gain_pct": float(gain_pct)
    }

def analyze_performance_and_losses(df, rated_kwp, inverter_limit_kw):
    """
    Calculates Performance Ratio (PR) and decomposes solar losses into:
    1. Cloud attenuation loss
    2. Temperature thermal derating loss
    3. Inverter clipping loss
    """
    actual_gen = df["energy_kwh"].values
    clearsky_ghi = df["clearsky_ghi"].values
    ghi = df["Ghi"].values
    cell_temp = df["cell_temperature"].values
    
    # 1. Theoretical STC (Standard Test Conditions: 1000 W/m2, 25 C) energy under clear sky
    stc_power_clearsky = rated_kwp * (clearsky_ghi / 1000.0)
    
    # 2. Thermal derating loss (due to T_cell > 25 C)
    thermal_derate_factor = np.clip(1.0 - 0.0038 * (cell_temp - 25.0), 0.5, 1.1)
    temp_derated_clearsky = stc_power_clearsky * thermal_derate_factor
    temp_loss_kwh = np.maximum(0.0, stc_power_clearsky - temp_derated_clearsky)
    
    # 3. Cloud / atmospheric attenuation loss
    cloud_loss_kwh = np.maximum(0.0, (clearsky_ghi - ghi) / 1000.0 * rated_kwp * thermal_derate_factor)
    
    # 4. Inverter clipping loss (potential power exceeding inverter AC capacity)
    potential_ac = rated_kwp * (ghi / 1000.0) * thermal_derate_factor
    clipping_loss_kwh = np.maximum(0.0, potential_ac - inverter_limit_kw)
    
    # Total daytime metrics
    day_mask = df["solar_elevation"] > 5.0
    total_actual = float(np.sum(actual_gen[day_mask]))
    total_theoretical = float(np.sum(stc_power_clearsky[day_mask]))
    pr = (total_actual / total_theoretical) * 100.0 if total_theoretical > 0 else 0.0
    
    return {
        "performance_ratio_pct": pr,
        "total_actual_kwh": total_actual,
        "total_theoretical_stc_kwh": total_theoretical,
        "total_temp_loss_kwh": float(np.sum(temp_loss_kwh[day_mask])),
        "total_cloud_loss_kwh": float(np.sum(cloud_loss_kwh[day_mask])),
        "total_clipping_loss_kwh": float(np.sum(clipping_loss_kwh[day_mask]))
    }

def simulate_predictive_battery_dispatch(forecasted_generation, baseline_load=None, battery_specs=None):
    """
    Simulates optimal predictive battery dispatch over a 24-hour horizon:
    - Uses the 24-hour generation forecast to calculate available stored solar energy.
    - Solves for the optimal peak threshold using binary search to shave the grid peak.
    - Evaluates peak shaving and grid import reduction.
    """
    if battery_specs is None:
        battery_specs = {
            "capacity_kwh": 100.0,
            "max_power_kw": 40.0,
            "round_trip_efficiency": 0.90,
            "initial_soc_ratio": 0.30,
            "min_soc_ratio": 0.10,
            "max_soc_ratio": 0.95
        }
    
    cap = battery_specs["capacity_kwh"]
    max_p = battery_specs["max_power_kw"]
    eff = np.sqrt(battery_specs["round_trip_efficiency"])
    min_soc = cap * battery_specs["min_soc_ratio"]
    max_soc = cap * battery_specs["max_soc_ratio"]
    soc = cap * battery_specs.get("initial_soc_ratio", 0.30)
    
    n_hours = len(forecasted_generation)
    
    # Campus building load scaled to solar array capacity
    if baseline_load is None:
        hours = np.arange(n_hours) % 24
        baseline_load = 15.0 + 15.0 * np.sin(np.pi * np.clip(hours - 7, 0, 15) / 15)
        # Evening demand peak (17:00 - 20:00)
        baseline_load += np.where((hours >= 17) & (hours <= 20), 10.0, 0.0)
    
    battery_charge = np.zeros(n_hours)
    battery_discharge = np.zeros(n_hours)
    battery_soc = np.zeros(n_hours)
    grid_import_before = np.maximum(0.0, baseline_load - forecasted_generation)
    grid_import_after = np.copy(grid_import_before)
    
    # Pass 1: Solar charging during surplus generation
    for t in range(n_hours):
        s = forecasted_generation[t]
        l = baseline_load[t]
        if s > l:
            surplus = s - l
            charge_power = min(surplus, max_p, (max_soc - soc) / eff)
            soc += charge_power * eff
            battery_charge[t] = charge_power
            grid_import_after[t] = 0.0
        battery_soc[t] = soc
        
    # Pass 2: Predictive peak shaving optimization using stored solar
    usable_energy = (soc - min_soc) * eff
    low = 0.0
    high = float(np.max(grid_import_after)) if len(grid_import_after) > 0 else 0.0
    best_thresh = high
    
    for _ in range(35):
        mid = (low + high) / 2.0
        req_energy = np.sum(np.maximum(0.0, grid_import_after - mid))
        if req_energy <= usable_energy:
            best_thresh = mid
            high = mid
        else:
            low = mid
            
    # Apply optimal shaving discharge
    for t in range(n_hours):
        if grid_import_after[t] > best_thresh:
            wanted = grid_import_after[t] - best_thresh
            discharge_power = min(wanted, max_p, (soc - min_soc) * eff)
            soc -= discharge_power / eff
            battery_discharge[t] = discharge_power
            grid_import_after[t] -= discharge_power
        battery_soc[t] = soc
        
    peak_before = float(np.max(grid_import_before))
    peak_after = float(np.max(grid_import_after))
    peak_shaved_kw = peak_before - peak_after
    
    total_grid_before = float(np.sum(grid_import_before))
    total_grid_after = float(np.sum(grid_import_after))
    grid_reduction_pct = ((total_grid_before - total_grid_after) / total_grid_before) * 100.0 if total_grid_before > 0 else 0.0
    
    df_dispatch = pd.DataFrame({
        "hour_step": np.arange(n_hours),
        "solar_forecast_kwh": forecasted_generation,
        "load_kwh": baseline_load,
        "grid_import_before": grid_import_before,
        "battery_charge_kwh": battery_charge,
        "battery_discharge_kwh": battery_discharge,
        "battery_soc_kwh": battery_soc,
        "grid_import_after": grid_import_after
    })
    
    return {
        "dispatch_df": df_dispatch,
        "peak_shaved_kw": peak_shaved_kw,
        "peak_shaved_pct": (peak_shaved_kw / peak_before * 100.0) if peak_before > 0 else 0.0,
        "grid_import_reduction_pct": grid_reduction_pct,
        "total_solar_utilized_kwh": float(np.sum(forecasted_generation)),
        "total_battery_discharged_kwh": float(np.sum(battery_discharge))
    }
