import os
import numpy as np
import pandas as pd
from config import DATA_PATHS, CAMPUS_METADATA, CAMPUS_NAME_TO_KEY, DATA_CACHE_DIR
from src.physics_engine import (
    compute_solar_position,
    compute_clearsky_irradiance,
    compute_clearsky_index,
    estimate_cell_temperature,
    compute_theoretical_dc_power
)

def load_site_metadata(site_key):
    """
    Retrieves metadata for a specific site from Solar_Site_Details.csv.
    """
    df_sites = pd.read_csv(DATA_PATHS["site_details"])
    match = df_sites[df_sites["SiteKey"] == site_key]
    if match.empty:
        raise ValueError(f"SiteKey {site_key} not found in Solar_Site_Details.csv")
    site_info = match.iloc[0].to_dict()
    
    campus_key = int(site_info["CampusKey"])
    campus_meta = CAMPUS_METADATA.get(campus_key, {})
    
    # Fill in lat/lon from campus if missing in site
    lat = site_info["lat"] if pd.notnull(site_info.get("lat")) else campus_meta["lat"]
    lon = site_info["Lon"] if pd.notnull(site_info.get("Lon")) else campus_meta["lon"]
    kwp = float(site_info["kWp"]) if pd.notnull(site_info.get("kWp")) else 50.0  # default estimate if unlisted
    
    return {
        "site_key": site_key,
        "campus_key": campus_key,
        "campus_name": campus_meta.get("name", "Unknown"),
        "lat": lat,
        "lon": lon,
        "elevation_m": campus_meta.get("elevation_m", 100.0),
        "tz": campus_meta.get("tz", "Australia/Melbourne"),
        "kwp": kwp,
        "panel": site_info.get("Panel", "Unknown"),
        "inverter": site_info.get("Inverter", "Unknown")
    }

def get_active_date_range(site_key):
    """
    Queries Monthly_Summary_Solar.csv to determine the continuous active commissioning period.
    """
    df_monthly = pd.read_csv(DATA_PATHS["monthly_summary"])
    df_active = df_monthly[(df_monthly["SiteKey"] == site_key) & (df_monthly["DataStatus"] == True)]
    if df_active.empty:
        return None, None
    
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }
    
    df_active = df_active.copy()
    df_active["month_num"] = df_active["Month"].map(month_map)
    df_active["start_dt"] = pd.to_datetime(dict(year=df_active["Year"], month=df_active["month_num"], day=1))
    
    min_date = df_active["start_dt"].min()
    max_date = df_active["start_dt"].max() + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23, minutes=59)
    return min_date, max_date

def load_irradiance_data(campus_key):
    """
    Loads Solar_Irradiance.csv, converts UTC timestamps to Australia/Melbourne local time.
    """
    df_irr = pd.read_csv(DATA_PATHS["irradiance"])
    df_irr["CampusKey"] = df_irr["Campus"].map(CAMPUS_NAME_TO_KEY)
    df_irr = df_irr[df_irr["CampusKey"] == campus_key].copy()
    
    # Convert UTC to local Melbourne time
    dt_utc = pd.to_datetime(df_irr["Timestamp_UTC"], utc=True)
    df_irr["timestamp"] = dt_utc.dt.tz_convert("Australia/Melbourne").dt.tz_localize(None)
    df_irr = df_irr.sort_values("timestamp").drop_duplicates("timestamp")
    return df_irr[["timestamp", "Ghi", "CloudOpacity"]]

def load_weather_data(campus_key):
    """
    Loads Weather_Data_reordered_all.csv, parses local timestamps, handles small dropouts,
    and aggregates to 1h resolution.
    """
    df_w = pd.read_csv(DATA_PATHS["weather"])
    df_w = df_w[df_w["CampusKey"] == campus_key].copy()
    df_w["timestamp"] = pd.to_datetime(df_w["Timestamp"])
    df_w = df_w.sort_values("timestamp").drop_duplicates("timestamp")
    
    cols_to_resample = [
        "ApparentTemperature", "AirTemperature", "DewPointTemperature",
        "RelativeHumidity", "WindSpeed", "WindDirection"
    ]
    
    # 1h mean aggregation
    df_w_h = df_w.set_index("timestamp")[cols_to_resample].resample("1h").mean()
    # Forward and backward fill minor sensor gaps
    df_w_h = df_w_h.interpolate(method="time", limit=6).bfill().ffill().reset_index()
    return df_w_h

def load_generation_data(site_key):
    """
    Loads Solar_Energy_Generation.csv for the specified site.
    Resamples 15-min generation into 1h hourly total generation (kWh).
    """
    df_gen = pd.read_csv(DATA_PATHS["generation"])
    df_gen_site = df_gen[df_gen["SiteKey"] == site_key].copy()
    df_gen_site["timestamp"] = pd.to_datetime(df_gen_site["Timestamp"])
    df_gen_site = df_gen_site.sort_values("timestamp").drop_duplicates("timestamp")
    
    # Sum 15-min intervals into 1h total kWh
    df_gen_h = df_gen_site.set_index("timestamp")["SolarGeneration"].resample("1h").sum().reset_index()
    df_gen_h.rename(columns={"SolarGeneration": "energy_kwh"}, inplace=True)
    return df_gen_h

def build_harmonized_dataset(site_key=14, use_cache=True):
    """
    End-to-end pipeline to merge Site Details, Irradiance, Weather, and Generation,
    and enrich with pvlib physics variables.
    """
    cache_path = os.path.join(DATA_CACHE_DIR, f"harmonized_site_{site_key}.csv")
    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached harmonized dataset from {cache_path}...")
        df_cached = pd.read_csv(cache_path)
        df_cached["timestamp"] = pd.to_datetime(df_cached["timestamp"])
        return df_cached, load_site_metadata(site_key)

    site_meta = load_site_metadata(site_key)
    campus_key = site_meta["campus_key"]
    print(f"Building harmonized dataset for Site {site_key} ({site_meta['campus_name']}, {site_meta['kwp']} kWp)...")

    # 1. Load data streams
    df_irr = load_irradiance_data(campus_key)
    df_w = load_weather_data(campus_key)
    df_gen = load_generation_data(site_key)

    # 2. Merge on hourly timestamps
    merged = pd.merge(df_irr, df_w, on="timestamp", how="inner")
    merged = pd.merge(merged, df_gen, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # 3. Filter to active commissioning period
    start_dt, end_dt = get_active_date_range(site_key)
    if start_dt is not None:
        merged = merged[(merged["timestamp"] >= start_dt) & (merged["timestamp"] <= end_dt)].copy()

    # 4. Physical astronomical feature calculation via pvlib
    print("Calculating astronomical solar coordinates via pvlib...")
    solpos = compute_solar_position(merged["timestamp"], site_meta["lat"], site_meta["lon"], site_meta["elevation_m"])
    clearsky = compute_clearsky_irradiance(merged["timestamp"], site_meta["lat"], site_meta["lon"], site_meta["elevation_m"])
    
    merged["solar_zenith"] = solpos["solar_zenith"].values
    merged["solar_elevation"] = solpos["solar_elevation"].values
    merged["solar_azimuth"] = solpos["solar_azimuth"].values
    
    merged["clearsky_ghi"] = clearsky["clearsky_ghi"].values
    merged["clearsky_dni"] = clearsky["clearsky_dni"].values
    merged["clearsky_dhi"] = clearsky["clearsky_dhi"].values

    # Physical clamping: at night (solar elevation <= 0 or zenith >= 90), generation and GHI are strictly 0
    night_mask = merged["solar_elevation"] <= 0.0
    merged.loc[night_mask, "energy_kwh"] = 0.0
    merged.loc[night_mask, "Ghi"] = 0.0

    # 5. Clear-sky index Kt
    merged["clearsky_index"] = compute_clearsky_index(merged["Ghi"].values, merged["clearsky_ghi"].values)

    # 6. PV cell temperature
    merged["cell_temperature"] = estimate_cell_temperature(
        merged["AirTemperature"], merged["WindSpeed"], merged["Ghi"]
    )

    # 7. Theoretical clear-sky DC power
    merged["theoretical_dc_power"] = compute_theoretical_dc_power(
        merged["clearsky_ghi"].values, merged["cell_temperature"].values, site_meta["kwp"]
    )

    # Add site identifier columns
    merged["site_key"] = site_key
    merged["campus_key"] = campus_key
    merged["rated_kwp"] = site_meta["kwp"]

    # Cache dataset
    merged.to_csv(cache_path, index=False)
    print(f"Saved harmonized dataset to {cache_path} ({len(merged)} records).")

    return merged, site_meta
