import numpy as np
import pandas as pd
import pvlib
from pvlib import solarposition, clearsky, atmosphere, irradiance

def compute_solar_position(timestamps, latitude, longitude, elevation=0.0):
    """
    Computes solar astronomical coordinates (zenith, elevation, azimuth)
    using pvlib based on localized timestamps.
    """
def _ensure_datetime_index(timestamps):
    if not isinstance(timestamps, pd.DatetimeIndex):
        dti = pd.DatetimeIndex(timestamps)
    else:
        dti = timestamps
    if dti.tz is None:
        return dti.tz_localize("Australia/Melbourne", ambiguous="NaT", nonexistent="shift_forward")
    return dti

def compute_solar_position(timestamps, latitude, longitude, elevation=0.0):
    """
    Computes solar astronomical coordinates (zenith, elevation, azimuth)
    using pvlib based on localized timestamps.
    """
    dti = _ensure_datetime_index(timestamps)

    solpos = solarposition.get_solarposition(
        time=dti,
        latitude=latitude,
        longitude=longitude,
        altitude=elevation
    )
    
    df_solar = pd.DataFrame(index=timestamps.index if hasattr(timestamps, "index") else None)
    df_solar["solar_zenith"] = solpos["zenith"].values
    df_solar["solar_elevation"] = solpos["elevation"].values
    df_solar["solar_azimuth"] = solpos["azimuth"].values
    return df_solar

def compute_clearsky_irradiance(timestamps, latitude, longitude, elevation=0.0):
    """
    Computes theoretical clear-sky Global Horizontal Irradiance (GHI), Direct Normal Irradiance (DNI),
    and Diffuse Horizontal Irradiance (DHI) using pvlib's Location engine.
    """
    dti = _ensure_datetime_index(timestamps)
    loc = pvlib.location.Location(latitude, longitude, tz="Australia/Melbourne", altitude=elevation)
    cs = loc.get_clearsky(dti, model="ineichen")
    
    df_cs = pd.DataFrame(index=timestamps.index if hasattr(timestamps, "index") else None)
    df_cs["clearsky_ghi"] = cs["ghi"].fillna(0.0).clip(lower=0.0).values
    df_cs["clearsky_dni"] = cs["dni"].fillna(0.0).clip(lower=0.0).values
    df_cs["clearsky_dhi"] = cs["dhi"].fillna(0.0).clip(lower=0.0).values
    return df_cs

def compute_clearsky_index(ghi, clearsky_ghi, min_clearsky=50.0):
    """
    Computes clear-sky index Kt = GHI / GHI_clearsky.
    During night or very low sun (clearsky_ghi < min_clearsky), Kt is safely set to 0.0.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.where(clearsky_ghi >= min_clearsky, ghi / clearsky_ghi, 0.0)
    return np.clip(np.nan_to_num(kt, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.5)

def estimate_cell_temperature(air_temperature, wind_speed, ghi):
    """
    Estimates PV cell temperature using the Faiman model:
    T_cell = T_air + GHI / (u_0 + u_1 * wind_speed)
    Default parameters u_0 = 25.0 W/(m2 C), u_1 = 6.84 W/(m2 C)/(m/s)
    """
    ws = np.clip(wind_speed.fillna(1.0).values, 0.0, 30.0)
    ta = air_temperature.fillna(20.0).values
    g = np.clip(ghi.fillna(0.0).values, 0.0, 1400.0)
    
    u_0 = 25.0
    u_1 = 6.84
    t_cell = ta + g / (u_0 + u_1 * ws)
    return t_cell

def compute_theoretical_dc_power(ghi, cell_temp, rated_kwp, gamma=-0.0038):
    """
    Calculates theoretical DC power output (kW):
    P_ideal = rated_kWp * (GHI / 1000 W/m2) * [1 + gamma * (T_cell - 25)]
    Where gamma is the temperature coefficient (typically -0.38% / C for crystalline silicon).
    """
    derating = 1.0 + gamma * (cell_temp - 25.0)
    derating = np.clip(derating, 0.5, 1.1)
    p_ideal = rated_kwp * (ghi / 1000.0) * derating
    return np.clip(p_ideal, 0.0, rated_kwp * 1.2)

def compute_poa_irradiance(timestamps, ghi, latitude, longitude, elevation, surface_tilt, surface_azimuth=0.0):
    """
    Computes Plane-of-Array (POA) total irradiance for a specified surface tilt and azimuth.
    Surface azimuth: 0 = North (Southern Hemisphere optimal), 180 = South.
    Uses pvlib Erbs decomposition to estimate DNI and DHI from GHI, then translates to tilted surface.
    """
    dti = _ensure_datetime_index(timestamps)
    ghi_series = pd.Series(np.asarray(ghi, dtype=float), index=dti)

    solpos = solarposition.get_solarposition(dti, latitude, longitude, altitude=elevation)
    dni_extra = pvlib.irradiance.get_extra_radiation(dti)
    
    # Decompose GHI using Erbs model
    erbs = pvlib.irradiance.erbs(ghi_series, solpos["zenith"], dti.dayofyear)
    dni = pd.Series(erbs["dni"], index=dti).fillna(0.0).clip(lower=0.0)
    dhi = pd.Series(erbs["dhi"], index=dti).fillna(0.0).clip(lower=0.0)
    
    total_irr = irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=dni,
        ghi=ghi_series,
        dhi=dhi,
        dni_extra=dni_extra,
        model="haydavies"
    )
    return total_irr["poa_global"].fillna(0.0).clip(lower=0.0).values
