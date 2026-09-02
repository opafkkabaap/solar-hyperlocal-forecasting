"""
Configuration module for the Multi-Modal Solar & Weather Data Collection Pipeline.
Paper: Adaptive Multi-Modal AI Framework for Hyperlocal Weather Forecasting in Distributed Solar Rooftop Systems
Course: DBMS BCSE302L - Digital Assignment 1 (DA-1)
Authors: Revanth P. A. (24BCE1730) & Siripuram Srimannath (24BCE5034)
"""

from pathlib import Path

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Solar Plant Metadata & Representative Coordinates
PLANTS = {
    "plant_1": {
        "id": 4135001,
        "name": "Plant 1 (Pune Region)",
        "latitude": 18.5204,
        "longitude": 73.8567,
        "elevation": 560.0,
        "gen_file": "Plant_1_Generation_Data.csv",
        "weather_file": "Plant_1_Weather_Sensor_Data.csv",
        "nasa_file": "plant_1_nasa_hourly.json",
    },
    "plant_2": {
        "id": 4136001,
        "name": "Plant 2 (Rooftop Site B - Pune Region)",
        "latitude": 18.5204,
        "longitude": 73.8567,
        "elevation": 560.0,
        "gen_file": "Plant_2_Generation_Data.csv",
        "weather_file": "Plant_2_Weather_Sensor_Data.csv",
        "nasa_file": "plant_2_nasa_hourly.json",
    },
}

# Date Range for Ground Truth Dataset (34 Days in 2020)
START_DATE_STR = "20200515"  # May 15, 2020
END_DATE_STR = "20200617"    # June 17, 2020

# Remote URLs for Primary Kaggle Dataset (Verified Public Academic Mirrors)
KAGGLE_MIRROR_BASE_URL = "https://raw.githubusercontent.com/Rakshitha-Ireddi/Solar-Power-Generation-Forecast/main"
KAGGLE_FILES = [
    "Plant_1_Generation_Data.csv",
    "Plant_1_Weather_Sensor_Data.csv",
    "Plant_2_Generation_Data.csv",
    "Plant_2_Weather_Sensor_Data.csv",
]

# NASA POWER API Endpoint and Parameters
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
NASA_POWER_PARAMS = [
    "ALLSKY_SFC_SW_DWN",  # All Sky Surface Shortwave Downward Irradiance (W/m^2)
    "CLOUD_AMT",           # Daylight Cloud Amount (%) -> Satellite Cloud Proxy
    "T2M",                 # Temperature at 2 Meters (°C)
    "RH2M",                # Relative Humidity at 2 Meters (%)
    "WS2M",                # Wind Speed at 2 Meters (m/s)
    "PS",                  # Surface Pressure (kPa)
    "ALLSKY_KT",           # All Sky Insolation Clearness Index (0 to 1)
]
NASA_COMMUNITY = "RE"  # Renewable Energy community

# Physical Range Constraints for Data Sanitization
PHYSICAL_BOUNDS = {
    "GHI": (0.0, 1400.0),            # W/m^2
    "AMBIENT_TEMP": (-5.0, 60.0),    # °C
    "MODULE_TEMP": (-5.0, 90.0),     # °C
    "CLOUD_AMT": (0.0, 100.0),       # %
    "RH2M": (0.0, 100.0),            # %
    "WS2M": (0.0, 45.0),             # m/s
    "PS": (70.0, 110.0),             # kPa
    "DC_POWER": (0.0, 20000.0),      # kW
    "AC_POWER": (0.0, 2000.0),       # kW
    "DAILY_YIELD": (0.0, 15000.0),   # kWh
}

# Target Harmonization Frequency
TARGET_RESAMPLE_FREQ = "15min"
