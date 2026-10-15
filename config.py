import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
DATA_CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Dataset paths
DATA_PATHS = {
    "site_details": r"c:\Users\User\Downloads\archive (6)\unisolar\unisolar\Solar_Site_Details.csv",
    "weather": r"c:\Users\User\Downloads\archive (6)\unisolar\unisolar\Weather_Data_reordered_all.csv",
    "monthly_summary": r"c:\Users\User\Downloads\archive (6)\unisolar\unisolar\Monthly_Summary_Solar.csv",
    "generation": r"c:\Users\User\Downloads\archive (6)\unisolar\unisolar\Solar_Energy_Generation.csv",
    "irradiance": r"c:\Users\User\Downloads\archive (6)\unisolar\unisolar\Solar_Irradiance.csv"
}

# Campus geographical metadata
CAMPUS_METADATA = {
    1: {"name": "Bundoora", "lat": -37.71828652, "lon": 145.0509752, "elevation_m": 80.0, "tz": "Australia/Melbourne"},
    2: {"name": "Wodonga", "lat": -36.11120917, "lon": 146.8486788, "elevation_m": 160.0, "tz": "Australia/Melbourne"},
    3: {"name": "Bendigo", "lat": -36.77885832, "lon": 144.3012321, "elevation_m": 220.0, "tz": "Australia/Melbourne"},
    4: {"name": "Mildura", "lat": -34.20528877, "lon": 142.1665560, "elevation_m": 50.0, "tz": "Australia/Melbourne"},
    5: {"name": "Shepparton", "lat": -36.38031889, "lon": 145.4062897, "elevation_m": 110.0, "tz": "Australia/Melbourne"}
}

# Invert name to key for irradiance dataset matching
CAMPUS_NAME_TO_KEY = {v["name"]: k for k, v in CAMPUS_METADATA.items()}

# Default modeling settings
DEFAULT_CAMPUS = 1
DEFAULT_SITE = 14  # Site 14 at Bundoora: 66 kWp Trina 330W panels, 2 x SolarEdge SE27.6K inverters
DEFAULT_HORIZON = 24  # 24 hours ahead
TIME_RESOLUTION = "1h"  # Hourly aggregation for solar weather forecasting and battery dispatch

# Battery storage simulation defaults for optimization engine
BATTERY_SPECS = {
    "capacity_kwh": 100.0,      # 100 kWh storage
    "max_power_kw": 50.0,       # 0.5C rate max charge/discharge
    "round_trip_efficiency": 0.90,  # 90% round-trip efficiency
    "initial_soc_ratio": 0.20,  # Start at 20% SoC
    "min_soc_ratio": 0.10,      # Depth of discharge limit
    "max_soc_ratio": 0.95       # Upper buffer limit
}
