"""
Fetches and formats Secondary Hyperlocal Meteorological Data from the NASA POWER API.
Endpoints: Hourly temporal point data for solar irradiance, cloud amount, temperature,
humidity, wind speed, surface pressure, and clearness index.
"""

import json
import time
from pathlib import Path
from typing import Dict
import pandas as pd
import requests

from config import (
    NASA_POWER_BASE_URL,
    NASA_POWER_PARAMS,
    NASA_COMMUNITY,
    PLANTS,
    START_DATE_STR,
    END_DATE_STR,
    RAW_DATA_DIR,
)
from src.utils.logger import setup_logger

logger = setup_logger("nasa_fetcher")

PARAM_RENAME = {
    "ALLSKY_SFC_SW_DWN": "nasa_ghi",
    "CLOUD_AMT": "nasa_cloud_amt",
    "T2M": "nasa_ambient_temp",
    "RH2M": "nasa_rh",
    "WS2M": "nasa_wind_speed",
    "PS": "nasa_pressure",
    "ALLSKY_KT": "nasa_clearness_index",
}


def fetch_nasa_for_point(
    lat: float,
    lon: float,
    start_date: str = START_DATE_STR,
    end_date: str = END_DATE_STR,
    max_retries: int = 3,
) -> dict:
    """Query NASA POWER API for a given coordinate and date range with retry logic."""
    params = {
        "parameters": ",".join(NASA_POWER_PARAMS),
        "community": NASA_COMMUNITY,
        "longitude": f"{lon:.4f}",
        "latitude": f"{lat:.4f}",
        "start": start_date,
        "end": end_date,
        "format": "JSON",
    }

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Querying NASA POWER API (lat={lat:.4f}, lon={lon:.4f}, attempt={attempt})...")
            response = requests.get(NASA_POWER_BASE_URL, params=params, timeout=45)
            if response.status_code == 200:
                data = response.json()
                logger.info("NASA POWER API returned HTTP 200 successfully.")
                return data
            else:
                logger.warning(
                    f"NASA POWER API returned HTTP {response.status_code}: {response.text[:200]}"
                )
        except Exception as e:
            logger.warning(f"Request attempt {attempt} failed: {e}")
        time.sleep(2 * attempt)

    raise RuntimeError(f"Failed to fetch data from NASA POWER API after {max_retries} attempts.")


def parse_nasa_json_to_df(data: dict) -> pd.DataFrame:
    """Transform NASA POWER JSON dictionary into an hourly pandas DataFrame."""
    params_dict = data.get("properties", {}).get("parameter", {})
    if not params_dict:
        raise ValueError("NASA POWER response contains no parameter data.")

    # All parameters share the same 'YYYYMMDDHH' timestamp keys
    first_param = list(params_dict.keys())[0]
    date_strs = list(params_dict[first_param].keys())

    records = []
    for d_str in date_strs:
        # Timestamp format in NASA POWER is YYYYMMDDHH (e.g. 2020051500)
        dt = pd.to_datetime(d_str, format="%Y%m%d%H")
        row = {"timestamp": dt}
        for p_code, rename_to in PARAM_RENAME.items():
            if p_code in params_dict:
                val = params_dict[p_code].get(d_str)
                # NASA POWER missing value sentinel is -999.0
                row[rename_to] = None if val == -999.0 else val
        records.append(row)

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_all_nasa_datasets(force: bool = False) -> Dict[str, pd.DataFrame]:
    """Fetch and save NASA POWER hourly data for all configured plants."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    nasa_dfs: Dict[str, pd.DataFrame] = {}

    for plant_key, plant_info in PLANTS.items():
        json_path = RAW_DATA_DIR / plant_info["nasa_file"]
        csv_path = RAW_DATA_DIR / f"{plant_key}_nasa_hourly.csv"

        if not json_path.exists() or force:
            raw_json = fetch_nasa_for_point(
                lat=plant_info["latitude"],
                lon=plant_info["longitude"],
            )
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(raw_json, f, indent=2)
            logger.info(f"Saved raw NASA response to {json_path.name}")
        else:
            logger.info(f"Loading existing NASA JSON from {json_path.name}")
            with open(json_path, "r", encoding="utf-8") as f:
                raw_json = json.load(f)

        df = parse_nasa_json_to_df(raw_json)
        df["plant_id"] = plant_info["id"]
        df["latitude"] = plant_info["latitude"]
        df["longitude"] = plant_info["longitude"]
        df.to_csv(csv_path, index=False)

        logger.info(
            f"Saved structured NASA weather table to {csv_path.name} "
            f"({len(df)} hourly intervals from {df['timestamp'].min()} to {df['timestamp'].max()})."
        )
        nasa_dfs[plant_key] = df

    return nasa_dfs


if __name__ == "__main__":
    fetch_all_nasa_datasets()
