"""
Master Data Pipeline Execution Script.
Orchestrates:
1. Primary Kaggle Solar Plant Telemetry Ingestion (Plant 1 & Plant 2)
2. Secondary NASA POWER API Meteorological Ingestion (Plant 1 & Plant 2)
3. Physical Sanitization & Isolation Forest Anomaly Detection
4. Cubic Spline Resampling & Multi-Modal Harmonization (15-min frequency)
5. Export of Model-Ready Datasets & Statistical Verification Summary
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import PLANTS, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.ingestion.fetch_kaggle_data import fetch_all_kaggle_datasets
from src.ingestion.fetch_nasa_power import fetch_all_nasa_datasets
from src.preprocessing.harmonize_multimodal import harmonize_plant_data
from src.utils.logger import setup_logger

logger = setup_logger("pipeline_runner")


def run_pipeline(force_download: bool = False, skip_download: bool = False):
    logger.info("================================================================================")
    logger.info("Starting Multi-Modal Solar & Weather Data Collection Pipeline")
    logger.info("DBMS BCSE302L - Digital Assignment 1 (DA-1)")
    logger.info("Authors: Revanth P. A. (24BCE1730) & Siripuram Srimannath (24BCE5034)")
    logger.info("================================================================================")

    # 1. Primary Data Ingestion: Kaggle Solar Generation Datasets
    if not skip_download:
        logger.info("\n--- Phase 1.1: Fetching Primary Ground-Truth Solar Datasets (Kaggle) ---")
        kaggle_dfs = fetch_all_kaggle_datasets(force=force_download)

        logger.info("\n--- Phase 1.2: Fetching Secondary Hyperlocal Weather Feeds (NASA POWER API) ---")
        nasa_dfs = fetch_all_nasa_datasets(force=force_download)
    else:
        logger.info("Skipping download; loading from local raw directory...")
        kaggle_dfs = {
            "Plant_1_Generation_Data.csv": pd.read_csv(RAW_DATA_DIR / "Plant_1_Generation_Data.csv"),
            "Plant_1_Weather_Sensor_Data.csv": pd.read_csv(RAW_DATA_DIR / "Plant_1_Weather_Sensor_Data.csv"),
            "Plant_2_Generation_Data.csv": pd.read_csv(RAW_DATA_DIR / "Plant_2_Generation_Data.csv"),
            "Plant_2_Weather_Sensor_Data.csv": pd.read_csv(RAW_DATA_DIR / "Plant_2_Weather_Sensor_Data.csv"),
        }
        nasa_dfs = {
            "plant_1": pd.read_csv(RAW_DATA_DIR / "plant_1_nasa_hourly.csv"),
            "plant_2": pd.read_csv(RAW_DATA_DIR / "plant_2_nasa_hourly.csv"),
        }

    # 2. Harmonization & Multimodal Fusion for Plant 1
    logger.info("\n--- Phase 1.3: Harmonizing & Sanitizing Plant 1 (Pune Region) ---")
    plant1_fused, plant1_inverter = harmonize_plant_data(
        df_gen=kaggle_dfs["Plant_1_Generation_Data.csv"],
        df_weather=kaggle_dfs["Plant_1_Weather_Sensor_Data.csv"],
        df_nasa=nasa_dfs["plant_1"],
        plant_id=PLANTS["plant_1"]["id"],
    )

    # 3. Harmonization & Multimodal Fusion for Plant 2
    logger.info("\n--- Phase 1.4: Harmonizing & Sanitizing Plant 2 (Gandikota Region) ---")
    plant2_fused, plant2_inverter = harmonize_plant_data(
        df_gen=kaggle_dfs["Plant_2_Generation_Data.csv"],
        df_weather=kaggle_dfs["Plant_2_Weather_Sensor_Data.csv"],
        df_nasa=nasa_dfs["plant_2"],
        plant_id=PLANTS["plant_2"]["id"],
    )

    # 4. Combine into Cross-Plant Multi-Modal Dataset
    logger.info("\n--- Phase 1.5: Exporting Multi-Modal Training Datasets ---")
    combined_fused = pd.concat([plant1_fused, plant2_fused], ignore_index=True)

    # Save to CSV
    p1_path = PROCESSED_DATA_DIR / "plant1_multimodal_15min.csv"
    p2_path = PROCESSED_DATA_DIR / "plant2_multimodal_15min.csv"
    comb_path = PROCESSED_DATA_DIR / "combined_plants_multimodal_15min.csv"

    plant1_fused.to_csv(p1_path, index=False)
    plant2_fused.to_csv(p2_path, index=False)
    combined_fused.to_csv(comb_path, index=False)

    logger.info(f"Saved Plant 1 Multi-Modal Dataset: {p1_path.name} ({len(plant1_fused):,} records)")
    logger.info(f"Saved Plant 2 Multi-Modal Dataset: {p2_path.name} ({len(plant2_fused):,} records)")
    logger.info(f"Saved Combined Multi-Plant Dataset: {comb_path.name} ({len(combined_fused):,} records)")

    # 5. Generate Data Quality and Descriptive Audit Report
    summary = {
        "metadata": {
            "paper_title": "Adaptive Multi-Modal AI Framework for Hyperlocal Weather Forecasting in Distributed Solar Rooftop Systems",
            "course": "DBMS BCSE302L - Digital Assignment 1 (DA-1)",
            "authors": ["Revanth P. A. (24BCE1730)", "Siripuram Srimannath (24BCE5034)"],
            "resolution": "15-minute intervals",
        },
        "plant_1": {
            "plant_id": PLANTS["plant_1"]["id"],
            "name": PLANTS["plant_1"]["name"],
            "coordinates": {"lat": PLANTS["plant_1"]["latitude"], "lon": PLANTS["plant_1"]["longitude"]},
            "total_records": len(plant1_fused),
            "start_time": str(plant1_fused["timestamp"].min()),
            "end_time": str(plant1_fused["timestamp"].max()),
            "anomalies_flagged": int(plant1_fused["is_anomaly"].sum()),
            "columns": list(plant1_fused.columns),
            "missing_values": plant1_fused.isnull().sum().to_dict(),
        },
        "plant_2": {
            "plant_id": PLANTS["plant_2"]["id"],
            "name": PLANTS["plant_2"]["name"],
            "coordinates": {"lat": PLANTS["plant_2"]["latitude"], "lon": PLANTS["plant_2"]["longitude"]},
            "total_records": len(plant2_fused),
            "start_time": str(plant2_fused["timestamp"].min()),
            "end_time": str(plant2_fused["timestamp"].max()),
            "anomalies_flagged": int(plant2_fused["is_anomaly"].sum()),
            "columns": list(plant2_fused.columns),
            "missing_values": plant2_fused.isnull().sum().to_dict(),
        },
        "combined": {
            "total_records": len(combined_fused),
            "total_columns": len(combined_fused.columns),
        },
    }

    summary_path = PROCESSED_DATA_DIR / "data_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved Data Summary & Verification Audit to: {summary_path.name}")
    logger.info("\n================================================================================")
    logger.info("Multi-Modal Data Pipeline Successfully Completed!")
    logger.info("All files are saved locally on your PC under:")
    logger.info(f"Raw Data: {RAW_DATA_DIR.resolve()}")
    logger.info(f"Processed Data: {PROCESSED_DATA_DIR.resolve()}")
    logger.info("================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Multi-Modal Solar & Weather Data Pipeline")
    parser.add_argument("--force", action="store_true", help="Force re-download of raw files")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading and use existing files")
    args = parser.parse_args()

    run_pipeline(force_download=args.force, skip_download=args.skip_download)
