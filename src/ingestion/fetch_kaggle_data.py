"""
Fetches and validates the Primary Kaggle Solar Power Generation Dataset.
Contains:
- Plant 1 Generation Data & Weather Sensor Data
- Plant 2 Generation Data & Weather Sensor Data
"""

from pathlib import Path
from typing import Dict
import pandas as pd
import requests

from config import KAGGLE_MIRROR_BASE_URL, KAGGLE_FILES, RAW_DATA_DIR
from src.utils.logger import setup_logger

logger = setup_logger("kaggle_fetcher")


def download_file(url: str, dest_path: Path) -> None:
    """Download a remote file with stream chunking."""
    logger.info(f"Downloading from {url} to {dest_path.name}...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
    logger.info(f"Successfully saved {dest_path.name} ({dest_path.stat().st_size:,} bytes).")


def fetch_all_kaggle_datasets(force: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Downloads all 4 Kaggle solar plant CSVs if not present and returns loaded DataFrames.
    """
    datasets: Dict[str, pd.DataFrame] = {}
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename in KAGGLE_FILES:
        file_path = RAW_DATA_DIR / filename
        if not file_path.exists() or force:
            url = f"{KAGGLE_MIRROR_BASE_URL}/{filename}"
            download_file(url, file_path)
        else:
            logger.info(f"Local file already exists: {filename} ({file_path.stat().st_size:,} bytes).")

        df = pd.read_csv(file_path)
        logger.info(f"Loaded {filename}: {len(df):,} rows, columns: {list(df.columns)}")
        datasets[filename] = df

    return datasets


if __name__ == "__main__":
    fetch_all_kaggle_datasets()
