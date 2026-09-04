# Multi-Modal Solar & Hyperlocal Weather Dataset

**Course**: DBMS BCSE302L — Digital Assignment 1 (DA-1)  
**Authors**: Revanth P. A. (24BCE1730) & Siripuram Srimannath (24BCE5034)  
**Paper**: *Adaptive Multi-Modal AI Framework for Hyperlocal Weather Forecasting in Distributed Solar Rooftop Systems*

---

## 1. Overview
This repository implements the multi-modal ingestion, temporal harmonization, and deep learning forecasting framework for distributed solar rooftop systems.
- 📑 **[Read Comprehensive Project Report (REPORT.md)](REPORT.md)**: Full academic report covering system architecture, database schema, data engineering, model formulations, and physics-informed PV equations.
- 📈 **[View Empirical Model Benchmarks & Figures (RESULTS.md)](RESULTS.md)**: Side-by-side performance tables (LSTM vs. GRU), loss curves, and 5-day multi-step forecast comparisons.
- 🚀 Complete dataset, preprocessing scripts, and PyTorch training pipelines are fully reproducible.

---

## 2. Directory Structure

```
solar-hyperlocal-forecasting/
├── config.py                     # Central configuration (coordinates, API endpoints, bounds)
├── requirements.txt              # Pipeline requirements (pandas, numpy, requests, scipy, scikit-learn)
├── README.md                     # Dataset documentation and column dictionary
├── scripts/
│   └── run_data_pipeline.py      # Automated pipeline orchestrator
├── src/
│   ├── ingestion/
│   │   ├── fetch_kaggle_data.py  # Ground truth solar CSV downloader & validator
│   │   └── fetch_nasa_power.py   # NASA POWER API query engine
│   ├── preprocessing/
│   │   ├── clean_and_sanitize.py # Physical bounds check & Isolation Forest outlier detection
│   │   └── harmonize_multimodal.py # Cubic spline 1h->15m interpolation & multi-modal fusion
│   └── utils/
│       └── logger.py             # Formatted logging utility
└── data/
    ├── raw/                      # Unmodified downloaded files
    │   ├── Plant_1_Generation_Data.csv      (4.84 MB)
    │   ├── Plant_1_Weather_Sensor_Data.csv  (288 KB)
    │   ├── Plant_2_Generation_Data.csv      (5.81 MB)
    │   ├── Plant_2_Weather_Sensor_Data.csv  (301 KB)
    │   ├── plant_1_nasa_hourly.csv / .json  (816 hourly records)
    │   └── plant_2_nasa_hourly.csv / .json  (816 hourly records)
    └── processed/                # Fully sanitized, 15-minute aligned multi-modal tables
        ├── plant1_multimodal_15min.csv          (3,264 records, 832 KB)
        ├── plant2_multimodal_15min.csv          (3,264 records, 871 KB)
        ├── combined_plants_multimodal_15min.csv (6,528 records, 1.70 MB)
        └── data_summary.json                    (Quality audit & metadata)
```

---

## 3. Ground Truth & Meteorological Modalities

| Modality | Source | Native Frequency | Harmonized Frequency | Key Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **Solar Generation** | On-site plant inverters | 15 minutes | 15 minutes | DC Power, AC Power, Daily Yield, Total Yield |
| **On-site Weather** | Plant pyranometer & temp sensors | 15 minutes | 15 minutes | Ambient Temp, Module Temp, Irradiation ($kW/m^2$) |
| **Macro/Satellite Weather** | NASA POWER API (MERRA-2 / CERES) | 1 hour | 15 minutes (Cubic Spline) | GHI ($W/m^2$), Cloud Amount (%), 2m Temp (°C), RH (%), Wind Speed ($m/s$), Pressure ($kPa$) |

---

## 4. Feature Column Dictionary

| Column | Unit / Range | Description |
| :--- | :--- | :--- |
| `timestamp` | `YYYY-MM-DD HH:MM:SS` | 15-minute standardized temporal stamp |
| `plant_id` | Integer (`4135001` or `4136001`) | Plant identifier |
| `total_ac_power` | $kW$ | Total AC power generated across all inverters |
| `total_dc_power` | $kW$ | Total DC power generated |
| `mean_ac_power` | $kW$ | Average AC power per inverter |
| `mean_dc_power` | $kW$ | Average DC power per inverter |
| `total_daily_yield` | $kWh$ | Cumulative energy produced during current day |
| `inverter_count` | Count | Number of active online inverters |
| `sensor_irradiation`| $kW/m^2$ | Ground pyranometer sensor solar irradiance |
| `sensor_ambient_temp` | $^\circ C$ | On-site ambient surface temperature |
| `sensor_module_temp` | $^\circ C$ | On-site PV module back-of-panel temperature |
| `nasa_ghi` | $W/m^2$ | Global Horizontal Irradiance from NASA satellite feed |
| `nasa_cloud_amt` | $\%$ ($0-100$) | Daylight Cloud Coverage Index |
| `nasa_ambient_temp` | $^\circ C$ | 2-meter air temperature |
| `nasa_rh` | $\%$ ($0-100$) | Relative Humidity at 2 meters |
| `nasa_wind_speed` | $m/s$ | Wind speed at 2 meters |
| `nasa_pressure` | $kPa$ | Atmospheric surface pressure |
| `nasa_clearness_index` | $[0, 1]$ | All-sky insolation clearness index ($K_t$) |
| `is_anomaly` | $0$ or $1$ | Isolation Forest anomaly flag ($1 = \text{anomaly}$) |
| `anomaly_score` | Float | Isolation Forest decision function score |

---

## 5. Pipeline Execution

To re-run the entire pipeline from scratch:
```bash
python scripts/run_data_pipeline.py --force
```

To re-process local raw files with updated configurations:
```bash
python scripts/run_data_pipeline.py --skip-download
```
