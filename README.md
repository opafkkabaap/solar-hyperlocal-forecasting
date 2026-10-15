# Solar Weather Forecasting & Optimal Energy Output System

An end-to-end physics-informed machine learning and energy optimization system designed for the **UniSolar (La Trobe University)** multi-site dataset.

## System Architecture

```
                       UniSolar Dataset (5 Campuses, 42 Sites)
       [Weather Data]      [Solar Irradiance (UTC)]      [Solar Energy Generation]
              │                        │                             │
              ▼                        ▼                             ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 1. Timezone & Data Alignment (UTC -> Australia/Melbourne, 1h Resample) │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 2. Astronomical & Physical Engine (pvlib)                              │
   │    - Solar Zenith, Elevation, Azimuth                                  │
   │    - Ineichen Clear-Sky GHI, DNI, DHI & Clearsky Index (Kt)            │
   │    - Sandia / Faiman PV Cell Temperature Model                         │
   │    - Theoretical STC & Temperature-Derated DC Power                    │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 3. Dual Forecasting Engine                                             │
   │    - Stage 1: XGBoost Solar Weather & Irradiance Forecaster (GHI/Cloud) │
   │    - Stage 2: Physics-Augmented XGBoost Solar Generation Forecaster    │
   │    - Benchmark: Deep Learning Bidirectional GRU Sequence Model         │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 4. Energy Optimization & Decision Engine                               │
   │    - Mathematical Optimal Tilt Angle Search (0° to 60° True North)     │
   │    - Performance Ratio (PR) & Loss Attribution (Thermal/Cloud/Clipping)│
   │    - 24-Hour Predictive Battery Storage Dispatch & Peak Shaving        │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 5. Outputs: High-Resolution Figures & Executive Markdown Summary Report│
   └────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
solar-weather-optimal-forecasting/
├── config.py                 # Campus metadata, file paths, battery specifications
├── run_pipeline.py           # CLI entry point to run end-to-end pipeline
├── src/
│   ├── data_pipeline.py      # Timezone harmonization, active range filtering, caching
│   ├── physics_engine.py     # pvlib astronomical calculations & clear-sky models
│   ├── feature_engineering.py# Cyclical, lag, rolling, and physical features
│   ├── models_weather.py     # XGBoost irradiance & cloud opacity forecaster
│   ├── models_solar.py       # Physics-informed XGBoost & Deep Bi-GRU generation models
│   ├── optimizer.py          # Optimal tilt, performance ratio, battery dispatch
│   └── evaluate.py           # Metrics (MAE, RMSE, R², Skill Score) & visualizations
└── outputs/
    ├── cache/                # Pre-processed harmonized datasets
    ├── figures/              # Generated high-resolution visualization plots
    └── summary_report.md     # Executive markdown summary report
```

## How to Run

### Run on Default Site (Site 14, Bundoora 66 kWp)
```bash
python run_pipeline.py --site 14
```

### Run on Large Scale Site (Site 27, Bundoora 540 kWp)
```bash
python run_pipeline.py --site 27
```

### Fast Execution (Skipping Deep Learning Benchmark)
```bash
python run_pipeline.py --site 14 --no-deep
```

### Specify Number of Visualization Days
```bash
python run_pipeline.py --site 14 --days 7
```
