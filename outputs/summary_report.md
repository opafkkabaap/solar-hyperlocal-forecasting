# Solar Forecasting & Energy Optimization System Report

**Site:** 14 (Bundoora, Lat: -37.72, Lon: 145.05)
**Rated DC Capacity:** 66.0 kWp | **Panel:** Trina 330W | **Inverter:** 2 x SolarEdge SE27.6K

## 1. Forecasting Performance Metrics

| Model | RMSE (kWh) | MAE (kWh) | nRMSE (%) | R² Score | Skill Score vs Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Physics-Augmented XGBoost** | **4.69** | **2.08** | **7.10%** | **0.907** | **+51.9%** |
| 24h Persistence Baseline | 9.74 | 4.81 | 14.75% | 0.597 | 0.0% |

## 2. Energy Optimization Insights

- **Optimal Tilt Angle:** `20°` True North (Yield Gain: **+3.8%** over horizontal).
- **System Performance Ratio (PR):** `56.8%`.
- **Loss Attribution:** Thermal Derating: `1967.5 kWh` | Cloud Attenuation: `17144.8 kWh` | Clipping: `823.1 kWh`.
- **Predictive Battery Storage:** Peak demand shaved by **8.5 kW (24.1%)**, reducing net grid import by **6.3%**.
