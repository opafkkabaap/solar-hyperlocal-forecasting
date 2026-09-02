# Experimental Results & Model Benchmarks

**Paper Title**: *Adaptive Multi-Modal AI Framework for Hyperlocal Weather Forecasting in Distributed Solar Rooftop Systems*  
**Course**: DBMS BCSE302L — Digital Assignment 1 (DA-1)  
**Authors**: Revanth P. A. (24BCE1730) & Siripuram Srimannath (24BCE5034)  

---

## 1. Executive Summary

This document reports the empirical benchmarking results of the **SolarLSTM** sequential neural network trained on multi-modal telemetry from **Plant 1 (Rooftop Site A — Pune Region)**. The model was trained on the initial 24 days (May 15 to June 7, 2020), tuned on 5 validation days (June 8 to June 12), and evaluated strictly on the unseen **5-day chronological test partition (June 13 to June 17, 2020)**.

The task is **multi-step ultra-short-term forecasting ($H = 4$ steps)** predicting photovoltaic AC power generation across 15, 30, 45, and 60 minutes ahead ($t+15\text{m}$ to $t+60\text{m}$) using a 24-hour historical lookback window ($L = 96$ time steps).

---

## 2. Model Architecture & Hyperparameters

| Hyperparameter | Specification |
| :--- | :--- |
| **Model Type** | Stacked Long Short-Term Memory (`SolarLSTM`) |
| **Input Features ($D$)** | **13 features** (Autoregressive AC Power + 3 Ground Weather Sensors + 7 NASA POWER Weather Feeds + 2 Diurnal Encodings) |
| **Sequence Lookback ($L$)** | **96 steps** (24.0 continuous hours) |
| **Forecast Horizon ($H$)** | **4 steps** (15, 30, 45, 60 minutes ahead) |
| **LSTM Layers** | 2 stacked layers with inter-layer dropout |
| **Hidden Dimension** | 64 units per LSTM cell |
| **Dropout Regularization**| 0.20 (LSTM) / 0.10 (FC projection head) |
| **Total Trainable Parameters** | **55,716 parameters** |
| **Loss Function** | Mean Squared Error ($\text{MSE}$) on scaled target $[0, 1]$ |
| **Optimizer** | Adam ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$) |
| **Learning Rate Schedule** | `ReduceLROnPlateau(mode='min', factor=0.5, patience=5)` |
| **Early Stopping** | Monitored validation loss with patience $= 10$ epochs |
| **Best Epoch** | Epoch 31 (Validation Loss: $0.00639$, $\text{LR} = 0.0005$) |

---

## 3. Empirical Test Set Performance Table

Evaluated on the unseen test partition (June 13 to June 17, 2020 — 477 test sequences):

| Metric | Full 1-Hour Horizon | $t+15\text{ min}$ Ahead | $t+30\text{ min}$ Ahead | $t+45\text{ min}$ Ahead | $t+60\text{ min}$ Ahead |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$R^2$ Score** | **0.9034** | **0.9221** | **0.9075** | **0.8972** | **0.8869** |
| **RMSE ($kW$)** | **2,542.68** | 2,283.87 | 2,488.26 | 2,623.55 | 2,751.45 |
| **MAE ($kW$)** | **1,377.94** | 1,232.84 | 1,325.16 | 1,430.87 | 1,522.89 |
| **Daytime MAPE** | **31.00%** | **27.72%** | **29.56%** | **32.61%** | **34.10%** |

*Note on Evaluation Methodology:*
- **Daytime MAPE Threshold**: Computed exclusively on active generation using the data-derived threshold $y > 0.01 \times y_{\max}$ ($278.95\text{ kW}$, 1% of peak capacity). This avoids artificial percentage blowups caused by near-zero nocturnal and dawn/dusk denominators without relying on arbitrary hardcoded cutoffs.
- **Horizon Degradation Trend**: Error increases monotonically from $t+15\text{m}$ ($R^2 = 0.9221$, $\text{RMSE} = 2,283.87\text{ kW}$) to $t+60\text{m}$ ($R^2 = 0.8869$, $\text{RMSE} = 2,751.45\text{ kW}$), consistent with physical atmospheric volatility principles.

---

## 4. Visualizations

### 4.1 Training & Validation Loss Curve
The model demonstrated stable convergence without overfitting, aided by inter-layer dropout and learning rate annealing:

![LSTM Loss Curve](results/figures/lstm_plant1_loss.png)

### 4.2 Multi-Step Forecast vs. Actual Generation
5-day test partition (June 13 to June 17, 2020) comparing ground-truth AC power generation against the 15-minute ($t+15\text{m}$) and 60-minute ($t+60\text{m}$) forecasts:

![LSTM Forecast Comparison](results/figures/lstm_plant1_forecast.png)

---

## 5. Input Feature Vectors ($D = 13$)

| Index | Feature Name | Source / Modality | Unit | Physical Function |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `total_ac_power` | Ground Telemetry (Inverters) | $kW$ | Autoregressive base signal |
| 2 | `sensor_irradiation` | Ground Telemetry (Pyranometer)| $kW/m^2$ | On-site plane-of-array irradiance |
| 3 | `sensor_ambient_temp` | Ground Telemetry (Weather Mast)| $^\circ C$ | Microclimatic surface temperature |
| 4 | `sensor_module_temp` | Ground Telemetry (Back of PV) | $^\circ C$ | Direct thermal degradation proxy |
| 5 | `nasa_ghi` | NASA POWER API (CERES/MERRA-2) | $W/m^2$ | Macro-scale solar irradiance |
| 6 | `nasa_cloud_amt` | NASA POWER API (Satellite) | $\%$ | Cloud attenuation index |
| 7 | `nasa_ambient_temp` | NASA POWER API (MERRA-2) | $^\circ C$ | Regional 2-meter air temperature |
| 8 | `nasa_rh` | NASA POWER API (MERRA-2) | $\%$ | Atmospheric moisture / relative humidity |
| 9 | `nasa_wind_speed` | NASA POWER API (MERRA-2) | $m/s$ | Convective panel cooling factor |
| 10 | `nasa_pressure` | NASA POWER API (MERRA-2) | $kPa$ | Surface atmospheric density |
| 11 | `nasa_clearness_index` | NASA POWER API (Derived) | $[0, 1]$ | Atmospheric transmission ratio ($K_t$) |
| 12 | `hour_sin` | Engineered Cyclical Encoding | $[-1, 1]$ | Continuous diurnal progression |
| 13 | `hour_cos` | Engineered Cyclical Encoding | $[-1, 1]$ | Continuous diurnal progression |

---

## 6. How to Reproduce

To retrain the model and regenerate all figures and benchmark metrics:

```bash
# 1. Ensure dependencies are installed
pip install -r requirements.txt
pip install torch matplotlib

# 2. Run the training routine
python train_lstm.py --plant plant1 --epochs 40 --batch_size 32 --lr 0.001

# 3. View the generated metrics
cat results/lstm_plant1_metrics.json
```

---

## 7. Artifact Checklist
- [x] Model Weights: `models/lstm_plant1_best.pth`
- [x] Loss Curve: `results/figures/lstm_plant1_loss.png`
- [x] Forecast Curve: `results/figures/lstm_plant1_forecast.png`
- [x] Benchmark JSON: `results/lstm_plant1_metrics.json`
- [x] Training Utility: `train_utils.py`
- [x] Model Architecture: `model_lstm.py`
- [x] Training Orchestrator: `train_lstm.py`
