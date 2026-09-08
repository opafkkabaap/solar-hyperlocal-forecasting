# Experimental Results & Model Benchmarks (LSTM vs. GRU)

**Paper Title**: *Adaptive Multi-Modal AI Framework for Hyperlocal Weather Forecasting in Distributed Solar Rooftop Systems*  
**Course**: DBMS BCSE302L — Digital Assignment 1 (DA-1)  
**Authors**: Revanth P. A. (24BCE1730) & Siripuram Srimannath (24BCE5034)  

---

## 1. Executive Summary

This document reports the comparative empirical benchmarking results of the **SolarLSTM** and **SolarGRU** sequential neural networks trained on multi-modal telemetry from **Plant 1 (Rooftop Site A — Pune Region)**. Both models were trained on the initial 24 days (May 15 to June 7, 2020), validated on 5 days (June 8 to June 12), and evaluated strictly on the unseen **5-day chronological test partition (June 13 to June 17, 2020)**.

The task is **multi-step ultra-short-term forecasting ($H = 4$ steps)** predicting photovoltaic AC power generation across 15, 30, 45, and 60 minutes ahead ($t+15\text{m}$ to $t+60\text{m}$) using a 24-hour historical lookback window ($L = 96$ time steps).

---

## 2. Comparative Model Architecture & Parameters

| Parameter | SolarLSTM | SolarGRU | Comparison / Note |
| :--- | :--- | :--- | :--- |
| **Input Features ($D$)** | **13 features** | **13 features** | Identical multi-modal feature vector |
| **Sequence Lookback ($L$)**| **96 steps** (24h) | **96 steps** (24h) | Identical historical window |
| **Forecast Horizon ($H$)** | **4 steps** (1h ahead) | **4 steps** (1h ahead) | Multi-step direct nowcasting |
| **Hidden Dimension** | 64 units | 64 units | 2 stacked recurrent layers |
| **Gating Mechanism** | 3 gates ($f, i, o$) + Cell $C$ | 2 gates ($z, r$) | GRU eliminates separate cell state |
| **Trainable Parameters** | **55,716** | **42,340** | **GRU has 24.0% fewer parameters** |
| **Optimizer & Loss** | Adam ($\text{lr}=10^{-3}$), MSE | Adam ($\text{lr}=10^{-3}$), MSE | Identical learning setup |
| **Convergence Epoch** | Epoch 31 ($\text{loss}=0.00639$) | Epoch 26 ($\text{loss}=0.00605$) | GRU converged faster |

---

## 3. Empirical Test Set Performance Table

Evaluated on the unseen test partition (June 13 to June 17, 2020 — 477 test sequences):

### 3.1 Overall Horizon Benchmarks
| Model Architecture | Trainable Params | Best Val Loss | Overall $R^2$ | Overall RMSE ($kW$) | Overall MAE ($kW$) | Overall Daytime MAPE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SolarLSTM** | **55,716** | 0.006390 | **0.9034** | **2,542.68** | **1,377.94** | 31.00% |
| **SolarGRU** | **42,340** (-24.0%) | **0.006055** | 0.8991 | 2,598.91 | 1,433.31 | **30.53%** |

### 3.2 Horizon Step-by-Step Breakdown ($t+15\text{m}$ to $t+60\text{m}$)

| Forecast Step | Lead Time | LSTM $R^2$ | GRU $R^2$ | LSTM RMSE ($kW$) | GRU RMSE ($kW$) | LSTM MAE ($kW$) | GRU MAE ($kW$) | LSTM MAPE | GRU MAPE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Step 1** | $t+15\text{ min}$ | **0.9221** | 0.9219 | **2,283.87** | 2,286.97 | **1,232.84** | 1,253.54 | 27.72% | **25.44%** |
| **Step 2** | $t+30\text{ min}$ | **0.9075** | 0.9034 | **2,488.26** | 2,542.15 | **1,325.16** | 1,400.31 | 29.56% | **29.53%** |
| **Step 3** | $t+45\text{ min}$ | **0.8972** | 0.8917 | **2,623.55** | 2,692.60 | **1,430.87** | 1,474.74 | 32.61% | **32.45%** |
| **Step 4** | $t+60\text{ min}$ | **0.8869** | 0.8794 | **2,751.45** | 2,841.56 | **1,522.89** | 1,604.64 | **34.10%** | 34.70% |

*Evaluation Details:*
- **Daytime MAPE Threshold**: Computed on active generation using the data-derived threshold $y > 0.01 \times y_{\max}$ ($278.95\text{ kW}$).
- **Key Insight**: GRU delivers slightly better daytime percentage accuracy at the immediate step ($t+15\text{m}$ MAPE = 25.44% vs 27.72%), while LSTM maintains higher variance explanation at the 60-minute mark ($R^2 = 0.8869$ vs $0.8794$).

---

## 4. Visualizations

### 4.1 Training & Validation Loss Curves
| SolarLSTM Loss Curve | SolarGRU Loss Curve |
| :---: | :---: |
| ![LSTM Loss](results/figures/lstm_plant1_loss.png) | ![GRU Loss](results/figures/gru_plant1_loss.png) |

### 4.2 Multi-Step Forecast vs. Actual Generation
| SolarLSTM Forecast vs. Actual | SolarGRU Forecast vs. Actual |
| :---: | :---: |
| ![LSTM Forecast](results/figures/lstm_plant1_forecast.png) | ![GRU Forecast](results/figures/gru_plant1_forecast.png) |

---
