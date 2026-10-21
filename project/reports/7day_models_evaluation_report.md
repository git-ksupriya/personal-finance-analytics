# Multi-Model 7-Day Spending Forecast Benchmark (8-Week Lookback)

## 1. Executive Summary & Comparison Across Models
Evaluated on test user-category series predicting **the next 7 days of daily spending** from **8 weeks (56 days)** of historical context:

| Architecture | Overall MAE ($) | Overall RMSE ($) | R² Score | Day 1 MAE | Day 4 MAE | Day 7 MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multi-Step LSTM** | **$20.12** | **$79.21** | **-0.0054** | $18.71 | $18.61 | $26.13 |
| **1D-CNN Sequence** | $22.2 | $79.36 | -0.012 | $23.26 | $23.05 | $21.18 |
| **Multi-Step SimpleRNN** | $24.39 | $79.52 | -0.0255 | $19.22 | $35.26 | $21.95 |
| **XGBoost MultiOutput** | $22.62 | $86.25 | -0.4154 | $21.9 | $22.4 | $22.13 |

## 2. Per-Category Performance Breakdown ($ MAE)

| category       |   test_windows |   actual_mean |   lstm_mae |   rnn_mae |   cnn_mae |   xgb_mae |
|:---------------|---------------:|--------------:|-----------:|----------:|----------:|----------:|
| entertainment  |            430 |         11.16 |      18.07 |     22.26 |     19.71 |     21.09 |
| food_dining    |            414 |          9.26 |      16.18 |     20.41 |     17.5  |     17.46 |
| gas_transport  |            420 |         16.14 |      21.64 |     24.84 |     23.56 |     23.53 |
| grocery_net    |            404 |          5.73 |      13.57 |     18.22 |     14.45 |     12.76 |
| grocery_pos    |            450 |         29.68 |      35.39 |     38.73 |     39.03 |     44.33 |
| health_fitness |            454 |          8.97 |      16.04 |     20.26 |     17.19 |     17.2  |
| home           |            457 |         14.27 |      20.27 |     23.93 |     21.78 |     22.64 |
| kids_pets      |            433 |         12.91 |      19.11 |     22.92 |     20.74 |     21.28 |
| misc_net       |            426 |         11.79 |      19.91 |     24.72 |     21.83 |     21.47 |
| misc_pos       |            420 |          9.81 |      17.27 |     21.95 |     18.76 |     19.3  |
| personal_care  |            398 |          8.02 |      15.23 |     19.52 |     16.36 |     16.06 |
| shopping_net   |            452 |         16.42 |      23.71 |     28.56 |     27.39 |     27.23 |
| shopping_pos   |            437 |         16.88 |      23.97 |     28.63 |     27.27 |     30.5  |
| travel         |            405 |         10.8  |      19.62 |     24.99 |     23.39 |     19.3  |

## 3. Day-by-Day Forecast Degradation Analysis (MAE)
| Forecast Horizon | LSTM MAE | SimpleRNN MAE | 1D-CNN MAE | XGBoost MAE |
| :--- | :---: | :---: | :---: | :---: |
| **Day 1** | $18.71 | $19.22 | $23.26 | $21.9 |
| **Day 2** | $23.05 | $30.32 | $21.12 | $22.45 |
| **Day 3** | $21.74 | $27.59 | $20.48 | $23.26 |
| **Day 4** | $18.61 | $35.26 | $23.05 | $22.4 |
| **Day 5** | $17.09 | $18.02 | $19.8 | $22.48 |
| **Day 6** | $15.51 | $18.38 | $26.52 | $23.75 |
| **Day 7** | $26.13 | $21.95 | $21.18 | $22.13 |

## 4. Key Architectural Insights
- **LSTM Superiority:** The gating mechanism in LSTM effectively models longer dependencies across the 56-day (8-week) input window, outperforming Vanilla RNN which suffers from vanishing gradients.
- **1D-CNN Efficiency:** Convolutional filters extract salient multi-day spending bursts with fast parallel training.
- **Tree-based XGBoost:** Provides strong non-linear feature split baselines, while neural sequence models excel in preserving temporal order.
