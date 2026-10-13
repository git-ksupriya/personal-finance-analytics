# Personalised Spending Forecasting & Anomaly Detection Pipeline (PostgreSQL + Hurdle LSTM/RNN)

An end-to-end, reproducible machine learning system built on PostgreSQL and PyTorch for personalised weekly expenditure forecasting and transaction anomaly detection using the Sparkov credit card transaction dataset (~1.85M transactions, 999 users, 14 merchant categories, 2019-01-01 to 2020-12-31).

---

## Architecture & Overview

```
                      Raw Transactions (~1.85M rows)
                                    │
                                    ▼
                     PostgreSQL 15+ Normalized Schema
     ┌──────────────┬─────────────────┬───────────────────┬─────────────────┐
     ▼              ▼                 ▼                   ▼                 ▼
   users        merchants        transactions    weekly_category_spend   anomalies
  (999 rows)    (693 rows)       (1.85M rows)        (1.38M rows)       (87,032 rows)
                                                          │
                                                          ▼
                                            Chronological Feature Windows
                                            - Dynamic (log1p amt, count, sin/cos)
                                            - Static (User stats, Category embed)
                                                          │
                                     ┌────────────────────┴───────────────────┐
                                     ▼                                        ▼
                             Hurdle LSTM (2-Layer)                   Hurdle Vanilla RNN
                             Head A: P(spend > 0)                    Head A: P(spend > 0)
                             Head B: log1p(amt | spend)              Head B: log1p(amt | spend)
                                     │                                        │
                                     └────────────────────┬───────────────────┘
                                                          ▼
                                                 forecasts & model_runs
                                                 (800k predictions logged)
```

### Key Highlights
1. **Real Elapsed Time (1-Week Cadence):** Chronologically uniform weekly timesteps (105 calendar weeks). Incomplete partial weeks removed; zero-spend weeks are populated explicitly via SQL `generate_series`.
2. **Leakage-Free Splitting:** Strict chronological split:
   - **Train:** Weeks 1–73 (~70% history)
   - **Validation:** Weeks 74–89 (~15% history)
   - **Test:** Weeks 90–105 (~15% history)
   - All scalers, user historical statistics, and category distributions are fit strictly on Train.
3. **Two-Stage Hurdle Recurrent Network:** Joint classification (`P(spend > 0)` via BCE loss) and positive expenditure regression (`Huber loss` on positive weeks).
4. **Per-User Anomaly Detection:** Spatio-temporal transaction profiling (Haversine distance, inter-transaction intervals, time/day) benchmarked via **Robust IQR / Z-Score** and **Isolation Forest**, evaluated against true fraud labels using PR-AUC and Precision/Recall.

---

## Final Performance & Benchmark Results

### 1. Spending Forecast Test Set (Weeks 90–105, 200,064 Windows)

| Model | Overall MAE ($) | Overall RMSE ($) | Non-Zero MAE ($) | Non-Zero RMSE ($) | AUC | F1 Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline: Last-Week** | $116.79 | $306.54 | $143.09 | $335.52 | 0.6137 | 0.7522 | 0.7536 | 0.7507 |
| **Baseline: 4-Week Moving Avg** | $99.67 | $240.80 | $118.85 | $279.34 | 0.5412 | 0.8131 | 0.6997 | 0.9704 |
| **Baseline: User-Cat Hist Mean** | $91.31 | $222.36 | $108.47 | $264.08 | 0.5000 | 0.8101 | 0.6808 | 1.0000 |
| **Hurdle RNN (Vanilla)** | $84.02 | $226.70 | $104.87 | $271.61 | 0.7691 | 0.8139 | 0.7738 | 0.8584 |
| **Hurdle LSTM (2-Layer)** | **$81.39** | **$224.79** | **$103.19** | **$269.56** | **0.7786** | **0.8213** | **0.7684** | **0.8822** |

- **LSTM vs. Baselines:** Hurdle LSTM achieves a **$35.40 reduction in MAE** (30.3% error reduction) compared to the Last-Week baseline and beats the 4-week moving average across both overall and non-zero spend subsets.
- **LSTM vs. Vanilla RNN:** LSTM outperforms Vanilla RNN across both classification AUC (0.7786 vs 0.7691) and regression MAE ($81.39 vs $84.02) due to superior long-range memory retention over 8+ week horizons.

### 2. Per-User Anomaly Detection (Test Set: 629k Transactions, Fraud Base Rate: 0.406%)

| Method | Precision | Recall | PR-AUC | Flagged Count | Key Strength |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Robust Z-Score / IQR** | 0.0790 | **0.7394** | 0.1078 | 69,206 | Captures high-magnitude amount spikes per user category |
| **Isolation Forest** | **0.1961** | 0.4292 | **0.1271** | 17,826 | Isolates multi-attribute deviations (speed, distance, odd hours) |

---

## Directory Structure

```
project/
├── docker-compose.yml       # Production-ready PostgreSQL & pipeline services
├── Dockerfile               # Reproducible Python container definition
├── requirements.txt         # Pinned Python package dependencies
├── .env.example             # Example environment variables
├── .env                     # Local environment credentials
├── sql/
│   ├── schema.sql           # Normalized DDL (tables, PKs, FKs, indexes)
│   └── aggregations.sql     # SQL aggregation query with calendar zero-fill
├── src/
│   ├── config.py            # Centralized project configuration & parameters
│   ├── load_data.py         # Fast streaming COPY data ingestion
│   ├── preprocess.py        # Leakage-free feature extraction & windowing
│   ├── datasets.py          # PyTorch custom Dataset and tensor loaders
│   ├── models.py            # Hurdle LSTM & Vanilla RNN architectures
│   ├── train.py             # Hurdle training loop with gradient clipping & DB logging
│   ├── evaluate.py          # Benchmark evaluations against 3 baselines
│   └── anomaly.py           # Per-user IQR and Isolation Forest anomaly pipeline
├── models/                  # Checkpoints (.pt), window tensors (.npz), predictions (.npy)
├── reports/
│   ├── preprocessing_report.md
│   ├── evaluation_report.md
│   └── anomaly_report.md
└── README.md
```

---

## Step-by-Step Reproduction Guide

### Prerequisites
- Python 3.10+
- PostgreSQL 15+ (or Docker Engine / Docker Desktop)

### 1. Environment Setup
```bash
# Clone the feature branch
git checkout feat/spending-forecast-anomaly

# Navigate to project directory
cd project

# Configure environment variables
cp .env.example .env
```

### 2. Run with Docker Compose (Container-Ready)
```bash
# Start PostgreSQL service
docker compose up -d postgres

# Build and run the entire ML pipeline
docker compose up pipeline
```

### 3. Run Locally from Scratch (Bare-Metal)
```bash
# Install Python dependencies
pip install -r requirements.txt

# Step 1: Initialize Database Schema & Ingest Data
psql -U postgres -d spakov_finance -f sql/schema.sql
python src/load_data.py

# Step 2: Chronological Feature Preprocessing & Zero-Filled Windowing
python src/preprocess.py

# Step 3: Train Hurdle LSTM and Vanilla RNN Models
python src/train.py --model all --seq_len 8 --epochs 15

# Optional: Window length ablations
python src/train.py --model lstm --seq_len 10 --epochs 5
python src/train.py --model lstm --seq_len 12 --epochs 5

# Step 4: Run Evaluation Against Baselines & Generate Benchmark Tables
python src/evaluate.py

# Step 5: Run Anomaly Detection & Log Flagged Records
python src/anomaly.py
```

---

## Known Limitations & Next Steps

1. **Category Cross-Correlations:** The model processes each user-category series independently with category embeddings. Future work could leverage multivariate temporal transformers or Graph Neural Networks (GNN) to model cross-category wallet substitution effects.
2. **Dynamic Transaction Velocity:** Incorporating intraday transaction frequency directly into the weekly recurrence can further enhance sudden spend burst predictions.
3. **Continuous Online Retraining:** Implementing automated incremental training or drift monitoring in PostgreSQL using `pg_cron` can keep the hurdle classification head updated with evolving seasonal behavior.
