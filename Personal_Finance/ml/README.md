# Machine Learning Module — Personal Finance Analytics

This directory contains the machine learning multi-step sequence forecasting models, training scripts, inference tools, evaluation reports, and database population pipelines for the Personal Finance Analytics system.

---

## 📌 Overview

The primary objective of the ML sequence models is to analyze a user's past spending sequence (10 timesteps) and predict their likely **next 7 days of spending amounts**.

### Models Implemented
1. **Multi-Step LSTM Model (`lstm/train_lstm.py`)**: 2-layer Long Short-Term Memory network outputting a 7-step future spending vector `[Day 1, Day 2, Day 3, Day 4, Day 5, Day 6, Day 7]`.
2. **Multi-Step SimpleRNN Model (`rnn/train_rnn.py`)**: 2-layer Recurrent Neural Network for comparative 7-day sequence forecasting.

---

## 📁 Structured Directory Layout

```
ml/
├── lstm/                       # Dedicated LSTM Model Module
│   ├── train_lstm.py           # Multi-step LSTM training script
│   ├── lstm_model.pt           # PyTorch LSTM weights checkpoint
│   ├── lstm_metrics.json       # Evaluation metrics (Overall & per-day RMSE/MAE/R²)
│   ├── lstm_results.csv        # Test set actual vs predicted 7-day values
│   └── lstm_report.md         # Markdown report with day-by-day accuracy & sample forecasts
├── rnn/                        # Dedicated SimpleRNN Model Module
│   ├── train_rnn.py            # Multi-step SimpleRNN training script
│   ├── rnn_model.pt            # PyTorch SimpleRNN weights checkpoint
│   ├── rnn_metrics.json        # Evaluation metrics (Overall & per-day RMSE/MAE/R²)
│   ├── rnn_results.csv         # Test set actual vs predicted 7-day values
│   └── rnn_report.md          # Markdown report with day-by-day accuracy & sample forecasts
├── utils/
│   └── scaler.pkl              # Shared MinMaxScaler for transaction amounts
├── predict.py                  # Inference engine for 7-day spending predictions
├── db_populate.py              # Script to populate database tables & model predictions
├── requirements.txt            # ML module dependencies
└── README.md                   # Documentation and usage reference
```

---

## 🚀 Quick Start & Usage

### 1. Install Dependencies
```bash
pip install -r ml/requirements.txt
```

### 2. Preprocess Data & Clean Outliers
Cleans raw data (capping amounts `< 100,000` INR to eliminate artificial `999,999` noise) and generates sequence arrays:
```bash
python preprocessing/preprocessing.py
```

### 3. Train Multi-Step LSTM Model (7-Day Forecast)
```bash
python ml/lstm/train_lstm.py
```
- Saves model weights to `ml/lstm/lstm_model.pt`.
- Saves metrics JSON to `ml/lstm/lstm_metrics.json`.
- Saves test predictions CSV to `ml/lstm/lstm_results.csv`.
- Saves human-readable markdown report to `ml/lstm/lstm_report.md`.

### 4. Train Multi-Step SimpleRNN Model (7-Day Forecast)
```bash
python ml/rnn/train_rnn.py
```
- Saves model weights to `ml/rnn/rnn_model.pt`.
- Saves metrics JSON to `ml/rnn/rnn_metrics.json`.
- Saves test predictions CSV to `ml/rnn/rnn_results.csv`.
- Saves human-readable markdown report to `ml/rnn/rnn_report.md`.

### 5. Run 7-Day Spending Prediction Inference
Predict a user's next 7 days of spending given their last 10 transaction amounts:
```python
from ml.predict import SpendingPredictor

# Initialize predictor with trained LSTM model
predictor = SpendingPredictor(model_type="lstm")

# Past 10 transaction amounts for a user
past_spending = [1200.0, 450.0, 3500.0, 890.0, 2100.0, 1500.0, 620.0, 4100.0, 950.0, 2800.0]

preds_7_days = predictor.predict_next_7_days(past_spending)
for day, val in preds_7_days.items():
    print(f"{day}: INR {val:,.2f}")
```

### 6. Populate Database Tables
Fill domain database tables (`users`, `categories`, `payment_modes`, `locations`, `transactions`, `monthly_summary`, `category_monthly_summary`), model registry (`model_registry`), and 7-day predictions (`predictions`):
```bash
python ml/db_populate.py
```

## Seven-Day Category Seasonal-Naive Baseline

Run these commands from the `Personal_Finance` directory after the standardized dataset is available at `preprocessing/processed/standardised_dataset.csv`:

```bash
python -m ml.seasonal_naives.preprocessing.feature_extraction
python -m ml.seasonal_naives.preprocessing.sequence_preparation
python -m ml.seasonal_naives.train_seasonal_naive
python -m ml.seasonal_naives.evaluate_seasonal_naive
```

The preprocessing uses the category LSTM's expense filtering, daily aggregation, zero-filled dates, and calendar-day series. Sequence preparation reuses the LSTM's per-user/category 80/20 chronological split, 90-day history, and seven-day test-window origins. It also creates an overall daily total per user using the same daily split and window protocol. For each test window, the seasonal-naive rule repeats the corresponding values from the most recent complete seven-day period. This is a deterministic baseline and has no learned weights; the training step records its configuration and produces test forecasts.

Generated daily series, sequence arrays, and metadata are saved under `ml/seasonal_naives/preprocessed/`. These reproducible, data-derived files are git-ignored. The baseline configuration, test predictions, and raw-currency overall-total and per-category MAE, RMSE, and R² metrics are saved under `ml/seasonal_naives/models/` as `seasonal_naive_model.json`, `test_predictions.csv`, and `seasonal_naive_metrics.json`, respectively. The large prediction CSV is also git-ignored; the compact metrics and configuration files remain in the repository.

Run baseline tests from `Personal_Finance` with:

```bash
python -m unittest ml.seasonal_naives.test_seasonal_naive -v
```
