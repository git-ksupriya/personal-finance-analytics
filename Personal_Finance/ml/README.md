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
