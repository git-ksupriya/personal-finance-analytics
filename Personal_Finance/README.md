# Personal Finance Analytics

An end-to-end Machine Learning, Preprocessing, Database, and Analytics application designed to track, clean, analyze, and forecast personal finance transactions.

---

## 🏗 Project Architecture

```
Personal_Finance/
├── raw/                         # Raw budgetwise finance dataset CSV
├── preprocessing/               # Data cleaning, standardization, feature engineering & sequence prep
│   ├── data_cleaning.py
│   ├── data_standardisation.py
│   ├── Feature_Extraction.py
│   ├── sequence_preparation.py
│   └── preprocessing.py
├── ml/                          # Deep Learning models, sequence forecasting, and DB loader
│   ├── models/                  # Saved weights (.pt), scaler (.pkl), and evaluation JSON metrics
│   ├── train_lstm.py            # LSTM sequence model training script
│   ├── train_rnn.py             # SimpleRNN sequence model training script
│   ├── predict.py               # Next spending prediction inference engine
│   ├── db_populate.py           # Database population script for core domain tables & AI models
│   ├── requirements.txt
│   └── README.md
├── dbms/                        # Database migration scripts & query definitions
│   └── migrations/
│       ├── 001_users.sql
│       ├── 002_transactions.sql
│       ├── 003_analytics.sql
│       └── 004_ai.sql
└── backend/                     # Backend API services
```

---

## ⚡ Workflow & Execution Guide

### 1. Run Preprocessing Pipeline
Cleans raw transaction logs, standardizes categories and locations, extracts temporal and financial features, and builds normalized sequence windows:
```bash
python preprocessing/preprocessing.py
```

### 2. Train Sequence Machine Learning Models

#### Train LSTM Model:
Looks at a user's past spending sequence to predict their likely next transaction spending:
```bash
python ml/train_lstm.py
```

#### Train SimpleRNN Model:
```bash
python ml/train_rnn.py
```

### 3. Run Spending Predictions
```bash
python ml/predict.py
```

### 4. Populate Database Tables
Executes schema creation and fills `users`, `categories`, `payment_modes`, `locations`, `transactions`, `monthly_summary`, `category_monthly_summary`, `model_registry`, and `predictions` tables:
```bash
python ml/db_populate.py
```

---

## 🗄 Database Tables Overview

| Domain | Table Name | Description |
|---|---|---|
| **Users** | `users` | User credentials, emails, and profile information |
| **Transactions** | `transactions` | Cleaned transaction records (amount, type, category, date) |
| **Lookups** | `categories`, `payment_modes`, `locations` | Standardized category, payment mode, and location dictionaries |
| **Analytics** | `monthly_summary`, `category_monthly_summary` | Monthly financial rollups, spending totals, and savings rates |
| **AI Module** | `model_registry` | Trained ML model metadata, weights paths, and evaluation metrics |
| **AI Module** | `predictions` | Sequence forecasting predictions for user spending |
