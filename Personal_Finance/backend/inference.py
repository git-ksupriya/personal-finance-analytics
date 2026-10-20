from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from ml.lstm.train_lstm import LSTMModel as OverallLSTM
from ml.lstm_categories.preprocessing.feature_extraction import (
    add_calendar_features,
    aggregate_daily_category_spending,
    prepare_expense_data,
)
from ml.lstm_categories.train_lstm import LSTMModel as CategoryLSTM


PROJECT_DIR = Path(__file__).resolve().parents[1]
OVERALL_MODEL_PATH = PROJECT_DIR / "ml" / "lstm" / "lstm_model.pt"
OVERALL_SCALER_PATH = PROJECT_DIR / "ml" / "utils" / "scaler.pkl"
CATEGORY_MODELS_DIR = PROJECT_DIR / "ml" / "lstm_categories" / "models"
DEVICE = torch.device("cpu")

CATEGORY_FEATURES = [
    "daily_spending",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]


@lru_cache(maxsize=1)
def load_overall_model():
    if not OVERALL_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Overall LSTM weights not found: {OVERALL_MODEL_PATH}")
    if not OVERALL_SCALER_PATH.is_file():
        raise FileNotFoundError(f"Overall LSTM scaler not found: {OVERALL_SCALER_PATH}")

    model = OverallLSTM(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=7,
        dropout=0.2,
    )
    model.load_state_dict(
        torch.load(OVERALL_MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model, joblib.load(OVERALL_SCALER_PATH)


@lru_cache(maxsize=16)
def load_category_model(category):
    category_dir = CATEGORY_MODELS_DIR / category.strip().lower().replace(" ", "_")
    model_path = category_dir / "model.pt"
    scaler_path = category_dir / "scaler.pkl"
    if not model_path.is_file() or not scaler_path.is_file():
        raise FileNotFoundError(f"Category LSTM artifacts not found for {category}")

    model = CategoryLSTM()
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model, joblib.load(scaler_path)


def predict_overall_transaction_steps(transactions):
    if len(transactions) < 10:
        return {
            "available": False,
            "reason": "At least 10 transactions are required for the overall LSTM.",
            "values": [],
        }

    model, scaler = load_overall_model()
    amounts = np.asarray(
        [float(transaction["amount"]) for transaction in transactions[-10:]],
        dtype=np.float32,
    ).reshape(-1, 1)
    scaled = scaler.transform(amounts).reshape(1, 10, 1)
    with torch.inference_mode():
        scaled_predictions = model(torch.from_numpy(scaled)).cpu().numpy().reshape(-1, 1)
    values = scaler.inverse_transform(scaled_predictions).reshape(-1)
    return {
        "available": True,
        "horizon": "next 7 transaction steps (not calendar days)",
        "values": [round(max(0.0, float(value)), 2) for value in values],
    }


def _category_daily_history(expenses, category, as_of):
    category_expenses = expenses[expenses["category"] == category]
    if category_expenses.empty:
        return None

    daily = aggregate_daily_category_spending(category_expenses)
    first_date = daily["date"].min()
    dates = pd.date_range(first_date, pd.Timestamp(as_of), freq="D")
    if len(dates) < 90:
        return None

    series = (
        daily.set_index("date")["daily_spending"]
        .reindex(dates, fill_value=0.0)
        .rename_axis("date")
        .reset_index(name="daily_spending")
    )
    return add_calendar_features(series).tail(90).reset_index(drop=True)


def predict_category_daily(transactions, as_of=None):
    as_of = as_of or date.today()
    expense_rows = [
        {
            "user_id": row["user_id"],
            "category": row["category"],
            "date": row["transaction_date"],
            "transaction_type": row["transaction_type"],
            "amount": row["amount"],
        }
        for row in transactions
    ]
    if not expense_rows:
        return {
            "available": False,
            "reason": "No expense history is available.",
            "categories": {},
            "errors": [],
        }

    expenses = prepare_expense_data(pd.DataFrame(expense_rows))
    if expenses.empty:
        return {
            "available": False,
            "reason": "No expense history is available.",
            "categories": {},
            "errors": [],
        }
    expenses["date"] = pd.to_datetime(expenses["date"], errors="coerce")

    latest_user_expense = expenses["date"].max().date()
    if (as_of - latest_user_expense).days > 7:
        return {
            "available": False,
            "reason": "Add a recent transaction to enable current-period forecasts.",
            "categories": {},
            "errors": [],
        }

    forecasts = {}
    errors = []
    for category in sorted(expenses["category"].unique()):
        history = _category_daily_history(expenses, category, as_of)
        if history is None:
            errors.append(f"{category}: requires at least 90 calendar days of history.")
            continue
        try:
            model, scaler = load_category_model(str(category))
            features = history[CATEGORY_FEATURES].to_numpy(dtype=np.float32)
            features[:, 0] = scaler.transform(features[:, 0].reshape(-1, 1)).reshape(-1)
            with torch.inference_mode():
                scaled_predictions = (
                    model(torch.from_numpy(features[None, :, :]))
                    .cpu()
                    .numpy()
                    .reshape(-1, 1)
                )
            values = scaler.inverse_transform(scaled_predictions).reshape(-1)
            forecasts[str(category)] = [
                round(max(0.0, float(value)), 2) for value in values
            ]
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
            errors.append(f"{category}: {error}")

    return {
        "available": bool(forecasts),
        "reason": None if forecasts else "No category model could produce a forecast.",
        "categories": forecasts,
        "forecast_start": (as_of + timedelta(days=1)).isoformat(),
        "errors": errors,
    }
