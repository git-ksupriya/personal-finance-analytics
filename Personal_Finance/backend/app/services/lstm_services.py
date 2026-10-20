import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from backend.app.services.database import get_db_connection


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIRECTORY = PROJECT_ROOT / "ml" / "lstm"

MODEL_CANDIDATES = [
    MODEL_DIRECTORY / "tuning_results" / "best_lstm_model.pt",
    MODEL_DIRECTORY / "lstm_model.pt",
]

SCALER_CANDIDATES = [
    PROJECT_ROOT / "preprocessing" / "processed" / "lstm_daily" / "daily_scaler.pkl",
    PROJECT_ROOT / "preprocessing" / "processed" / "scaler.pkl",
    MODEL_DIRECTORY / "preprocessing" / "processed" / "lstm_daily" / "daily_scaler.pkl",
]

INPUT_DAYS = 30
FORECAST_DAYS = 7


class LSTMModel(torch.nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc1 = torch.nn.Linear(hidden_size, 32)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, 7)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        x = self.fc1(last_output)
        x = self.relu(x)
        return self.fc2(x)


# ==================================================
# FETCH USER'S DAILY EXPENSES
# ==================================================

def get_user_daily_spending(user_id: str) -> pd.DataFrame:
    """Retrieve a single user's daily expenditure totals."""

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT transaction_date, COALESCE(SUM(amount), 0) AS daily_spending
                FROM transactions
                WHERE user_id = %s
                  AND transaction_type = 'Expense'
                GROUP BY transaction_date
                ORDER BY transaction_date ASC
                """,
                (user_id,),
            )
            rows = cursor.fetchall()
    finally:
        connection.close()

    if not rows:
        return pd.DataFrame(columns=["transaction_date", "daily_spending"])

    df = pd.DataFrame(rows, columns=["transaction_date", "daily_spending"])
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["daily_spending"] = pd.to_numeric(df["daily_spending"], errors="coerce").fillna(0.0)
    df = df.sort_values("transaction_date").groupby("transaction_date", as_index=False)["daily_spending"].sum()
    return df


# ==================================================
# PREPARE THE LAST 30 DAYS
# ==================================================

def prepare_user_input(daily_spending: pd.DataFrame) -> np.ndarray:
    """Create the model's 30-day input sequence for one user."""

    if daily_spending.empty:
        raise ValueError("This user has no recorded expense transactions.")

    daily_spending = daily_spending.copy()
    daily_spending["transaction_date"] = pd.to_datetime(daily_spending["transaction_date"])
    daily_spending = daily_spending.sort_values("transaction_date")
    daily_spending = daily_spending.set_index("transaction_date")
    daily_spending = daily_spending[["daily_spending"]].resample("D").sum()

    latest_date = daily_spending.index.max()
    start_date = latest_date - pd.Timedelta(days=INPUT_DAYS - 1)
    daily_spending = daily_spending.loc[start_date:latest_date]

    full_dates = pd.date_range(start=start_date, end=latest_date, freq="D")
    daily_spending = daily_spending.reindex(full_dates, fill_value=0.0)

    if len(daily_spending) < INPUT_DAYS:
        missing_days = INPUT_DAYS - len(daily_spending)
        padding = pd.DataFrame(
            {"daily_spending": [0.0] * missing_days},
            index=pd.date_range(end=daily_spending.index.min() - pd.Timedelta(days=1), periods=missing_days, freq="D"),
        )
        daily_spending = pd.concat([padding, daily_spending])

    values = daily_spending["daily_spending"].to_numpy(dtype=np.float32)
    values = values[-INPUT_DAYS:]

    if len(values) != INPUT_DAYS:
        raise ValueError("Could not prepare the required 30-day input.")

    return values.reshape(1, INPUT_DAYS, 1)


# ==================================================
# CHECK MODEL ARTIFACTS
# ==================================================

def _resolve_existing_path(paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def check_model_artifacts() -> dict:
    """Report whether the compatible daily LSTM model and scaler are present."""

    model_path = _resolve_existing_path(MODEL_CANDIDATES)
    scaler_path = _resolve_existing_path(SCALER_CANDIDATES)

    return {
        "model_exists": model_path.is_file(),
        "scaler_exists": scaler_path.is_file(),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "input_days": INPUT_DAYS,
        "forecast_days": FORECAST_DAYS,
    }


def _load_daily_scaler():
    scaler_path = _resolve_existing_path(SCALER_CANDIDATES)
    if not scaler_path.is_file():
        raise RuntimeError("The daily-spending scaler is not available. Use the scaler saved by the matching preprocessing pipeline.")
    return joblib.load(scaler_path)


def _load_daily_model():
    model_path = _resolve_existing_path(MODEL_CANDIDATES)
    if not model_path.is_file():
        raise RuntimeError("The daily-spending LSTM checkpoint is not available. Complete training before generating predictions.")

    model = LSTMModel(hidden_size=64, num_layers=1, dropout=0.0)
    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, dict) and any(key.startswith("lstm") or key.startswith("fc") for key in checkpoint):
        model.load_state_dict(checkpoint)
    elif hasattr(checkpoint, "state_dict"):
        model.load_state_dict(checkpoint.state_dict())
    else:
        raise RuntimeError("The saved LSTM checkpoint format is incompatible with the expected architecture.")

    model.eval()
    return model


# ==================================================
# GENERATE SEVEN-DAY FORECAST
# ==================================================

def predict_user_spending(user_id: str) -> dict:
    """Generate a 7-day daily expenditure forecast for the given user."""

    daily_spending = get_user_daily_spending(user_id)
    if daily_spending.empty:
        raise ValueError("No expense history found for this user.")

    latest_date = pd.Timestamp(daily_spending["transaction_date"].max())
    input_sequence = prepare_user_input(daily_spending)

    artifacts = check_model_artifacts()
    if not artifacts["model_exists"]:
        raise RuntimeError("The daily-spending LSTM checkpoint is not available. Complete training before generating predictions.")
    if not artifacts["scaler_exists"]:
        raise RuntimeError("The daily-spending scaler is not available. Use the scaler saved by the matching preprocessing pipeline.")

    scaler = _load_daily_scaler()
    model = _load_daily_model()

    input_scaled = scaler.transform(input_sequence.reshape(-1, 1)).reshape(input_sequence.shape).astype(np.float32)
    with torch.no_grad():
        prediction_scaled = model(torch.from_numpy(input_scaled)).numpy()[0]

    prediction_raw = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).reshape(-1)
    prediction_raw = np.clip(prediction_raw, 0.0, None)

    forecast_dates = pd.date_range(start=latest_date + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="D")
    predictions = []
    total_predicted_expenditure = 0.0

    for forecast_date, value in zip(forecast_dates, prediction_raw):
        amount = round(float(value), 2)
        total_predicted_expenditure += amount
        predictions.append(
            {
                "date": forecast_date.strftime("%Y-%m-%d"),
                "predicted_amount": amount,
            }
        )

    return {
        "forecast_days": FORECAST_DAYS,
        "start_date": forecast_dates[0].strftime("%Y-%m-%d"),
        "end_date": forecast_dates[-1].strftime("%Y-%m-%d"),
        "total_predicted_expenditure": round(float(total_predicted_expenditure), 2),
        "predictions": predictions,
    }


__all__ = [
    "LSTMModel",
    "INPUT_DAYS",
    "FORECAST_DAYS",
    "get_user_daily_spending",
    "prepare_user_input",
    "check_model_artifacts",
    "predict_user_spending",
]