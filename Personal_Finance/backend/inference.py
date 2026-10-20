from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from ml.lstm.train_lstm import LSTMModel as OverallLSTM
from ml.lstm_categories.preprocessing.feature_extraction import (
    add_monthly_calendar_features,
    aggregate_monthly_category_spending,
    prepare_expense_data,
)
from ml.lstm_categories.train_lstm import LSTMModel as CategoryLSTM
from backend.spending_analytics import split_user_history


PROJECT_DIR = Path(__file__).resolve().parents[1]
OVERALL_MODEL_DIR = PROJECT_DIR / "ml" / "lstm" / "monthly"
CATEGORY_MODELS_DIR = PROJECT_DIR / "ml" / "lstm_categories" / "models" / "monthly"
DEVICE = torch.device("cpu")
HISTORY_MONTHS = 10
FORECAST_MONTHS = 7
CATEGORY_FEATURES = [
    "monthly_spending",
    "month_sin",
    "month_cos",
    "quarter_sin",
    "quarter_cos",
    "days_in_month",
]


def _category_dir(category):
    return (
        CATEGORY_MODELS_DIR
        / category.strip().lower().replace("/", "_").replace("\\", "_").replace(" ", "_")
    )


@lru_cache(maxsize=1)
def load_overall_model():
    model_path = OVERALL_MODEL_DIR / "lstm_model.pt"
    scaler_path = OVERALL_MODEL_DIR / "scaler.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(f"Monthly overall LSTM weights not found: {model_path}")
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Monthly overall LSTM scaler not found: {scaler_path}")

    model = OverallLSTM(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=FORECAST_MONTHS,
        dropout=0.2,
    )
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model, joblib.load(scaler_path)


@lru_cache(maxsize=32)
def load_category_model(category):
    category_dir = _category_dir(category)
    model_path = category_dir / "model.pt"
    scaler_path = category_dir / "scaler.pkl"
    if not model_path.is_file() or not scaler_path.is_file():
        raise FileNotFoundError(
            f"Monthly category LSTM artifacts not found for {category}"
        )

    model = CategoryLSTM()
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model, joblib.load(scaler_path)


def _complete_month_range(segment):
    if not segment:
        return None
    first = segment[0]["transaction_date"]
    last = segment[-1]["transaction_date"]
    first_month = pd.Timestamp(first).to_period("M")
    last_month = pd.Timestamp(last).to_period("M")
    if first.day != 1:
        first_month += 1
    if last.day != pd.Timestamp(last).days_in_month:
        last_month -= 1
    if first_month > last_month:
        return None
    return pd.period_range(first_month, last_month, freq="M")


def _expense_frame(segment):
    rows = [
        {
            "user_id": row.get("user_id"),
            "category": row.get("category"),
            "date": row["transaction_date"],
            "transaction_type": row.get("transaction_type"),
            "amount": row.get("amount"),
        }
        for row in segment
    ]
    if not rows:
        return pd.DataFrame(
            columns=["user_id", "category", "date", "transaction_type", "amount"]
        )
    expenses = prepare_expense_data(pd.DataFrame(rows))
    if not expenses.empty:
        expenses["date"] = pd.to_datetime(expenses["date"], errors="coerce")
    return expenses


def _monthly_overall_series(segment, periods):
    expenses = _expense_frame(segment)
    if expenses.empty:
        return np.zeros(len(periods), dtype=np.float32)
    expenses["month"] = pd.to_datetime(expenses["date"]).dt.to_period("M")
    monthly = expenses.groupby("month")["amount"].sum()
    return monthly.reindex(periods, fill_value=0.0).to_numpy(dtype=np.float32)


def _history_diagnostics(segment, diagnostics):
    result = dict(diagnostics)
    if segment:
        result["history_start_date"] = segment[0]["transaction_date"].isoformat()
        result["history_end_date"] = segment[-1]["transaction_date"].isoformat()
        result["is_stale"] = (
            date.today() - segment[-1]["transaction_date"]
        ).days > 7
    else:
        result["is_stale"] = True
    return result


def _eligible_monthly_segment(segments, minimum_months):
    eligible = []
    for segment in segments:
        periods = _complete_month_range(segment)
        if periods is not None and len(periods) >= minimum_months:
            eligible.append((segment, periods))
    return eligible[-1] if eligible else ([], None)


def predict_overall_monthly(transactions, account_created_at=None):
    segments, base_diagnostics = split_user_history(
        transactions, account_created_at
    )
    segment, periods = _eligible_monthly_segment(segments, HISTORY_MONTHS)
    diagnostics = _history_diagnostics(segment, base_diagnostics)
    if periods is None:
        return {
            "available": False,
            "reason": (
                f"At least {HISTORY_MONTHS} complete calendar months in one "
                "contiguous history segment are required."
            ),
            "values": [],
            "target_periods": [],
            "target_unit": "INR per calendar month",
            "diagnostics": diagnostics,
        }

    monthly_amounts = _monthly_overall_series(segment, periods)
    model, scaler = load_overall_model()
    scaled_input = scaler.transform(
        monthly_amounts[-HISTORY_MONTHS:].reshape(-1, 1)
    ).reshape(1, HISTORY_MONTHS, 1)
    with torch.inference_mode():
        scaled_predictions = (
            model(torch.from_numpy(scaled_input)).cpu().numpy().reshape(-1, 1)
        )
    values = scaler.inverse_transform(scaled_predictions).reshape(-1)
    first_target = periods[-1] + 1
    target_periods = [
        (first_target + offset).start_time.date().isoformat()
        for offset in range(FORECAST_MONTHS)
    ]
    return {
        "available": True,
        "horizon": "next 7 calendar months",
        "forecast_start": target_periods[0],
        "target_periods": target_periods,
        "target_unit": "INR per calendar month",
        "values": [round(max(0.0, float(value)), 2) for value in values],
        "diagnostics": diagnostics,
    }


def predict_category_monthly(transactions, account_created_at=None):
    segments, base_diagnostics = split_user_history(
        transactions, account_created_at
    )
    segment, periods = _eligible_monthly_segment(segments, 12)
    diagnostics = _history_diagnostics(segment, base_diagnostics)
    if periods is None:
        return {
            "available": False,
            "reason": (
                "At least 12 complete calendar months in one contiguous history "
                "segment are required."
            ),
            "categories": {},
            "errors": [],
            "target_periods": [],
            "target_unit": "INR per calendar month",
            "diagnostics": diagnostics,
        }

    expenses = _expense_frame(segment)
    if expenses.empty:
        return {
            "available": False,
            "reason": "No expense history is available.",
            "categories": {},
            "errors": [],
            "target_periods": [],
            "target_unit": "INR per calendar month",
            "diagnostics": diagnostics,
        }

    monthly = aggregate_monthly_category_spending(expenses)
    monthly = monthly[monthly["date"].dt.to_period("M").isin(periods)]
    categories = {}
    errors = []
    first_target = periods[-1] + 1
    target_periods = [
        (first_target + offset).start_time.date().isoformat()
        for offset in range(FORECAST_MONTHS)
    ]

    for category in sorted(monthly["category"].dropna().unique()):
        category_rows = monthly.loc[monthly["category"] == category].copy()
        category_rows["month"] = category_rows["date"].dt.to_period("M")
        category_months = category_rows.set_index("month")[
            "monthly_spending"
        ].reindex(periods, fill_value=0.0)
        frame = pd.DataFrame(
            {
                "date": periods.to_timestamp(),
                "monthly_spending": category_months.to_numpy(dtype=np.float32),
            }
        )
        history = add_monthly_calendar_features(frame)
        try:
            model, scaler = load_category_model(str(category))
            features = history[CATEGORY_FEATURES].to_numpy(dtype=np.float32)
            features[:, 0] = scaler.transform(
                features[:, 0].reshape(-1, 1)
            ).reshape(-1)
            with torch.inference_mode():
                scaled_predictions = (
                    model(torch.from_numpy(features[-12:].reshape(1, 12, -1)))
                    .cpu()
                    .numpy()
                    .reshape(-1, 1)
                )
            values = scaler.inverse_transform(scaled_predictions).reshape(-1)
            categories[str(category)] = [
                round(max(0.0, float(value)), 2) for value in values
            ]
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
            errors.append(f"{category}: {error}")

    return {
        "available": bool(categories),
        "reason": (
            None
            if categories
            else "No monthly category model could produce a forecast."
        ),
        "categories": categories,
        "errors": errors,
        "forecast_start": target_periods[0],
        "target_periods": target_periods,
        "target_unit": "INR per calendar month",
        "diagnostics": diagnostics,
    }
