import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SEASONAL_NAIVE_DIR = Path(__file__).resolve().parent
MODELS_DIR = SEASONAL_NAIVE_DIR / "models"
PREDICTIONS_PATH = MODELS_DIR / "test_predictions.csv"
METRICS_PATH = MODELS_DIR / "seasonal_naive_metrics.json"
FORECAST_DAYS = 7
OVERALL_CATEGORY = "__overall__"


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)

    if actual.size == 0:
        raise ValueError("Cannot calculate metrics without test observations")
    if actual.shape != predicted.shape:
        raise ValueError(
            f"Actual and predicted shapes differ: {actual.shape} vs "
            f"{predicted.shape}"
        )

    mse = mean_squared_error(actual, predicted)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(actual, predicted)),
        "test_forecast_values": int(actual.size),
    }


def evaluate_predictions(predictions_path=PREDICTIONS_PATH, metrics_path=METRICS_PATH):
    predictions_path = Path(predictions_path)
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Test predictions not found: {predictions_path}. "
            "Run the seasonal-naive training step first."
        )

    data = pd.read_csv(predictions_path)
    required_columns = {"category"}
    for day in range(1, FORECAST_DAYS + 1):
        required_columns.add(f"actual_day_{day}")
        required_columns.add(f"predicted_day_{day}")
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Prediction file is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    actual_columns = [f"actual_day_{day}" for day in range(1, FORECAST_DAYS + 1)]
    predicted_columns = [
        f"predicted_day_{day}" for day in range(1, FORECAST_DAYS + 1)
    ]

    overall_data = data[data["category"] == OVERALL_CATEGORY]
    category_data = data[data["category"] != OVERALL_CATEGORY]
    if overall_data.empty:
        raise ValueError("Prediction file contains no overall daily forecasts")
    if category_data.empty:
        raise ValueError("Prediction file contains no category forecasts")

    metrics = {
        "model": "seasonal_naive",
        "forecast_days": FORECAST_DAYS,
        "overall": calculate_metrics(
            overall_data[actual_columns].to_numpy(),
            overall_data[predicted_columns].to_numpy(),
        ),
        "categories": {},
    }

    for category, category_predictions in category_data.groupby(
        "category",
        sort=True,
    ):
        metrics["categories"][str(category)] = calculate_metrics(
            category_predictions[actual_columns].to_numpy(),
            category_predictions[predicted_columns].to_numpy(),
        )

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"Test metrics saved: {metrics_path}")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate_predictions()
