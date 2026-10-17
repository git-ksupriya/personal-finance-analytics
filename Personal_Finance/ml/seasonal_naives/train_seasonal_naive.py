import json
from pathlib import Path

import numpy as np
import pandas as pd


SEASONAL_NAIVE_DIR = Path(__file__).resolve().parent
PREPROCESSED_DIR = SEASONAL_NAIVE_DIR / "preprocessed"
MODELS_DIR = SEASONAL_NAIVE_DIR / "models"

SEASON_LENGTH_DAYS = 7
HISTORY_DAYS = 90
FORECAST_DAYS = 7


def seasonal_naive_forecast(
    history,
    season_length=SEASON_LENGTH_DAYS,
    forecast_horizon=FORECAST_DAYS,
):
    history = np.asarray(history, dtype=float)

    if history.ndim != 1:
        raise ValueError("history must be a one-dimensional sequence")
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be positive")
    if len(history) < season_length:
        raise ValueError(
            f"history must contain at least {season_length} values"
        )

    last_complete_season = history[-season_length:]
    repeat_count = (forecast_horizon + season_length - 1) // season_length
    return np.tile(last_complete_season, repeat_count)[:forecast_horizon]


def train_seasonal_naive(
    preprocessed_dir=PREPROCESSED_DIR,
    models_dir=MODELS_DIR,
):
    preprocessed_dir = Path(preprocessed_dir)
    required_files = {
        name: preprocessed_dir / name
        for name in (
            "X_train.npy",
            "X_test.npy",
            "y_test.npy",
            "test_metadata.csv",
        )
    }
    missing = [str(path) for path in required_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required preprocessed data is missing; run feature extraction and "
            f"sequence preparation first: {', '.join(missing)}"
        )

    X_train = np.load(required_files["X_train.npy"])
    X_test = np.load(required_files["X_test.npy"])
    y_test = np.load(required_files["y_test.npy"])
    metadata = pd.read_csv(required_files["test_metadata.csv"])

    if X_train.ndim != 3 or X_train.shape[1] < SEASON_LENGTH_DAYS:
        raise ValueError(
            "Training inputs must contain daily sequences with at least one "
            "complete seven-day season"
        )
    if X_test.ndim != 3 or X_test.shape[1] < SEASON_LENGTH_DAYS:
        raise ValueError(
            "Test inputs must contain daily sequences with at least one "
            "complete seven-day season"
        )
    if y_test.shape != (len(X_test), FORECAST_DAYS):
        raise ValueError(
            f"Expected test targets with shape ({len(X_test)}, {FORECAST_DAYS}); "
            f"got {y_test.shape}"
        )
    if len(metadata) != len(X_test):
        raise ValueError(
            f"Test metadata rows ({len(metadata)}) do not match test windows "
            f"({len(X_test)})"
        )
    if len(X_train) == 0:
        raise ValueError("No training windows were generated")
    if len(X_test) == 0:
        raise ValueError("No test windows were generated")

    predictions = np.stack(
        [
            seasonal_naive_forecast(window[:, 0])
            for window in X_test
        ]
    )

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    prediction_data = metadata.reset_index(drop=True).copy()
    for day in range(FORECAST_DAYS):
        prediction_data[f"actual_day_{day + 1}"] = y_test[:, day]
        prediction_data[f"predicted_day_{day + 1}"] = predictions[:, day]

    predictions_path = models_dir / "test_predictions.csv"
    prediction_data.to_csv(predictions_path, index=False)

    model_description = {
        "model": "seasonal_naive",
        "season_length_days": SEASON_LENGTH_DAYS,
        "history_days": HISTORY_DAYS,
        "forecast_days": FORECAST_DAYS,
        "train_windows": int(len(X_train)),
        "test_windows": int(len(X_test)),
        "forecast_rule": (
            "Repeat the corresponding values from the most recent complete "
            "seven-day period."
        ),
    }
    model_path = models_dir / "seasonal_naive_model.json"
    with model_path.open("w", encoding="utf-8") as model_file:
        json.dump(model_description, model_file, indent=2)

    print(f"Baseline configuration saved: {model_path}")
    print(f"Test predictions saved: {predictions_path}")
    return prediction_data


if __name__ == "__main__":
    train_seasonal_naive()
