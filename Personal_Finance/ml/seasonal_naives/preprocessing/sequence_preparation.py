from pathlib import Path

import numpy as np
import pandas as pd

from ml.lstm_categories.preprocessing.sequence_preparation import (
    CategorySequencePreparation,
)
from ml.lstm_categories.preprocessing.feature_extraction import (
    add_calendar_features,
    create_complete_daily_series,
)


SEASONAL_NAIVE_DIR = Path(__file__).resolve().parents[1]
PREPROCESSED_DIR = SEASONAL_NAIVE_DIR / "preprocessed"
DAILY_DATA_PATH = PREPROCESSED_DIR / "daily_category_spending.csv"

HISTORY_DAYS = 90
FORECAST_DAYS = 7
TRAIN_RATIO = 0.8
FEATURE_COUNT = 6
OVERALL_CATEGORY = "__overall__"


def _save_sequences(prefix, X, y, metadata, output_dir):
    if X.size == 0:
        X = np.empty((0, HISTORY_DAYS, FEATURE_COUNT), dtype=np.float32)
    if y.size == 0:
        y = np.empty((0, FORECAST_DAYS), dtype=np.float32)

    np.save(output_dir / f"X_{prefix}.npy", X)
    np.save(output_dir / f"y_{prefix}.npy", y)
    metadata.to_csv(output_dir / f"{prefix}_metadata.csv", index=False)


def prepare_sequences(daily_data_path=DAILY_DATA_PATH, output_dir=PREPROCESSED_DIR):
    daily_data_path = Path(daily_data_path)
    if not daily_data_path.is_file():
        raise FileNotFoundError(
            f"Daily category data not found: {daily_data_path}. "
            "Run the seasonal-naive feature extraction step first."
        )

    daily = pd.read_csv(daily_data_path)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")

    overall_daily = (
        daily.groupby(["user_id", "date"], as_index=False)["daily_spending"]
        .sum()
    )
    overall_daily["category"] = OVERALL_CATEGORY
    overall_daily = create_complete_daily_series(overall_daily)
    overall_daily = add_calendar_features(overall_daily)
    daily = pd.concat([daily, overall_daily], ignore_index=True)

    preparation = CategorySequencePreparation(
        history_days=HISTORY_DAYS,
        forecast_days=FORECAST_DAYS,
        train_ratio=TRAIN_RATIO,
    )
    daily = preparation.sort_data(daily)

    X_train, y_train, train_metadata = preparation.create_train_sequences(daily)
    X_test, y_test, test_metadata = preparation.create_test_sequences(daily)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_sequences("train", X_train, y_train, train_metadata, output_dir)
    _save_sequences("test", X_test, y_test, test_metadata, output_dir)

    print(f"Training windows: {len(X_train):,}")
    print(f"Test windows: {len(X_test):,}")
    print(f"Saved sequences and metadata: {output_dir}")
    return X_train, y_train, train_metadata, X_test, y_test, test_metadata


if __name__ == "__main__":
    prepare_sequences()
