import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"


def main():
    actual = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    )["target"]

    results = pd.read_csv(
        PROCESSED_DIR / "xgboost_predictions.csv"
    )

    predicted = results["predicted"]

    print("XGBoost Evaluation")
    print("=" * 35)

    print("MAE :", mean_absolute_error(actual, predicted))
    print("RMSE:", np.sqrt(mean_squared_error(actual, predicted)))
    print("R2  :", r2_score(actual, predicted))

    print("\nAbsolute error distribution:")
    print((actual - predicted).abs().describe())


if __name__ == "__main__":
    main()