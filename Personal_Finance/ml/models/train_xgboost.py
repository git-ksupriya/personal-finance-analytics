
import numpy as np
import pandas as pd
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = (
    BASE_DIR / "xgboost" / "preprocessing" / "processed"
)

MODEL_PATH = BASE_DIR.parent / "xgboost_model.json"


def show_metrics(name, actual, predicted):
    if len(actual) == 0:
        print(f"\n{name}: No records")
        return

    print(f"\n{name}")
    print("-" * 40)
    print("Records:", len(actual))
    print("MAE :", round(mean_absolute_error(actual, predicted), 2))
    print("RMSE:", round(np.sqrt(mean_squared_error(actual, predicted)), 2))
    print("R2  :", round(r2_score(actual, predicted), 4))


def main():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")

    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["target"]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")["target"]

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=5,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    print("Training XGBoost...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Evaluate all test transactions
    show_metrics("XGBoost - All Transactions", y_test, predictions)

    # Median baseline
    baseline = np.full(len(y_test), y_train.median())
    show_metrics("Median Baseline - All Transactions", y_test, baseline)

    # Separate ordinary and high-value transactions
    high_mask = y_test.abs() >= 100000
    normal_mask = ~high_mask

    show_metrics(
        "XGBoost - Transactions Below 100000",
        y_test[normal_mask],
        predictions[normal_mask]
    )

    show_metrics(
        "XGBoost - Transactions At Least 100000",
        y_test[high_mask],
        predictions[high_mask]
    )

    # Save model and predictions
    model.save_model(str(MODEL_PATH))

    pd.DataFrame({
        "actual": y_test,
        "predicted": predictions
    }).to_csv(
        PROCESSED_DIR / "xgboost_predictions.csv",
        index=False
    )

    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        PROCESSED_DIR / "xgboost_feature_importance.csv",
        index=False
    )

    print("\nModel saved:", MODEL_PATH)
    print("\nPredictions saved:", PROCESSED_DIR / "xgboost_predictions.csv")

    print("\nTop 15 features:")
    print(importance.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
