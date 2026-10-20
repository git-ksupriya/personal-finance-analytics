
import numpy as np
import pandas as pd
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = (
    BASE_DIR / "xgboost" / "preprocessing" /
    "processed" / "monthly_features.csv"
)
OUTPUT_DIR = DATA_FILE.parent
MODEL_PATH = BASE_DIR.parent / "monthly_xgboost_model.json"

FEATURES = [
    "expense_lag_1",
    "expense_lag_2",
    "expense_lag_3",
    "transaction_count_lag_1",
    "expense_mean_3",
    "expense_mean_6",
    "month_number",
    "target_month_number",
]


def evaluate(name, actual, predicted):
    print(f"\n{name}")
    print("-" * 40)
    print("MAE :", round(mean_absolute_error(actual, predicted), 2))
    print("RMSE:", round(np.sqrt(mean_squared_error(actual, predicted)), 2))
    print("R2  :", round(r2_score(actual, predicted), 4))


def main():
    df = pd.read_csv(DATA_FILE)
    df["target_month"] = pd.PeriodIndex(df["target_month"], freq="M")

    # Train on earlier months and test on the latest months
    train = df[df["target_month"] <= "2024-06"].copy()
    test = df[df["target_month"] > "2024-06"].copy()

    X_train = train[FEATURES]
    y_train = train["target"]
    X_test = test[FEATURES]
    y_test = test["target"]

    print("Training rows:", len(train))
    print("Testing rows:", len(test))
    print("Training target months:",
          train["target_month"].min(), "to", train["target_month"].max())
    print("Testing target months:",
          test["target_month"].min(), "to", test["target_month"].max())

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=5,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    print("\nTraining monthly XGBoost...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    evaluate("Monthly XGBoost", y_test, predictions)

    # Predict each user's training-period average as a simple baseline
    user_means = train.groupby("user_id")["target"].mean()
    baseline = test["user_id"].map(user_means).fillna(y_train.mean())

    evaluate("User-average baseline", y_test, baseline.to_numpy())

    # Save model, predictions, and feature importance
    model.save_model(str(MODEL_PATH))

    pd.DataFrame({
        "user_id": test["user_id"].to_numpy(),
        "target_month": test["target_month"].astype(str).to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": predictions,
        "baseline": baseline.to_numpy()
    }).to_csv(OUTPUT_DIR / "monthly_xgboost_predictions.csv", index=False)

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        OUTPUT_DIR / "monthly_xgboost_feature_importance.csv",
        index=False
    )

    print("\nModel saved:", MODEL_PATH)
    print("Predictions saved to monthly_xgboost_predictions.csv")
    print("\nFeature importance:")
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
