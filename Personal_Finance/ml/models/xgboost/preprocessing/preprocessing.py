import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"


def main():
    input_path = PROCESSED_DIR / "xgboost_dataset.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            "Run feature_extraction.py first."
        )

    df = pd.read_csv(input_path, parse_dates=["target_date"])

    numeric_features = [
        "amount", "year", "month", "day", "weekday",
        "month_start", "month_end",
        "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6",
        "rolling_mean_3", "rolling_mean_5", "rolling_mean_10",
        "rolling_std_5", "rolling_std_10",
        "days_since_last_transaction",
        "user_mean_amount", "user_median_amount",
        "user_transaction_count"
    ]

    categorical_features = [
        "transaction_type", "category",
        "payment_mode", "location"
    ]

    df = df.sort_values("target_date").reset_index(drop=True)

    split = int(len(df) * 0.8)

    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    X_train = train[
        numeric_features + categorical_features
    ].copy()

    X_test = test[
        numeric_features + categorical_features
    ].copy()

    # Encode categorical features
    X_train = pd.get_dummies(
        X_train,
        columns=categorical_features,
        dummy_na=True,
        dtype=int
    )

    X_test = pd.get_dummies(
        X_test,
        columns=categorical_features,
        dummy_na=True,
        dtype=int
    )

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    X_train = X_train.replace(
        [float("inf"), -float("inf")], float("nan")
    )

    X_test = X_test.replace(
        [float("inf"), -float("inf")], float("nan")
    )

    imputer = SimpleImputer(strategy="median")

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_train.columns
    )

    y_train = train["target"]
    y_test = test["target"]

    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)

    y_train.to_frame("target").to_csv(
        PROCESSED_DIR / "y_train.csv", index=False
    )

    y_test.to_frame("target").to_csv(
        PROCESSED_DIR / "y_test.csv", index=False
    )

    pd.Series(X_train.columns).to_csv(
        PROCESSED_DIR / "feature_columns.csv",
        index=False,
        header=False
    )

    print("Preprocessing completed.")
    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)


if __name__ == "__main__":
    main()