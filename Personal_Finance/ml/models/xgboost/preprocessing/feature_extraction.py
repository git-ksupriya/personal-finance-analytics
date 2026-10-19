import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"


def main():
    input_path = PROCESSED_DIR / "standardized_transactions.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            "Run data_standardisation.py first."
        )

    df = pd.read_csv(input_path, parse_dates=["date"])

    df = df.sort_values(
        ["user_id", "date", "transaction_id"],
        kind="stable"
    ).reset_index(drop=True)

    amount_group = df.groupby("user_id")["amount"]
    date_group = df.groupby("user_id")["date"]

    # Previous transaction amounts
    for lag in range(1, 7):
        df[f"lag_{lag}"] = amount_group.shift(lag)

    # Rolling averages using previous transactions only
    for window in [3, 5, 10]:
        df[f"rolling_mean_{window}"] = (
            amount_group.transform(
                lambda s: s.shift(1).rolling(
                    window, min_periods=1
                ).mean()
            )
        )

    # Historical spending variability
    for window in [5, 10]:
        df[f"rolling_std_{window}"] = (
            amount_group.transform(
                lambda s: s.shift(1).rolling(
                    window, min_periods=2
                ).std()
            )
        )

    # User-specific historical averages
    df["user_mean_amount"] = amount_group.transform(
        lambda s: s.shift(1).expanding(
            min_periods=1
        ).mean()
    )

    df["user_median_amount"] = amount_group.transform(
        lambda s: s.shift(1).expanding(
            min_periods=1
        ).median()
    )

    df["user_transaction_count"] = (
        df.groupby("user_id").cumcount()
    )

    # Time-related features
    df["days_since_last_transaction"] = (
        date_group.diff().dt.days
    )

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.dayofweek

    df["month_start"] = df["date"].dt.is_month_start.astype(int)
    df["month_end"] = df["date"].dt.is_month_end.astype(int)

    # Target is the next transaction for the same user
    df["target"] = amount_group.shift(-1)
    df["target_date"] = date_group.shift(-1)

    df = df.dropna(
        subset=["target", "target_date"]
    ).copy()

    df = df.sort_values(
        ["target_date", "user_id"],
        kind="stable"
    )

    output_path = PROCESSED_DIR / "xgboost_dataset.csv"
    df.to_csv(output_path, index=False)

    print("Feature extraction completed.")
    print("Dataset shape:", df.shape)
    print("\nTarget distribution:")
    print(df["target"].describe())

    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()