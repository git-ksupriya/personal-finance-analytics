
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"

INPUT_FILE = PROCESSED_DIR / "monthly_features.csv"
OUTPUT_FILE = PROCESSED_DIR / "monthly_features_v2.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    df["target_month"] = pd.PeriodIndex(
        df["target_month"], freq="M"
    )

    df = df.sort_values(
        ["user_id", "month"]
    ).reset_index(drop=True)

    grouped = df.groupby("user_id", sort=False)

    # Historical user spending statistics.
    # Shift first so the current target month is never included.
    df["user_expense_mean"] = grouped["monthly_expense"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    df["user_expense_std"] = grouped["monthly_expense"].transform(
        lambda s: s.shift(1).expanding().std()
    )

    # Cyclical representation of the month being predicted.
    import numpy as np

    month = df["target_month"].map(lambda m: m.month)

    df["target_month_sin"] = np.sin(
        2 * np.pi * month / 12
    )

    df["target_month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("User-specific and seasonal features created.")
    print("Rows:", len(df))
    print("Users:", df["user_id"].nunique())
    print("Missing values:")
    print(
        df[
            [
                "user_expense_mean",
                "user_expense_std",
                "target_month_sin",
                "target_month_cos",
            ]
        ].isna().sum().to_string()
    )
    print("Saved to:", OUTPUT_FILE)

    print("\nSample:")
    print(
        df[
            [
                "user_id",
                "target_month",
                "user_expense_mean",
                "user_expense_std",
                "target_month_sin",
                "target_month_cos",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
