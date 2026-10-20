
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"

INPUT_FILE = PROCESSED_DIR / "monthly_expenses.csv"
OUTPUT_FILE = PROCESSED_DIR / "monthly_features.csv"


def main():
    # Load monthly expense data
    df = pd.read_csv(INPUT_FILE)

    df["month"] = pd.PeriodIndex(df["month"], freq="M")

    # Sort each user's records chronologically
    df = df.sort_values(
        ["user_id", "month"]
    ).reset_index(drop=True)

    grouped = df.groupby("user_id", sort=False)

    # Previous completed months' expenses
    # The current month is the latest known month.
    df["expense_lag_1"] = df["monthly_expense"]
    df["expense_lag_2"] = grouped["monthly_expense"].shift(1)
    df["expense_lag_3"] = grouped["monthly_expense"].shift(2)

    # Transaction count in the latest known month
    df["transaction_count_lag_1"] = df["transaction_count"]

    # Average expenses over the latest 3 months
    df["expense_mean_3"] = grouped["monthly_expense"].transform(
        lambda s: s.rolling(window=3, min_periods=3).mean()
    )

    # Average expenses over the latest 6 months
    df["expense_mean_6"] = grouped["monthly_expense"].transform(
        lambda s: s.rolling(window=6, min_periods=6).mean()
    )

    # Predict the next calendar month's expenses
    df["target_month"] = grouped["month"].shift(-1)
    df["target"] = grouped["monthly_expense"].shift(-1)

    # Month number of the month being predicted
    df["target_month_number"] = df["target_month"].map(
        lambda m: m.month if pd.notna(m) else None
    )

    # Remove rows without sufficient history or a future target
    df = df.dropna(
        subset=[
            "expense_lag_1",
            "expense_lag_2",
            "expense_lag_3",
            "transaction_count_lag_1",
            "expense_mean_3",
            "expense_mean_6",
            "target",
            "target_month_number",
        ]
    ).copy()

    df["target_month_number"] = (
        df["target_month_number"].astype(int)
    )

    # Calendar month of the latest completed month
    df["month_number"] = df["month"].map(
        lambda m: m.month
    )

    # Convert Period columns to strings before saving
    df["month"] = df["month"].astype(str)
    df["target_month"] = df["target_month"].astype(str)

    # Save the forecasting dataset
    df.to_csv(OUTPUT_FILE, index=False)

    features = [
        "expense_lag_1",
        "expense_lag_2",
        "expense_lag_3",
        "transaction_count_lag_1",
        "expense_mean_3",
        "expense_mean_6",
        "month_number",
        "target_month_number",
    ]

    print("Monthly forecasting features created.")
    print("Rows:", len(df))
    print("Users:", df["user_id"].nunique())
    print("Features:", features)
    print("Missing values:", int(df.isna().sum().sum()))
    print("Saved to:", OUTPUT_FILE)

    print("\nSample:")
    print(
        df[
            [
                "user_id",
                "month",
                "expense_lag_1",
                "expense_lag_2",
                "expense_lag_3",
                "target_month",
                "target",
            ]
        ].head(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()
