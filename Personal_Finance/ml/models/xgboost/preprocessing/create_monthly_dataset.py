
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"

INPUT_FILE = PROCESSED_DIR / "cleaned_transactions.csv"
OUTPUT_FILE = PROCESSED_DIR / "monthly_expenses.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(
        df["date"], dayfirst=True, errors="coerce"
    )
    df["transaction_type"] = (
        df["transaction_type"].astype(str).str.strip().str.title()
    )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    expenses = df[
        (df["transaction_type"] == "Expense")
        & df["date"].notna()
        & df["amount"].notna()
    ].copy()

    expenses["month"] = expenses["date"].dt.to_period("M")

    monthly = (
        expenses.groupby(["user_id", "month"], as_index=False)
        .agg(
            monthly_expense=("amount", "sum"),
            transaction_count=("amount", "size"),
            average_transaction=("amount", "mean")
        )
    )

    # Add every calendar month for each user
    all_months = pd.period_range(
        start=expenses["date"].min().to_period("M"),
        end=expenses["date"].max().to_period("M"),
        freq="M"
    )

    users = monthly["user_id"].unique()
    full_index = pd.MultiIndex.from_product(
        [users, all_months],
        names=["user_id", "month"]
    )

    monthly = (
        monthly.set_index(["user_id", "month"])
        .reindex(full_index)
        .reset_index()
    )

    # No recorded expense transactions in a month
    monthly["monthly_expense"] = monthly["monthly_expense"].fillna(0)
    monthly["transaction_count"] = (
        monthly["transaction_count"].fillna(0).astype(int)
    )
    monthly["average_transaction"] = (
        monthly["average_transaction"].fillna(0)
    )

    monthly["month"] = monthly["month"].astype(str)
    monthly = monthly.sort_values(
        ["user_id", "month"]
    ).reset_index(drop=True)

    monthly.to_csv(OUTPUT_FILE, index=False)

    print("Continuous monthly dataset created.")
    print("Rows:", len(monthly))
    print("Users:", monthly["user_id"].nunique())
    print("Months:", monthly["month"].nunique())
    print("Missing monthly expense values:", monthly["monthly_expense"].isna().sum())
    print("Saved to:", OUTPUT_FILE)

    print("\nSample records:")
    print(monthly.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
