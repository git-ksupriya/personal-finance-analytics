'''
Feature Engineering
Builds on top of the standardised dataset (processed/standardised_dataset.csv)
'''

import pandas as pd
import os


def extract_date_parts(df):
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.day_name()
    return df

def calculate_days_between_transactions(df):
    """Gap in days since each user's previous transaction. First txn per user is NaN."""
    df = df.sort_values(["user_id", "date"])
    df["days_since_last_txn"] = df.groupby("user_id")["date"].diff().dt.days
    return df

def calculate_monthly_spending(df):
    """Total Expense amount per user per calendar month, merged back onto every row."""
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    expense = df[df["transaction_type"] == "Expense"]
    monthly_total = (
        expense.groupby(["user_id", "year_month"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "monthly_total_spend"})
    )

    df = df.merge(monthly_total, on=["user_id", "year_month"], how="left")
    return df
def calculate_category_spending(df):
    """Total Expense amount per user, per month, per category. Requires 'year_month' column
    (created in calculate_monthly_spending) — run that function first."""
    expense = df[df["transaction_type"] == "Expense"]
    category_monthly = (
        expense.groupby(["user_id", "year_month", "category"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "category_monthly_spend"})
    )

    df = df.merge(category_monthly, on=["user_id", "year_month", "category"], how="left")
    return df

def calculate_transaction_frequency(df):
    """Total number of transactions each user has made, added as a column on every row."""
    df["user_txn_frequency"] = df.groupby("user_id")["transaction_id"].transform("count")
    return df

def calculate_rolling_average(df, window=3):
    """Rolling average of transaction amount over the last N transactions, per user."""
    df = df.sort_values(["user_id", "date"])
    df[f"rolling_avg_{window}"] = (
        df.groupby("user_id")["amount"]
        .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )
    return df

def main():
    df = pd.read_csv("processed/standardised_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])

    #df = pd.read_csv("processed/standardised_dataset.csv")
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed/standardised_dataset.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = extract_date_parts(df)
    df = calculate_days_between_transactions(df)
    df = calculate_monthly_spending(df)          # must run before calculate_category_spending
    df = calculate_category_spending(df)          # depends on year_month from the line above
    df = calculate_transaction_frequency(df)
    df = calculate_rolling_average(df)


    df.to_csv("processed/feature_engineered_dataset.csv", index=False)
    print("Feature engineered dataset saved.")

    #df.to_csv("processed/feature_engineered_dataset.csv", index=False)
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed/feature_engineered_dataset.csv"), index=False)
    print("Feature engineered dataset saved.")


if __name__ == "__main__":
    main()
