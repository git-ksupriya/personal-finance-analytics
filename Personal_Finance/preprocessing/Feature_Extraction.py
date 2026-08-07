'''
Feature Engineering
Builds on top of the standardised dataset (processed/standardised_dataset.csv)
'''

import pandas as pd


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