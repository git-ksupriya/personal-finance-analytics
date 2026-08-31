import pandas as pd
import numpy as np
import re


def load_dataset(file_path):

    df = pd.read_csv(file_path)

    print("Rows    :", df.shape[0])
    print("Columns :", df.shape[1])

    return df


def handle_missing_values(df):

    print("\nHandling missing values...")

    df.replace(
        ["N/A", "NA", "", "null", "NULL"],
        np.nan,
        inplace=True
    )

    if "notes" in df.columns:
        df["notes"] = df["notes"].fillna("No Notes")

    if "location" in df.columns:
        df["location"] = df["location"].fillna("Unknown")

    df.dropna(
        subset=["user_id", "date", "amount"],
        inplace=True
    )

    return df


def remove_duplicates(df):

    duplicate_count = df.duplicated().sum()

    print("Duplicate rows:", duplicate_count)

    df.drop_duplicates(inplace=True)

    return df


def clean_amount_column(df):

    def clean_amount(value):

        value = str(value)

        value = re.sub(r"Rs\.?|₹|\$|,", "", value)

        value = re.sub(r"[^0-9.\-]", "", value)

        if value == "":
            return np.nan

        try:
            return float(value)
        except ValueError:
            return np.nan

    df["amount"] = df["amount"].apply(clean_amount)

    return df


def validate_amounts(df):

    df = df[df["amount"] > 0]

    return df


def validate_dates(df):

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df.dropna(subset=["date"], inplace=True)

    return df


def remove_outliers(df):

    # Same threshold used in your existing preprocessing
    df = df[df["amount"] < 1000000]

    return df