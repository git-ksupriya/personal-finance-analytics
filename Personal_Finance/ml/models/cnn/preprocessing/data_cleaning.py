import pandas as pd
import numpy as np
import re


def load_dataset(file_path):
    df = pd.read_csv(file_path)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


def check_missing_values(df):
    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())


def handle_missing_values(df):

    print("\nHandling Missing Values...")

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

    print(f"\nDuplicate Rows Found : {duplicate_count}")

    df = df.drop_duplicates()

    print(f"Duplicate Rows Removed : {duplicate_count}")

    return df


def clean_amount_column(df):

    def clean_amount(value):

        value = str(value)

        value = re.sub(r"Rs\.?|₹|\$|,", "", value)
        value = re.sub(r"[^0-9.\-]", "", value)

        if value == "":
            return np.nan

        return float(value)

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

    df = df.dropna(subset=["date"])

    return df


def remove_outliers(df):

    print("\nRemoving unrealistic amounts...")

    before = len(df)

    df = df[df["amount"] < 1000000]

    removed = before - len(df)

    print(f"Outliers removed : {removed}")

    return df


def main():

    file_path = (
        "../../../preprocessing/raw/"
        "budgetwise_finance_dataset.csv"
    )

    df = load_dataset(file_path)

    original_rows = len(df)

    check_missing_values(df)

    df = handle_missing_values(df)

    df = remove_duplicates(df)

    df = clean_amount_column(df)

    df = validate_amounts(df)

    df = validate_dates(df)

    df = remove_outliers(df)

    print("\n========== CLEANING REPORT ==========")
    print("Original rows :", original_rows)
    print("Final rows    :", len(df))
    print("Rows removed  :", original_rows - len(df))

    output_path = "processed/cleaned_dataset.csv"

    df.to_csv(output_path, index=False)

    print("\nCleaned dataset saved to:")
    print(output_path)


if __name__ == "__main__":
    main()