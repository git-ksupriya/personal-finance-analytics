import pandas as pd
import numpy as np
import re
from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Current file:
# Personal_Finance/ml/models/cnn/preprocessing/data_cleaning.py

CURRENT_DIR = Path(__file__).resolve().parent

# Go up to Personal_Finance
PERSONAL_FINANCE_DIR = CURRENT_DIR.parents[3]

# Raw dataset
RAW_DATA_PATH = (
    PERSONAL_FINANCE_DIR
    / "preprocessing"
    / "raw"
    / "budgetwise_finance_dataset.csv"
)

# CNN preprocessing output directory
PROCESSED_DIR = CURRENT_DIR.parent / "processed"

# Create processed directory if it doesn't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(file_path):

    df = pd.read_csv(file_path)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


# ============================================================
# CHECK MISSING VALUES
# ============================================================

def check_missing_values(df):

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df):

    print("\nHandling Missing Values...")

    # Replace common missing representations
    df.replace(
        ["N/A", "NA", "", "null", "NULL"],
        np.nan,
        inplace=True
    )

    # Fill missing notes
    if "notes" in df.columns:
        df["notes"] = df["notes"].fillna("No Notes")

    # Fill missing locations
    if "location" in df.columns:
        df["location"] = df["location"].fillna("Unknown")

    # Remove rows missing essential fields
    df.dropna(
        subset=["user_id", "date", "amount"],
        inplace=True
    )

    return df


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):

    duplicate_count = df.duplicated().sum()

    print(f"\nDuplicate Rows Found : {duplicate_count}")

    df = df.drop_duplicates()

    print(f"Duplicate Rows Removed : {duplicate_count}")

    return df


# ============================================================
# CLEAN AMOUNT COLUMN
# ============================================================

def clean_amount_column(df):

    def clean_amount(value):

        value = str(value)

        # Remove currency symbols and commas
        value = re.sub(r"Rs\.?|₹|\$", "", value)
        value = re.sub(r",", "", value)

        # Keep only numbers, decimal point and minus sign
        value = re.sub(r"[^0-9.\-]", "", value)

        if value == "":
            return np.nan

        return float(value)

    df["amount"] = df["amount"].apply(clean_amount)

    return df


# ============================================================
# VALIDATE AMOUNTS
# ============================================================

def validate_amounts(df):

    df = df[df["amount"] > 0]

    return df


# ============================================================
# VALIDATE DATES
# ============================================================

def validate_dates(df):

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(subset=["date"])

    return df


# ============================================================
# REMOVE UNREALISTIC AMOUNTS
# ============================================================

def remove_outliers(df):

    print("\nRemoving unrealistic amounts...")

    before = len(df)

    df = df[df["amount"] < 1000000]

    removed = before - len(df)

    print(f"Outliers removed : {removed}")

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CNN DATA CLEANING")
    print("=" * 60)

    print("\nDataset path:")
    print(RAW_DATA_PATH)

    # Load dataset
    df = load_dataset(RAW_DATA_PATH)

    original_rows = len(df)

    # Cleaning pipeline
    check_missing_values(df)

    df = handle_missing_values(df)

    df = remove_duplicates(df)

    df = clean_amount_column(df)

    df = validate_amounts(df)

    df = validate_dates(df)

    df = remove_outliers(df)

    # Cleaning report
    print("\n========== CLEANING REPORT ==========")

    print("Original rows :", original_rows)
    print("Final rows    :", len(df))
    print("Rows removed  :", original_rows - len(df))

    # Save cleaned dataset
    output_path = PROCESSED_DIR / "cleaned_dataset.csv"

    df.to_csv(output_path, index=False)

    print("\nCleaned dataset saved to:")
    print(output_path)


if __name__ == "__main__":
    main()