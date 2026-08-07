import pandas as pd
import numpy as np
import re

# =====================================================
# Load Dataset
# =====================================================

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    print("=" * 60)
    print("Dataset Loaded Successfully")
    print("=" * 60)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}\n")
    return df


# =====================================================
# Dataset Overview
# =====================================================

def dataset_overview(df):
    print("\n========== DATASET INFO ==========\n")
    print(df.info())

    print("\n========== FIRST 5 ROWS ==========\n")
    print(df.head())

    print("\n========== DATA TYPES ==========\n")
    print(df.dtypes)


# =====================================================
# Check Missing Values
# =====================================================

def check_missing_values(df):

    print("\n========== MISSING VALUES ==========\n")

    missing = df.isnull().sum()

    print(missing)

    return missing


# =====================================================
# Handle Missing Values
# =====================================================

def handle_missing_values(df):

    print("\nHandling Missing Values...")

    # Replace common missing representations
    df.replace(["N/A", "NA", "", "null", "NULL"], np.nan, inplace=True)

    # Fill Notes
    if "notes" in df.columns:
        df["notes"] = df["notes"].fillna("No Notes")

    # Fill Location
    if "location" in df.columns:
        df["location"] = df["location"].fillna("Unknown")

    # Remove rows missing essential fields
    df.dropna(subset=["user_id", "date", "amount"], inplace=True)

    return df


# =====================================================
# Remove Duplicates
# =====================================================

def remove_duplicates(df):

    duplicate_count = df.duplicated().sum()

    print(f"\nDuplicate Rows Found : {duplicate_count}")

    df.drop_duplicates(inplace=True)

    print(f"Duplicate Rows Removed : {duplicate_count}")

    return df


# =====================================================
# Validate Transaction IDs
# =====================================================

def validate_transaction_ids(df):

    print("\n========== TRANSACTION ID CHECK ==========\n")

    duplicate_ids = df["transaction_id"].duplicated().sum()

    print("Duplicate Transaction IDs :", duplicate_ids)

    print("Unique Transaction IDs :", df["transaction_id"].nunique())


# =====================================================
# Validate User IDs
# =====================================================

def validate_user_ids(df):

    print("\n========== USER ID CHECK ==========\n")

    print("Unique Users :", df["user_id"].nunique())

    print("\nTop 10 Users by Transactions\n")

    print(df["user_id"].value_counts().head(10))


# =====================================================
# Clean Amount Column
# =====================================================

def clean_amount_column(df):

    print("\nCleaning Amount Column...")

    def clean_amount(value):

        value = str(value)

        value = re.sub(r"Rs\.?|₹|\$|,", "", value)

        value = re.sub(r"[^0-9.\-]", "", value)

        if value == "":
            return np.nan

        return float(value)

    df["amount"] = df["amount"].apply(clean_amount)

    return df


# =====================================================
# Validate Amounts
# =====================================================

def validate_amounts(df):

    print("\n========== AMOUNT SUMMARY ==========\n")

    print(df["amount"].describe())

    negative = len(df[df["amount"] < 0])

    zero = len(df[df["amount"] == 0])

    print("\nNegative Amounts :", negative)

    print("Zero Amounts :", zero)

    # Remove invalid amounts
    df = df[df["amount"] > 0]

    return df


# =====================================================
# Validate Dates
# =====================================================

def validate_dates(df):

    print("Removing rows with Date as nan")

    df.dropna(subset=["date"], inplace=True)

    return df


# =====================================================
# Remove Unrealistic Amounts
# =====================================================

def remove_outliers(df):

    print("\nRemoving Unrealistic Amounts...")

    before = len(df)

    df = df[df["amount"] < 1000000]

    removed = before - len(df)

    print("Outlier Records Removed :", removed)

    return df


# =====================================================
# Cleaning Report
# =====================================================

def cleaning_report(original_rows, cleaned_df):

    print("\n" + "=" * 60)

    print("DATA CLEANING REPORT")

    print("=" * 60)

    print(f"Original Rows : {original_rows}")

    print(f"Final Rows    : {len(cleaned_df)}")

    print(f"Rows Removed  : {original_rows - len(cleaned_df)}")

    print("=" * 60)


# =====================================================
# Save Dataset
# =====================================================

def save_dataset(df, filename):

    df.to_csv(filename, index=False)

    print(f"\nDataset saved as {filename}")


# =====================================================
# MAIN
# =====================================================

def main():

    file_path = r"raw\budgetwise_finance_dataset.csv"

    df = load_dataset(file_path)

    original_rows = len(df)

    dataset_overview(df)

    check_missing_values(df)

    df = handle_missing_values(df)

    df = remove_duplicates(df)

    validate_transaction_ids(df)

    validate_user_ids(df)

    df = clean_amount_column(df)

    df = validate_amounts(df)

    df = validate_dates(df)

    df = remove_outliers(df)

    cleaning_report(original_rows, df)

    save_dataset(df, "processed/cleaned_dataset.csv")


if __name__ == "__main__":
    main()