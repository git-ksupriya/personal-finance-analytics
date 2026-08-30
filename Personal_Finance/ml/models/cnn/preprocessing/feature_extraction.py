import pandas as pd
from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Current directory:
# Personal_Finance/ml/models/cnn/preprocessing/

CURRENT_DIR = Path(__file__).resolve().parent

# CNN preprocessing directory
PROCESSED_DIR = CURRENT_DIR.parent / "processed"

# Make sure processed directory exists
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXTRACT DATE FEATURES
# ============================================================

def extract_date_parts(df):

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.day_name()

    return df


# ============================================================
# DAYS BETWEEN TRANSACTIONS
# ============================================================

def calculate_days_between_transactions(df):

    df = df.sort_values(
        ["user_id", "date"]
    )

    df["days_since_last_txn"] = (
        df.groupby("user_id")["date"]
        .diff()
        .dt.days
    )

    # First transaction of each user has no previous transaction
    df["days_since_last_txn"] = (
        df["days_since_last_txn"]
        .fillna(0)
    )

    return df


# ============================================================
# MONTHLY SPENDING
# ============================================================

def calculate_monthly_spending(df):

    df["year_month"] = (
        df["date"]
        .dt.to_period("M")
        .astype(str)
    )

    # Consider only expenses
    expense = df[
        df["transaction_type"] == "Expense"
    ]

    monthly_total = (
        expense
        .groupby(
            ["user_id", "year_month"]
        )["amount"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "amount": "monthly_total_spend"
            }
        )
    )

    df = df.merge(
        monthly_total,
        on=["user_id", "year_month"],
        how="left"
    )

    df["monthly_total_spend"] = (
        df["monthly_total_spend"]
        .fillna(0)
    )

    return df


# ============================================================
# CATEGORY-WISE SPENDING
# ============================================================

def calculate_category_spending(df):

    # Consider only expenses
    expense = df[
        df["transaction_type"] == "Expense"
    ]

    category_monthly = (
        expense
        .groupby(
            [
                "user_id",
                "year_month",
                "category"
            ]
        )["amount"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "amount": "category_monthly_spend"
            }
        )
    )

    df = df.merge(
        category_monthly,
        on=[
            "user_id",
            "year_month",
            "category"
        ],
        how="left"
    )

    df["category_monthly_spend"] = (
        df["category_monthly_spend"]
        .fillna(0)
    )

    return df


# ============================================================
# TRANSACTION FREQUENCY
# ============================================================

def calculate_transaction_frequency(df):

    df["user_txn_frequency"] = (
        df.groupby("user_id")["transaction_id"]
        .transform("count")
    )

    return df


# ============================================================
# ROLLING AVERAGE
# ============================================================

def calculate_rolling_average(df, window=3):

    df = df.sort_values(
        ["user_id", "date"]
    )

    df[f"rolling_avg_{window}"] = (
        df.groupby("user_id")["amount"]
        .transform(
            lambda s:
            s.rolling(
                window,
                min_periods=1
            ).mean()
        )
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CNN FEATURE EXTRACTION")
    print("=" * 60)

    # Input dataset
    input_path = (
        PROCESSED_DIR
        / "standardised_dataset.csv"
    )

    print("\nReading:")
    print(input_path)

    df = pd.read_csv(input_path)

    # Convert date to datetime
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    print("\nInitial shape:", df.shape)

    # Feature extraction pipeline
    df = extract_date_parts(df)

    df = calculate_days_between_transactions(df)

    df = calculate_monthly_spending(df)

    df = calculate_category_spending(df)

    df = calculate_transaction_frequency(df)

    df = calculate_rolling_average(df)

    # Output path
    output_path = (
        PROCESSED_DIR
        / "feature_engineered_dataset.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\nFinal shape:", df.shape)

    print("\nFeature engineered dataset saved to:")
    print(output_path)


if __name__ == "__main__":
    main()