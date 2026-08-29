import pandas as pd


def extract_date_parts(df):

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.day_name()

    return df


def calculate_days_between_transactions(df):

    df = df.sort_values(
        ["user_id", "date"]
    )

    df["days_since_last_txn"] = (
        df.groupby("user_id")["date"]
        .diff()
        .dt.days
    )

    df["days_since_last_txn"] = (
        df["days_since_last_txn"]
        .fillna(0)
    )

    return df


def calculate_monthly_spending(df):

    df["year_month"] = (
        df["date"]
        .dt.to_period("M")
        .astype(str)
    )

    expense = df[
        df["transaction_type"] == "Expense"
    ]

    monthly_total = (
        expense
        .groupby(["user_id", "year_month"])["amount"]
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


def calculate_category_spending(df):

    expense = df[
        df["transaction_type"] == "Expense"
    ]

    category_monthly = (
        expense
        .groupby(
            ["user_id", "year_month", "category"]
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


def calculate_transaction_frequency(df):

    df["user_txn_frequency"] = (
        df.groupby("user_id")["transaction_id"]
        .transform("count")
    )

    return df


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


def main():

    df = pd.read_csv(
        "processed/standardised_dataset.csv"
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = extract_date_parts(df)

    df = calculate_days_between_transactions(df)

    df = calculate_monthly_spending(df)

    df = calculate_category_spending(df)

    df = calculate_transaction_frequency(df)

    df = calculate_rolling_average(df)

    df.to_csv(
        "processed/feature_engineered_dataset.csv",
        index=False
    )

    print("Feature engineered dataset saved.")


if __name__ == "__main__":
    main()