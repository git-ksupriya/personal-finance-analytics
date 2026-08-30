import os
import pandas as pd
import numpy as np


def load_dataset(file_path):
    """Load the standardised transaction dataset."""
    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def prepare_expense_data(df):
    """
    Keep only valid expense transactions.

    Expected columns:
        user_id
        date
        category
        amount
        transaction_type
    """

    df = df.copy()

    # Keep only expenses
    df = df[df["transaction_type"].str.lower() == "expense"]

    # Remove invalid rows
    df = df.dropna(subset=["user_id", "date", "category", "amount"])

    # Ensure amount is numeric
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])

    # Spending should be non-negative
    df["amount"] = df["amount"].abs()

    return df


def aggregate_daily_category_spending(df):
    """
    Aggregate transactions into:

        user_id × category × date

    Example:

        U001 | Food | 2025-01-01 | 300
        U001 | Food | 2025-01-02 | 0
        U001 | Food | 2025-01-03 | 150

    Multiple transactions on the same day/category are summed.
    """

    daily = (
        df.groupby(
            ["user_id", "category", "date"],
            as_index=False
        )["amount"]
        .sum()
        .rename(columns={"amount": "daily_spending"})
    )

    return daily


def create_complete_daily_series(daily_df):
    """
    Create a continuous daily time series for every
    user-category combination.

    Days with no transaction are assigned 0 spending.
    """

    results = []

    for (user_id, category), group in daily_df.groupby(
        ["user_id", "category"]
    ):

        group = group.sort_values("date")

        start_date = group["date"].min()
        end_date = group["date"].max()

        full_dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq="D"
        )

        complete = pd.DataFrame({
            "date": full_dates
        })

        complete["user_id"] = user_id
        complete["category"] = category

        complete = complete.merge(
            group,
            on=["user_id", "category", "date"],
            how="left"
        )

        complete["daily_spending"] = (
            complete["daily_spending"]
            .fillna(0.0)
        )

        results.append(complete)

    if not results:
        return pd.DataFrame(
            columns=[
                "user_id",
                "category",
                "date",
                "daily_spending"
            ]
        )

    return pd.concat(results, ignore_index=True)


def add_calendar_features(df):
    """
    Optional calendar features.

    These are useful later if you want the model to know
    weekday/month/seasonality.
    """

    df = df.copy()

    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month

    # Weekend indicator
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    return df


def create_category_dataset(input_path, output_path):
    """
    Complete feature extraction pipeline.
    """

    df = load_dataset(input_path)

    df = prepare_expense_data(df)

    daily = aggregate_daily_category_spending(df)

    daily = create_complete_daily_series(daily)

    daily = add_calendar_features(daily)

    daily = daily.sort_values(
        ["user_id", "category", "date"]
    ).reset_index(drop=True)

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    daily.to_csv(
        output_path,
        index=False
    )

    print(
        f"Category-wise daily dataset saved to: {output_path}"
    )

    print(f"Rows: {len(daily)}")
    print(f"Users: {daily['user_id'].nunique()}")
    print(f"Categories: {daily['category'].nunique()}")

    return daily


if __name__ == "__main__":

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    # Adjust this depending on your actual project structure
    input_path = os.path.abspath(
        os.path.join(
            base_dir,
            "..",
            "..",
            "preprocessing",
            "processed",
            "standardised_dataset.csv"
        )
    )

    output_path = os.path.join(
        base_dir,
        "processed",
        "daily_category_spending.csv"
    )

    create_category_dataset(
        input_path,
        output_path
    )