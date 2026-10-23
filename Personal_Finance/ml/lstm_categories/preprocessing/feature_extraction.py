import os
import sys
import pandas as pd
import numpy as np


def load_dataset(file_path):
    """Load the standardised transaction dataset."""

    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    return df


def prepare_expense_data(df):
    """Keep valid expense transactions only."""

    df = df.copy()

    df["transaction_type"] = (
        df["transaction_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[
        df["transaction_type"] == "expense"
    ]

    df = df.dropna(
        subset=[
            "user_id",
            "date",
            "category",
            "amount"
        ]
    )

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["amount"]
    )

    # Spending should be represented as positive magnitude.
    df["amount"] = df["amount"].abs()

    return df


def aggregate_daily_category_spending(df):
    """
    Aggregate multiple transactions on the same
    user/category/date into one daily spending value.
    """

    daily = (
        df.groupby(
            [
                "user_id",
                "category",
                "date"
            ],
            as_index=False
        )["amount"]
        .sum()
        .rename(
            columns={
                "amount": "daily_spending"
            }
        )
    )

    return daily


def create_complete_daily_series(daily_df):
    """
    Create a continuous calendar-day series for every
    user-category pair.

    A day with no spending is explicitly represented as 0.
    """

    results = []

    for (
        user_id,
        category
    ), group in daily_df.groupby(
        ["user_id", "category"]
    ):

        group = group.sort_values(
            "date"
        )

        full_dates = pd.date_range(
            start=group["date"].min(),
            end=group["date"].max(),
            freq="D"
        )

        complete = pd.DataFrame(
            {"date": full_dates}
        )

        complete["user_id"] = user_id
        complete["category"] = category

        complete = complete.merge(
            group,
            on=[
                "user_id",
                "category",
                "date"
            ],
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

    return pd.concat(
        results,
        ignore_index=True
    )


def add_calendar_features(df):
    """
    Add calendar features while preserving the full
    chronological daily series.

    Weekday is encoded cyclically so that Sunday and
    Monday are treated as adjacent.
    """

    df = df.copy()

    day_of_week = df["date"].dt.dayofweek

    df["weekday_sin"] = np.sin(
        2 * np.pi * day_of_week / 7
    )

    df["weekday_cos"] = np.cos(
        2 * np.pi * day_of_week / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["date"].dt.month / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["date"].dt.month / 12
    )

    df["is_weekend"] = (
        day_of_week >= 5
    ).astype(int)

    return df


def aggregate_monthly_category_spending(df):
    return (
        df.assign(month=df["date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["user_id", "category", "month"], as_index=False)["amount"]
        .sum()
        .rename(columns={"month": "date", "amount": "monthly_spending"})
    )


def create_complete_monthly_series(monthly_df, user_coverage=None):
    results = []
    user_bounds = user_coverage or (
        monthly_df.groupby("user_id")["date"]
        .agg(["min", "max"])
        .to_dict("index")
    )
    for (user_id, category), group in monthly_df.groupby(["user_id", "category"]):
        bounds = user_bounds[user_id]
        first_date = pd.Timestamp(bounds["min"])
        last_date = pd.Timestamp(bounds["max"])
        first_month = first_date.to_period("M")
        last_month = last_date.to_period("M")
        if first_date.date() > first_month.start_time.date():
            first_month += 1
        if last_date.date() < last_month.end_time.date():
            last_month -= 1
        if first_month > last_month:
            continue
        months = pd.period_range(
            first_month,
            last_month,
            freq="M",
        ).to_timestamp()
        complete = (
            group.set_index("date")["monthly_spending"]
            .reindex(months, fill_value=0.0)
            .rename_axis("date")
            .reset_index()
        )
        complete["user_id"] = user_id
        complete["category"] = category
        results.append(complete)
    if not results:
        return pd.DataFrame(
            columns=["user_id", "category", "date", "monthly_spending"]
        )
    return pd.concat(results, ignore_index=True)


def add_monthly_calendar_features(df):
    df = df.copy()
    month = df["date"].dt.month
    quarter = df["date"].dt.quarter
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["quarter_sin"] = np.sin(2 * np.pi * quarter / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * quarter / 4)
    df["days_in_month"] = df["date"].dt.days_in_month / 31.0
    return df


def create_category_dataset(
    input_path,
    output_path
):

    df = load_dataset(
        input_path
    )

    df = prepare_expense_data(
        df
    )

    daily = aggregate_daily_category_spending(
        df
    )

    daily = create_complete_daily_series(
        daily
    )

    daily = add_calendar_features(
        daily
    )

    daily = daily.sort_values(
        [
            "user_id",
            "category",
            "date"
        ]
    ).reset_index(
        drop=True
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    daily.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Rows: {len(daily)}"
    )

    print(
        f"Users: "
        f"{daily['user_id'].nunique()}"
    )

    print(
        f"Categories: "
        f"{daily['category'].nunique()}"
    )

    return daily


def create_monthly_category_dataset(input_path, output_path):
    df = load_dataset(input_path)
    coverage = (
        df.dropna(subset=["user_id", "date"])
        .groupby("user_id")["date"]
        .agg(["min", "max"])
        .to_dict("index")
    )
    expenses = prepare_expense_data(df)
    monthly = aggregate_monthly_category_spending(expenses)
    monthly = create_complete_monthly_series(monthly, coverage)
    monthly = add_monthly_calendar_features(monthly)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    monthly.to_csv(output_path, index=False)
    return monthly


if __name__ == "__main__":

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    input_path = os.path.abspath(
        os.path.join(
            base_dir,
            "..",
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

    if sys.argv[1:] == ["--monthly-only"]:
        create_monthly_category_dataset(
            input_path,
            os.path.join(
                base_dir,
                "processed",
                "monthly_category_spending.csv",
            ),
        )
    else:
        create_category_dataset(
            input_path,
            output_path
        )