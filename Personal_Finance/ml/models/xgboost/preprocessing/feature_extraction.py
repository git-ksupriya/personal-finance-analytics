import pandas as pd


def extract_date_features(df):

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday

    return df


def create_lag_features(df):

    df = df.sort_values(
        ["user_id", "date"]
    ).copy()

    # Previous transaction amounts
    df["lag_1"] = (
        df.groupby("user_id")["amount"]
        .shift(1)
    )

    df["lag_2"] = (
        df.groupby("user_id")["amount"]
        .shift(2)
    )

    df["lag_3"] = (
        df.groupby("user_id")["amount"]
        .shift(3)
    )

    return df


def create_rolling_features(df):

    df = df.sort_values(
        ["user_id", "date"]
    ).copy()

    df["rolling_mean_3"] = (
        df.groupby("user_id")["amount"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .mean()
        )
    )

    return df


def create_transaction_frequency(df):

    df["user_transaction_count"] = (
        df.groupby("user_id")
        .cumcount()
    )

    return df


def create_target(df):

    # Next transaction amount becomes the prediction target
    df["target"] = (
        df.groupby("user_id")["amount"]
        .shift(-1)
    )

    return df


def prepare_xgboost_features(df):

    df = extract_date_features(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    df = create_transaction_frequency(df)

    df = create_target(df)

    return df