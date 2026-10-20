import os
import re
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

WINDOW_SIZE = 30
FUTURE_DAYS = 7

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_FILE = os.path.join(
    BASE_DIR, "raw", "budgetwise_finance_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "processed", "lstm_daily"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# CLEAN AMOUNT
# --------------------------------------------------

def clean_amount(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    value = re.sub(r"Rs\.?|₹|\$|,", "", value, flags=re.I)

    value = re.sub(r"[^0-9.\-]", "", value)

    if not value or value in {".", "-", "-."}:
        return np.nan

    try:
        return float(value)
    except ValueError:
        return np.nan


# --------------------------------------------------
# LOAD AND CLEAN DATA
# --------------------------------------------------

def load_data():

    df = pd.read_csv(RAW_FILE)

    df["amount"] = df["amount"].apply(clean_amount)

    # The dataset contains multiple date formats.
    df["date"] = pd.to_datetime(
        df["date"],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    df = df.dropna(
        subset=["user_id", "date", "amount", "transaction_type"]
    ).copy()

    df["transaction_type"] = (
        df["transaction_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Exclude income: we want expenditure, not cash flow.
    df = df[df["transaction_type"] == "expense"].copy()

    # Negative and zero amounts are excluded for this initial model.
    df = df[df["amount"] > 0].copy()

    # Keep the date at calendar-day resolution.
    df["date"] = df["date"].dt.normalize()

    print("Expense records:", len(df))
    print("Users:", df["user_id"].nunique())

    return df


# --------------------------------------------------
# AGGREGATE TRANSACTIONS INTO DAILY EXPENDITURE
# --------------------------------------------------

def create_daily_data(df):

    daily = (
        df.groupby(["user_id", "date"], as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "daily_expenditure"})
    )

    print("Unique user-date records:", len(daily))

    return daily


# --------------------------------------------------
# CREATE CHRONOLOGICAL SEQUENCES
# --------------------------------------------------

def create_sequences(daily):

    all_X = []
    all_y = []

    # Track which user each sequence belongs to.
    all_users = []

    # Keep a record of target dates for evaluation.
    all_target_dates = []

    for user_id, user_data in daily.groupby("user_id"):

        user_data = user_data.sort_values("date").copy()

        # Build a calendar-day series.
        # Missing dates inside the observed range are filled with zero.
        # This assumes the transaction dataset captures all spending.
        user_data = user_data.set_index("date")

        full_dates = pd.date_range(
            start=user_data.index.min(),
            end=user_data.index.max(),
            freq="D"
        )

        user_data = user_data.reindex(full_dates)

        user_data["daily_expenditure"] = (
            user_data["daily_expenditure"].fillna(0)
        )

        user_data.index.name = "date"

        values = user_data["daily_expenditure"].to_numpy(
            dtype=np.float32
        )

        dates = user_data.index.to_numpy()

        n = len(values)

        if n < WINDOW_SIZE + FUTURE_DAYS:
            continue

        # Split by target dates, not randomly.
        train_end = int(n * TRAIN_RATIO)

        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        for i in range(WINDOW_SIZE, n - FUTURE_DAYS + 1):

            target_start = i
            target_end = i + FUTURE_DAYS

            # Training targets must be entirely in training period.
            if target_end <= train_end:
                split = "train"

            # Validation targets must be entirely in validation period.
            elif (
                target_start >= train_end
                and target_end <= val_end
            ):
                split = "val"

            # Test targets must be entirely in test period.
            elif target_start >= val_end:
                split = "test"

            else:
                # Discard sequences crossing a split boundary.
                continue

            X_window = values[i - WINDOW_SIZE:i]
            y_window = values[i:i + FUTURE_DAYS]

            all_X.append(X_window)
            all_y.append(y_window)
            all_users.append(user_id)
            all_target_dates.append(dates[i:i + FUTURE_DAYS])

            # Save the split label for each sequence.
            # The list remains aligned with X and y.
            if "splits" not in locals():
                splits = []

            splits.append(split)

    if not all_X:
        raise ValueError(
            "No sequences were created. "
            "Check the date range and available expenditure records."
        )

    X = np.asarray(all_X, dtype=np.float32)
    y = np.asarray(all_y, dtype=np.float32)

    splits = np.asarray(splits)

    users = np.asarray(all_users)

    target_dates = np.asarray(all_target_dates)

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    print("\nSequence shapes:")
    print("X:", X.shape)
    print("y:", y.shape)

    print("\nSplit sizes:")
    print("Training:", train_mask.sum())
    print("Validation:", val_mask.sum())
    print("Testing:", test_mask.sum())

    return (
        X, y, users, target_dates,
        train_mask, val_mask, test_mask
    )


# --------------------------------------------------
# SCALE WITHOUT USING VALIDATION OR TEST DATA
# --------------------------------------------------

def scale_and_save(
    X, y, users, target_dates,
    train_mask, val_mask, test_mask
):

    # Fit scaler using training values only.
    scaler = MinMaxScaler()

    scaler.fit(y[train_mask].reshape(-1, 1))

    # Apply the same scaler to inputs and targets.
    X_scaled = scaler.transform(
        X.reshape(-1, 1)
    ).reshape(X.shape)

    y_scaled = scaler.transform(
        y.reshape(-1, 1)
    ).reshape(y.shape)

    # LSTM input shape: samples, timesteps, features.
    X_scaled = X_scaled[..., np.newaxis]

    # Save distinct files for the new LSTM pipeline.
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"),
            X_scaled[train_mask])

    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"),
            y_scaled[train_mask])

    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"),
            X_scaled[val_mask])

    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"),
            y_scaled[val_mask])

    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),
            X_scaled[test_mask])

    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),
            y_scaled[test_mask])

    # Preserve actual rupee values for evaluation.
    np.save(os.path.join(OUTPUT_DIR, "y_train_raw.npy"),
            y[train_mask])

    np.save(os.path.join(OUTPUT_DIR, "y_val_raw.npy"),
            y[val_mask])

    np.save(os.path.join(OUTPUT_DIR, "y_test_raw.npy"),
            y[test_mask])

    np.save(os.path.join(OUTPUT_DIR, "test_users.npy"),
            users[test_mask])

    np.save(os.path.join(OUTPUT_DIR, "test_target_dates.npy"),
            target_dates[test_mask])

    joblib.dump(
        scaler,
        os.path.join(OUTPUT_DIR, "daily_scaler.pkl")
    )

    print("\nSaved new LSTM files to:")
    print(OUTPUT_DIR)

    print("\nFinal shapes:")
    print("X_train:", X_scaled[train_mask].shape)
    print("y_train:", y_scaled[train_mask].shape)

    print("X_val:", X_scaled[val_mask].shape)
    print("y_val:", y_scaled[val_mask].shape)

    print("X_test:", X_scaled[test_mask].shape)
    print("y_test:", y_scaled[test_mask].shape)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df = load_data()

    daily = create_daily_data(df)

    (
        X, y, users, target_dates,
        train_mask, val_mask, test_mask
    ) = create_sequences(daily)

    scale_and_save(
        X, y, users, target_dates,
        train_mask, val_mask, test_mask
    )


if __name__ == "__main__":
    main()