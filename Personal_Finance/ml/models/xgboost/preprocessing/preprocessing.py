import os
import sys
import pandas as pd

# Allow importing files from this folder
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

import data_cleaning
import data_standardisation
import feature_extraction


def main():

    print("=" * 60)
    print("XGBOOST PREPROCESSING PIPELINE")
    print("=" * 60)

    # --------------------------------------------------
    # Find project root
    # --------------------------------------------------

    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    project_root = os.path.abspath(
        os.path.join(current_dir, "../../../..")
    )

    # --------------------------------------------------
    # Dataset path
    # --------------------------------------------------

    raw_path = os.path.join(
        project_root,
        "preprocessing",
        "raw",
        "budgetwise_finance_dataset.csv"
    )

    print("\nDataset:")
    print(raw_path)

    # --------------------------------------------------
    # 1. Load
    # --------------------------------------------------

    df = data_cleaning.load_dataset(
        raw_path
    )

    # --------------------------------------------------
    # 2. Cleaning
    # --------------------------------------------------

    df = data_cleaning.handle_missing_values(df)

    df = data_cleaning.remove_duplicates(df)

    df = data_cleaning.clean_amount_column(df)

    df = data_cleaning.validate_amounts(df)

    df = data_cleaning.validate_dates(df)

    df = data_cleaning.remove_outliers(df)

    print("\n1. Cleaning complete")
    print("Shape:", df.shape)

    # --------------------------------------------------
    # 3. Standardisation
    # --------------------------------------------------

    df = data_standardisation.standardise_category(df)

    df = data_standardisation.standardise_locations(df)

    df = data_standardisation.standardise_payment_modes(df)

    df = data_standardisation.standardise_dates(df)

    df = data_standardisation.standardise_transaction_type(df)

    print("\n2. Standardisation complete")
    print("Shape:", df.shape)

    # --------------------------------------------------
    # 4. Feature Engineering
    # --------------------------------------------------

    df = feature_extraction.prepare_xgboost_features(
        df
    )

    # Remove rows where lag/target cannot be calculated
    df = df.dropna(
        subset=[
            "lag_1",
            "lag_2",
            "lag_3",
            "rolling_mean_3",
            "target"
        ]
    )

    print("\n3. Feature extraction complete")
    print("Shape:", df.shape)

    # --------------------------------------------------
    # Select XGBoost features
    # --------------------------------------------------

    feature_columns = [

        "amount",

        "lag_1",
        "lag_2",
        "lag_3",

        "rolling_mean_3",

        "year",
        "month",
        "day",
        "weekday",

        "user_transaction_count"
    ]

    X = df[feature_columns]

    y = df["target"]

    # --------------------------------------------------
    # Chronological split
    # --------------------------------------------------

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # --------------------------------------------------
    # Save processed data
    # --------------------------------------------------

    processed_dir = os.path.join(
        current_dir,
        "processed"
    )

    os.makedirs(
        processed_dir,
        exist_ok=True
    )

    X_train.to_csv(
        os.path.join(
            processed_dir,
            "X_train.csv"
        ),
        index=False
    )

    X_test.to_csv(
        os.path.join(
            processed_dir,
            "X_test.csv"
        ),
        index=False
    )

    y_train.to_csv(
        os.path.join(
            processed_dir,
            "y_train.csv"
        ),
        index=False
    )

    y_test.to_csv(
        os.path.join(
            processed_dir,
            "y_test.csv"
        ),
        index=False
    )

    df.to_csv(
        os.path.join(
            processed_dir,
            "xgboost_dataset.csv"
        ),
        index=False
    )

    print("\n4. Train/Test split complete")

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    print("\nXGBoost preprocessing completed successfully.")


if __name__ == "__main__":
    main()