import os
import sys
import numpy as np

import data_cleaning
import data_standardisation
import feature_extraction
from sequence_preparation import CNNSequencePreparation


def main():

    print("=" * 60)
    print("CNN PREPROCESSING PIPELINE")
    print("=" * 60)

    # ---------------------------------------------
    # 1. Load raw dataset
    # ---------------------------------------------

    raw_path = (
       r"C:\Users\Shivananda\personal-finance-analytics\Personal_Finance\preprocessing\raw\budgetwise_finance_dataset.csv"
    )

    df = data_cleaning.load_dataset(
        raw_path
    )

    print("\n1. Dataset loaded")

    # ---------------------------------------------
    # 2. Cleaning
    # ---------------------------------------------

    df = data_cleaning.handle_missing_values(df)

    df = data_cleaning.remove_duplicates(df)

    df = data_cleaning.clean_amount_column(df)

    df = data_cleaning.validate_amounts(df)

    df = data_cleaning.validate_dates(df)

    df = data_cleaning.remove_outliers(df)

    print("2. Cleaning complete")

    # ---------------------------------------------
    # 3. Standardisation
    # ---------------------------------------------

    df = data_standardisation.standardise_category(df)

    df = data_standardisation.standardise_locations(df)

    df = data_standardisation.standardise_payment_modes(df)

    df = data_standardisation.standardise_dates(df)

    df = data_standardisation.standardise_transaction_type(df)

    print("3. Standardisation complete")

    # ---------------------------------------------
    # 4. Feature engineering
    # ---------------------------------------------

    df = feature_extraction.extract_date_parts(df)

    df = (
        feature_extraction
        .calculate_days_between_transactions(df)
    )

    df = (
        feature_extraction
        .calculate_monthly_spending(df)
    )

    df = (
        feature_extraction
        .calculate_category_spending(df)
    )

    df = (
        feature_extraction
        .calculate_transaction_frequency(df)
    )

    df = (
        feature_extraction
        .calculate_rolling_average(df)
    )

    print("4. Feature engineering complete")

    # ---------------------------------------------
    # 5. CNN sequence preparation
    # ---------------------------------------------

    prep = CNNSequencePreparation(
        window_size=3
    )

    df = prep.sort_transactions(df)

    X, y = prep.create_sequences(df)

    print("\n5. Sequences created")

    print("X:", X.shape)

    print("y:", y.shape)

    # ---------------------------------------------
    # 6. Train-test split
    # ---------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = prep.chronological_split(
        X,
        y
    )

    # ---------------------------------------------
    # 7. CNN reshape
    # ---------------------------------------------

    X_train = prep.reshape_for_cnn(
        X_train
    )

    X_test = prep.reshape_for_cnn(
        X_test
    )

    print("\n6. CNN reshape complete")

    print("X_train:", X_train.shape)

    print("X_test :", X_test.shape)

    print("y_train:", y_train.shape)

    print("y_test :", y_test.shape)

    # ---------------------------------------------
    # 8. Save data
    # ---------------------------------------------

    processed_dir = "processed"

    os.makedirs(
        processed_dir,
        exist_ok=True
    )

    np.save(
        os.path.join(
            processed_dir,
            "X_train_cnn.npy"
        ),
        X_train
    )

    np.save(
        os.path.join(
            processed_dir,
            "X_test_cnn.npy"
        ),
        X_test
    )

    np.save(
        os.path.join(
            processed_dir,
            "y_train_cnn.npy"
        ),
        y_train
    )

    np.save(
        os.path.join(
            processed_dir,
            "y_test_cnn.npy"
        ),
        y_test
    )

    df.to_csv(
        os.path.join(
            processed_dir,
            "feature_engineered_dataset_cnn.csv"
        ),
        index=False
    )

    print("\nCNN preprocessing outputs saved.")

    print("=" * 60)


if __name__ == "__main__":
    main()