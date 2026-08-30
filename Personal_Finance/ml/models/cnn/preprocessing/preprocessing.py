from pathlib import Path
import sys
import numpy as np

import data_cleaning
import data_standardisation
import feature_extraction
from sequence_preparation import CNNSequencePreparation


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Current folder:
# Personal_Finance/ml/models/cnn/preprocessing/

CURRENT_DIR = Path(__file__).resolve().parent

# Project root:
# Personal_Finance/
PROJECT_ROOT = CURRENT_DIR.parents[4]

# Raw dataset location
RAW_DATA_PATH = (
    PROJECT_ROOT
    / "preprocessing"
    / "raw"
    / "budgetwise_finance_dataset.csv"
)

# CNN processed-data folder
PROCESSED_DIR = (
    CURRENT_DIR.parent
    / "processed"
)

# Create processed directory if it doesn't exist
PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("CNN PREPROCESSING PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Check raw dataset
    # --------------------------------------------------------

    print("\nChecking dataset path...")

    print("Dataset:")
    print(RAW_DATA_PATH)

    if not RAW_DATA_PATH.exists():

        print("\nERROR: Dataset not found!")
        print("Expected location:")
        print(RAW_DATA_PATH)

        print(
            "\nMake sure budgetwise_finance_dataset.csv "
            "exists inside:"
        )

        print(
            PROJECT_ROOT
            / "preprocessing"
            / "raw"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # 1. Load raw dataset
    # --------------------------------------------------------

    df = data_cleaning.load_dataset(
        str(RAW_DATA_PATH)
    )

    print("\n1. Dataset loaded")
    print("Shape:", df.shape)

    # --------------------------------------------------------
    # 2. Cleaning
    # --------------------------------------------------------

    df = data_cleaning.handle_missing_values(df)

    df = data_cleaning.remove_duplicates(df)

    df = data_cleaning.clean_amount_column(df)

    df = data_cleaning.validate_amounts(df)

    df = data_cleaning.validate_dates(df)

    df = data_cleaning.remove_outliers(df)

    print("\n2. Cleaning complete")
    print("Shape:", df.shape)

    # --------------------------------------------------------
    # 3. Standardisation
    # --------------------------------------------------------

    df = data_standardisation.standardise_category(df)

    df = data_standardisation.standardise_locations(df)

    df = data_standardisation.standardise_payment_modes(df)

    df = data_standardisation.standardise_dates(df)

    df = data_standardisation.standardise_transaction_type(df)

    print("\n3. Standardisation complete")
    print("Shape:", df.shape)

    # --------------------------------------------------------
    # 4. Feature Engineering
    # --------------------------------------------------------

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

    print("\n4. Feature engineering complete")
    print("Shape:", df.shape)

    # --------------------------------------------------------
    # 5. CNN Sequence Preparation
    # --------------------------------------------------------

    prep = CNNSequencePreparation(
        window_size=3
    )

    df = prep.sort_transactions(df)

    X, y = prep.create_sequences(df)

    print("\n5. Sequences created")

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # --------------------------------------------------------
    # 6. Chronological Train-Test Split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = prep.chronological_split(
        X,
        y
    )

    print("\n6. Train-test split complete")

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    # --------------------------------------------------------
    # 7. CNN Reshape
    # --------------------------------------------------------

    X_train = prep.reshape_for_cnn(
        X_train
    )

    X_test = prep.reshape_for_cnn(
        X_test
    )

    print("\n7. CNN reshape complete")

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    # --------------------------------------------------------
    # 8. Save processed data
    # --------------------------------------------------------

    print("\n8. Saving CNN preprocessing outputs...")

    np.save(
        PROCESSED_DIR / "X_train_cnn.npy",
        X_train
    )

    np.save(
        PROCESSED_DIR / "X_test_cnn.npy",
        X_test
    )

    np.save(
        PROCESSED_DIR / "y_train_cnn.npy",
        y_train
    )

    np.save(
        PROCESSED_DIR / "y_test_cnn.npy",
        y_test
    )

    df.to_csv(
        PROCESSED_DIR
        / "feature_engineered_dataset_cnn.csv",
        index=False
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CNN PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nOutput directory:")
    print(PROCESSED_DIR)

    print("\nGenerated files:")

    print("✓ X_train_cnn.npy")
    print("✓ X_test_cnn.npy")
    print("✓ y_train_cnn.npy")
    print("✓ y_test_cnn.npy")
    print("✓ feature_engineered_dataset_cnn.csv")

    print("\nFinal shapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()