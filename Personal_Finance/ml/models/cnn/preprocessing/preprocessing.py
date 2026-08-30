import os
from pathlib import Path

import numpy as np

import data_cleaning
import data_standardisation
import feature_extraction

from sequence_preparation import CNNSequencePreparation


# ============================================================
# FIND PROJECT DIRECTORIES
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent


def find_personal_finance_directory():
    """
    Find the Personal_Finance directory by searching
    through the parent directories of this file.
    """

    for parent in [CURRENT_DIR] + list(CURRENT_DIR.parents):

        if parent.name == "Personal_Finance":
            return parent

    raise FileNotFoundError(
        "Could not find the 'Personal_Finance' directory."
    )


PERSONAL_FINANCE_DIR = find_personal_finance_directory()

# Repository root
PROJECT_ROOT = PERSONAL_FINANCE_DIR.parent


# ============================================================
# DATASET PATHS
# ============================================================

RAW_DATASET = (
    PERSONAL_FINANCE_DIR
    / "preprocessing"
    / "raw"
    / "budgetwise_finance_dataset.csv"
)


# Processed directory
PROCESSED_DIR = PROJECT_ROOT / "processed"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CNN PREPROCESSING PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Display paths
    # --------------------------------------------------------

    print("\nCurrent file:")
    print(Path(__file__).resolve())

    print("\nPersonal_Finance directory:")
    print(PERSONAL_FINANCE_DIR)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nChecking dataset path...")

    print("Dataset:")
    print(RAW_DATASET)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not RAW_DATASET.exists():

        print("\nERROR: Dataset not found!")

        print("\nExpected location:")
        print(RAW_DATASET)

        print("\nPlease make sure the dataset exists at:")

        print(
            PERSONAL_FINANCE_DIR
            / "preprocessing"
            / "raw"
        )

        return

    print("\nDataset found successfully!")

    # ========================================================
    # 1. LOAD DATASET
    # ========================================================

    df = data_cleaning.load_dataset(
        str(RAW_DATASET)
    )

    print("\n1. Dataset loaded")
    print("Shape:", df.shape)

    # ========================================================
    # 2. DATA CLEANING
    # ========================================================

    df = data_cleaning.handle_missing_values(df)

    df = data_cleaning.remove_duplicates(df)

    df = data_cleaning.clean_amount_column(df)

    df = data_cleaning.validate_amounts(df)

    df = data_cleaning.validate_dates(df)

    df = data_cleaning.remove_outliers(df)

    print("\n2. Cleaning complete")
    print("Shape:", df.shape)

    # ========================================================
    # 3. STANDARDISATION
    # ========================================================

    df = data_standardisation.standardise_category(df)

    df = data_standardisation.standardise_locations(df)

    df = data_standardisation.standardise_payment_modes(df)

    df = data_standardisation.standardise_dates(df)

    df = data_standardisation.standardise_transaction_type(df)

    print("\n3. Standardisation complete")
    print("Shape:", df.shape)

    # ========================================================
    # 4. FEATURE ENGINEERING
    # ========================================================

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

    # ========================================================
    # 5. CNN SEQUENCE PREPARATION
    # ========================================================

    prep = CNNSequencePreparation(
        window_size=3
    )

    df = prep.sort_transactions(df)

    X, y = prep.create_sequences(df)

    print("\n5. CNN sequences created")

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # ========================================================
    # 6. TRAIN TEST SPLIT
    # ========================================================

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

    # ========================================================
    # 7. CNN RESHAPE
    # ========================================================

    X_train = prep.reshape_for_cnn(
        X_train
    )

    X_test = prep.reshape_for_cnn(
        X_test
    )

    print("\n7. CNN reshape complete")

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    # ========================================================
    # 8. CREATE PROCESSED DIRECTORY
    # ========================================================

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nProcessed directory:")
    print(PROCESSED_DIR)

    # ========================================================
    # 9. SAVE TRAINING DATA
    # ========================================================

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

    # ========================================================
    # 10. SAVE FEATURE ENGINEERED DATASET
    # ========================================================

    feature_dataset_path = (
        PROCESSED_DIR
        / "feature_engineered_dataset_cnn.csv"
    )

    df.to_csv(
        feature_dataset_path,
        index=False
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 60)
    print("CNN PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated files:")

    print(
        PROCESSED_DIR / "X_train_cnn.npy"
    )

    print(
        PROCESSED_DIR / "X_test_cnn.npy"
    )

    print(
        PROCESSED_DIR / "y_train_cnn.npy"
    )

    print(
        PROCESSED_DIR / "y_test_cnn.npy"
    )

    print(
        PROCESSED_DIR
        / "feature_engineered_dataset_cnn.csv"
    )

    print("\nFinal shapes:")

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()