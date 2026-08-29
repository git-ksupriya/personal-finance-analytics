import os
import pandas as pd

import data_cleaning
import data_standardisation
import feature_extraction


def main():

    print("=" * 60)
    print("XGBOOST PREPROCESSING PIPELINE")
    print("=" * 60)

    # -------------------------------------------------
    # Find project root
    # -------------------------------------------------

    BASE_DIR = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../../.."
        )
    )

    raw_path = os.path.join(
        BASE_DIR,
        "preprocessing",
        "raw",
        "budgetwise_finance_dataset.csv"
    )

    output_dir = os.path.join(
        os.path.dirname(__file__),
        "processed"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # -------------------------------------------------
    # 1. Load
    # -------------------------------------------------

    df = data_cleaning.load_dataset(
        raw_path
    )

    original_rows = len(df)

    print("\n1. Dataset loaded")

    # -------------------------------------------------
    # 2. Cleaning
    # -------------------------------------------------

    df = data_cleaning.handle_missing_values(df)

    df = data_cleaning.remove_duplicates(df)

    df = data_cleaning.clean_amount_column(df)

    df = data_cleaning.validate_amounts(df)

    df = data_cleaning.validate_dates(df)

    df = data_cleaning.remove_outliers(df)

    print(
        f"2. Cleaning complete: {df.shape}"
    )

    # -------------------------------------------------
    # 3. Standardisation
    # -------------------------------------------------

    df = data_standardisation.standardise_category(df)

    df = data_standardisation.standardise_locations(df)

    df = data_standardisation.standardise_payment_modes(df)

    df = data_standardisation.standardise_dates(df)

    df = data_standardisation.standardise_transaction_type(df)

    print(
        f"3. Standardisation complete: {df.shape}"
    )

    # -------------------------------------------------
    # 4. Feature Engineering
    # -------------------------------------------------

    df = feature_extraction.extract_date_features(df)

    df = feature_extraction.calculate_days_since_last_transaction(df)

    df = feature_extraction.calculate_monthly_spending(df)

    df = feature_extraction.calculate_category_spending(df)

    df = feature_extraction.calculate_transaction_frequency(df)

    df = feature_extraction.calculate_rolling_average(df)

    df = feature_extraction.calculate_user_average(df)

    df = feature_extraction.calculate_user_max(df)

    df = feature_extraction.calculate_spending_deviation(df)

    print(
        f"4. Feature engineering complete: {df.shape}"
    )

    # -------------------------------------------------
    # 5. Remove columns that XGBoost cannot directly use
    # -------------------------------------------------

    df = df.drop(
        columns=[
            "date",
            "notes"
        ],
        errors="ignore"
    )

    # -------------------------------------------------
    # 6. Convert categorical columns
    # -------------------------------------------------

    categorical_columns = [
        "transaction_type",
        "category",
        "payment_mode",
        "location"
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=False
    )

    # Convert boolean columns to integers
    bool_columns = df.select_dtypes(
        include="bool"
    ).columns

    df[bool_columns] = df[
        bool_columns
    ].astype(int)

    # -------------------------------------------------
    # 7. Handle remaining missing values
    # -------------------------------------------------

    df = df.fillna(0)

    # -------------------------------------------------
    # 8. Save
    # -------------------------------------------------

    output_path = os.path.join(
        output_dir,
        "xgboost_features.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("XGBOOST PREPROCESSING COMPLETE")
    print("=" * 60)

    print(f"Original rows : {original_rows}")
    print(f"Final rows    : {len(df)}")
    print(f"Final columns : {len(df.columns)}")

    print(
        f"\nSaved to:\n{output_path}"
    )

    print("\nFeature columns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()