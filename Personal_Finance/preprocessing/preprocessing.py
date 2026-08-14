import data_cleaning, data_standardisation, Feature_Extraction, sequence_preparation
import os, numpy as np

def main():
    # 1. Load raw dataset
    file_path = r".\\raw\\budgetwise_finance_dataset.csv"
    
    df = data_cleaning.load_dataset(file_path)

    # 2. Cleaning
    df = data_cleaning.handle_missing_values(df)
    df = data_cleaning.remove_duplicates(df)

    data_cleaning.validate_transaction_ids(df)
    data_cleaning.validate_user_ids(df)

    df = data_cleaning.clean_amount_column(df)
    df = data_cleaning.validate_amounts(df)
    df = data_cleaning.validate_dates(df)
    df = data_cleaning.remove_outliers(df)
    print("1. Cleaning complete")

    # 3. Standardisation
    df["category"] = df["category"].fillna("Others")
    df = data_standardisation.standardise_category(df)

    df = data_standardisation.standardise_locations(df)

    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    df = data_standardisation.standardise_payment_modes(df)

    df = data_standardisation.standardise_dates(df)
    df = data_standardisation.standardise_transaction_type(df)
    print("2. Standardisation complete:", df.shape)

    # 4. Feature extraction
    df = Feature_Extraction.extract_date_parts(df)
    df = Feature_Extraction.calculate_days_between_transactions(df)
    df = Feature_Extraction.calculate_monthly_spending(df)
    df = Feature_Extraction.calculate_category_spending(df)
    df = Feature_Extraction.calculate_transaction_frequency(df)
    df = Feature_Extraction.calculate_rolling_average(df)
    print("3. Feature extraction complete:", df.shape)

    # 5. Sequence preparation for LSTM
    prep = sequence_preparation.SequencePreparation(window_size=3)

    df = prep.sort_transactions(df)

    grouped = prep.group_transactions(df)

    X, y = prep.create_sequences(grouped)

    print("4. Sequences created")
    print("   X:", type(X), getattr(X, "shape", None))
    print("   y:", type(y), getattr(y, "shape", None))

    X_train, X_test, y_train, y_test = prep.chronological_split(X, y)

    print("5. Train/test split complete")
    print("   X_train:", getattr(X_train, "shape", None))
    print("   X_test :", getattr(X_test, "shape", None))
    print("   y_train:", getattr(y_train, "shape", None))
    print("   y_test :", getattr(y_test, "shape", None))

    X_train = prep.reshape_lstm(X_train)
    X_test = prep.reshape_lstm(X_test)

    print("6. LSTM reshape complete")
    print("   X_train:", X_train.shape)
    print("   X_test :", X_test.shape)

    # 6. Autoencoder data
    auto_data = prep.prepare_autoencoder(df)
    print("7. Autoencoder preparation complete")
    print("   type :", type(auto_data))
    print("   shape:", getattr(auto_data, "shape", None))

    # 7. Save outputs
    processed_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "processed")

    os.makedirs(processed_dir, exist_ok=True)

    np.save(os.path.join(processed_dir, "X_train.npy"), X_train)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)

    df.to_csv(os.path.join(processed_dir, "feature_engineered_dataset.csv"), index=False)

    if isinstance(auto_data, np.ndarray):
        np.save(os.path.join(processed_dir, "autoencoder_data.npy"), auto_data)
    else:
        auto_data.to_csv(os.path.join(processed_dir, "autoencoder_data.csv"), index=False)

    print("8. All preprocessing outputs saved.")

main()