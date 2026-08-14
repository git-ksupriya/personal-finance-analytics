import data_cleaning, data_standardisation, Feature_Extraction, sequence_preparation

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

    # 3. Standardisation
    df["category"] = df["category"].fillna("Others")
    df = data_standardisation.standardise_category(df)

    df = data_standardisation.standardise_locations(df)

    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    df = data_standardisation.standardise_payment_modes(df)

    df = data_standardisation.standardise_dates(df)
    df = data_standardisation.standardise_transaction_type(df)

    # 4. Feature extraction
    df = Feature_Extraction.extract_date_parts(df)
    df = Feature_Extraction.calculate_days_between_transactions(df)
    df = Feature_Extraction.calculate_monthly_spending(df)
    df = Feature_Extraction.calculate_category_spending(df)
    df = Feature_Extraction.calculate_transaction_frequency(df)
    df = Feature_Extraction.calculate_rolling_average(df)

    # 5. Sequence preparation for LSTM
    prep = sequence_preparation.SequencePreparation(window_size=3)

    df = prep.sort_transactions(df)

    grouped = prep.group_transactions(df)

    X, y = prep.create_sequences(grouped)

    X_train, X_test, y_train, y_test = prep.chronological_split(X, y)

    X_train = prep.reshape_lstm(X_train)
    X_test = prep.reshape_lstm(X_test)

    # 6. Autoencoder data
    auto_data = prep.prepare_autoencoder(df)


main()