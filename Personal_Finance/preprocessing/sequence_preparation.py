import pandas as pd
import numpy as np


class SequencePreparation:

    def __init__(self, window_size=5):
        self.window_size = window_size

    # -------------------------------------------------
    # Sort transactions by user and date
    # -------------------------------------------------
    def sort_transactions(self, df):

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        df = df.sort_values(by=["user_id", "date"])

        return df

    # -------------------------------------------------
    # Group transactions by user
    # -------------------------------------------------
    def group_transactions(self, df):

        return df.groupby("user_id")

    # -------------------------------------------------
    # Create sliding window sequences
    # -------------------------------------------------
    def create_sequences(self, grouped):

        X = []
        y = []

        for user_id, user_data in grouped:

            amounts = user_data["amount"].values

            if len(amounts) <= self.window_size:
                continue

            for i in range(len(amounts) - self.window_size):

                X.append(amounts[i:i + self.window_size])

                y.append(amounts[i + self.window_size])

        return np.array(X), np.array(y)

    # -------------------------------------------------
    # Chronological Train-Test Split
    # -------------------------------------------------
    def chronological_split(self, X, y, train_ratio=0.8):

        split_index = int(len(X) * train_ratio)

        X_train = X[:split_index]
        X_test = X[split_index:]

        y_train = y[:split_index]
        y_test = y[split_index:]

        return X_train, X_test, y_train, y_test

    # -------------------------------------------------
    # Reshape for LSTM
    # -------------------------------------------------
    def reshape_lstm(self, X):

        return X.reshape(X.shape[0], X.shape[1], 1)

    # -------------------------------------------------
    # Prepare Autoencoder Dataset
    # -------------------------------------------------
    def prepare_autoencoder(self, df):

        return df[["amount"]].values


# =====================================================
# Testing Module
# =====================================================

def main():

    data = {

        "user_id": [
            1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2
        ],

        "date": [

            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
            "2023-01-04",
            "2023-01-05",
            "2023-01-06",

            "2023-02-01",
            "2023-02-02",
            "2023-02-03",
            "2023-02-04",
            "2023-02-05",
            "2023-02-06"

        ],

        "amount": [
            100,
            150,
            180,
            250,
            300,
            350,
            500,
            450,
            600,
            700,
            650,
            800
        ]

    }

    df = pd.DataFrame(data)

    prep = SequencePreparation(window_size=3)

    print("\nOriginal Dataset\n")
    print(df)

    df = prep.sort_transactions(df)

    grouped = prep.group_transactions(df)

    X, y = prep.create_sequences(grouped)

    X_train, X_test, y_train, y_test = prep.chronological_split(X, y)

    X_train = prep.reshape_lstm(X_train)
    X_test = prep.reshape_lstm(X_test)

    auto_data = prep.prepare_autoencoder(df)

    print("\nGenerated Sequences (X)\n")
    print(X)

    print("\nTarget Values (y)\n")
    print(y)

    print("\nTraining Shape :", X_train.shape)
    print("Testing Shape  :", X_test.shape)

    print("\nAutoencoder Dataset Shape :", auto_data.shape)


if __name__ == "__main__":
    main()