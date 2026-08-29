import pandas as pd
import numpy as np


class CNNSequencePreparation:

    def __init__(self, window_size=3):

        self.window_size = window_size

    # ---------------------------------------------
    # Sort transactions
    # ---------------------------------------------

    def sort_transactions(self, df):

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df = df.sort_values(
            ["user_id", "date"]
        )

        return df

    # ---------------------------------------------
    # Create CNN sequences
    # ---------------------------------------------

    def create_sequences(self, df):

        X = []
        y = []

        for user_id, user_data in df.groupby(
            "user_id"
        ):

            amounts = (
                user_data["amount"]
                .values
                .astype(np.float32)
            )

            if len(amounts) <= self.window_size:
                continue

            for i in range(
                len(amounts) - self.window_size
            ):

                sequence = amounts[
                    i:i + self.window_size
                ]

                target = amounts[
                    i + self.window_size
                ]

                X.append(sequence)

                y.append(target)

        X = np.array(X, dtype=np.float32)

        y = np.array(y, dtype=np.float32)

        return X, y

    # ---------------------------------------------
    # Chronological split
    # ---------------------------------------------

    def chronological_split(
        self,
        X,
        y,
        train_ratio=0.8
    ):

        split_index = int(
            len(X) * train_ratio
        )

        X_train = X[:split_index]

        X_test = X[split_index:]

        y_train = y[:split_index]

        y_test = y[split_index:]

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )

    # ---------------------------------------------
    # CNN input reshape
    # ---------------------------------------------

    def reshape_for_cnn(self, X):

        return X.reshape(
            X.shape[0],
            X.shape[1],
            1
        )


def main():

    df = pd.read_csv(
        "processed/feature_engineered_dataset.csv"
    )

    prep = CNNSequencePreparation(
        window_size=3
    )

    df = prep.sort_transactions(df)

    X, y = prep.create_sequences(df)

    print("\nSequences created")

    print("X shape:", X.shape)

    print("y shape:", y.shape)

    X_train, X_test, y_train, y_test = (
        prep.chronological_split(X, y)
    )

    X_train = prep.reshape_for_cnn(
        X_train
    )

    X_test = prep.reshape_for_cnn(
        X_test
    )

    print("\nCNN data shapes")

    print("X_train:", X_train.shape)

    print("X_test :", X_test.shape)

    print("y_train:", y_train.shape)

    print("y_test :", y_test.shape)

    np.save(
        "processed/X_train_cnn.npy",
        X_train
    )

    np.save(
        "processed/X_test_cnn.npy",
        X_test
    )

    np.save(
        "processed/y_train_cnn.npy",
        y_train
    )

    np.save(
        "processed/y_test_cnn.npy",
        y_test
    )

    print("\nCNN preprocessing completed.")


if __name__ == "__main__":
    main()