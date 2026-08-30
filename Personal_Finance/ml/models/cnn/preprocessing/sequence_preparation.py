import pandas as pd
import numpy as np
from pathlib import Path


# =====================================================
# Find Project Root
# =====================================================

def find_project_root():

    current = Path(__file__).resolve()

    for parent in current.parents:

        if (
            (parent / ".git").exists()
            or (parent / "Personal_Finance").exists()
        ):
            return parent

    raise FileNotFoundError(
        "Could not locate project root."
    )


# =====================================================
# CNN Sequence Preparation
# =====================================================

class CNNSequencePreparation:

    def __init__(self, window_size=3):

        self.window_size = window_size

    # -------------------------------------------------
    # Sort transactions by user and date
    # -------------------------------------------------

    def sort_transactions(self, df):

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df = df.sort_values(
            ["user_id", "date"]
        )

        return df

    # -------------------------------------------------
    # Create CNN sequences
    # -------------------------------------------------

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

        X = np.array(
            X,
            dtype=np.float32
        )

        y = np.array(
            y,
            dtype=np.float32
        )

        return X, y

    # -------------------------------------------------
    # Chronological Train-Test Split
    # -------------------------------------------------

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

    # -------------------------------------------------
    # CNN Input Reshape
    # -------------------------------------------------

    def reshape_for_cnn(self, X):

        return X.reshape(
            X.shape[0],
            X.shape[1],
            1
        )


# =====================================================
# Main
# =====================================================

def main():

    print("=" * 60)
    print("CNN SEQUENCE PREPARATION")
    print("=" * 60)

    # -------------------------------------------------
    # Find project root
    # -------------------------------------------------

    project_root = find_project_root()

    processed_dir = (
        project_root / "processed"
    )

    # -------------------------------------------------
    # Input dataset
    # -------------------------------------------------

    input_path = (
        processed_dir
        / "feature_engineered_dataset_cnn.csv"
    )

    print("\nLoading dataset:")
    print(input_path)

    if not input_path.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n{input_path}\n"
            "Run preprocessing.py first."
        )

    df = pd.read_csv(
        input_path
    )

    # -------------------------------------------------
    # Create sequences
    # -------------------------------------------------

    prep = CNNSequencePreparation(
        window_size=3
    )

    df = prep.sort_transactions(df)

    X, y = prep.create_sequences(df)

    print("\nSequences created")

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # -------------------------------------------------
    # Train-test split
    # -------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = prep.chronological_split(
        X,
        y
    )

    print("\nTrain-test split")

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    # -------------------------------------------------
    # CNN reshape
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Save CNN datasets
    # -------------------------------------------------

    np.save(
        processed_dir / "X_train_cnn.npy",
        X_train
    )

    np.save(
        processed_dir / "X_test_cnn.npy",
        X_test
    )

    np.save(
        processed_dir / "y_train_cnn.npy",
        y_train
    )

    np.save(
        processed_dir / "y_test_cnn.npy",
        y_test
    )

    print("\nCNN sequence data saved successfully.")

    print("\nSaved files:")

    print(
        processed_dir / "X_train_cnn.npy"
    )

    print(
        processed_dir / "X_test_cnn.npy"
    )

    print(
        processed_dir / "y_train_cnn.npy"
    )

    print(
        processed_dir / "y_test_cnn.npy"
    )

    print("\n" + "=" * 60)
    print("SEQUENCE PREPARATION COMPLETED")
    print("=" * 60)


# =====================================================
# Run
# =====================================================

if __name__ == "__main__":

    main()