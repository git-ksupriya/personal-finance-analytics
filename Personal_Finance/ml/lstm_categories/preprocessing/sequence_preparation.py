import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler


class CategorySequencePreparation:

    def __init__(
        self,
        history_months=12,
        forecast_months=7,
        train_ratio=0.8
    ):

        self.history_months = history_months
        self.forecast_months = forecast_months
        self.train_ratio = train_ratio

        self.feature_columns = [
            "monthly_spending",
            "month_sin",
            "month_cos",
            "quarter_sin",
            "quarter_cos",
            "days_in_month",
        ]

    # --------------------------------------------------
    # Sort
    # --------------------------------------------------

    def sort_data(self, df):

        df = df.copy()

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        return df.sort_values(
            [
                "user_id",
                "category",
                "date"
            ]
        ).reset_index(drop=True)

    # --------------------------------------------------
    # Create training sequences
    # --------------------------------------------------

    def create_train_sequences(
        self,
        df
    ):

        X = []
        y = []

        metadata = []

        for (
            user_id,
            category
        ), group in df.groupby(
            ["user_id", "category"]
        ):

            group = group.sort_values(
                "date"
            )

            if len(group) < (
                self.history_months
                + self.forecast_months
            ):
                continue

            split_idx = int(
                len(group) *
                self.train_ratio
            )

            train_group = group.iloc[
                :split_idx
            ]

            if len(train_group) < (
                self.history_months
                + self.forecast_months
            ):
                continue

            feature_values = (
                train_group[
                    self.feature_columns
                ]
                .values
                .astype(np.float32)
            )

            target_values = (
                train_group[
                    "monthly_spending"
                ]
                .values
                .astype(np.float32)
            )

            for i in range(
                len(train_group)
                -
                self.history_months
                -
                self.forecast_months
                + 1
            ):

                X.append(
                    feature_values[
                        i:
                        i + self.history_months
                    ]
                )

                y.append(
                    target_values[
                        i + self.history_months:
                        i
                        + self.history_months
                        + self.forecast_months
                    ]
                )

                metadata.append({
                    "user_id": user_id,
                    "category": category,
                    "forecast_start": train_group[
                        "date"
                    ].iloc[
                        i + self.history_months
                    ],
                    "forecast_end": train_group[
                        "date"
                    ].iloc[
                        i
                        + self.history_months
                        + self.forecast_months
                        - 1
                    ]
                })

        return (
            np.asarray(
                X,
                dtype=np.float32
            ),
            np.asarray(
                y,
                dtype=np.float32
            ),
            pd.DataFrame(metadata)
        )

    # --------------------------------------------------
    # Create TEST sequences
    # --------------------------------------------------

    def create_test_sequences(
        self,
        df
    ):

        X = []
        y = []

        metadata = []

        for (
            user_id,
            category
        ), group in df.groupby(
            ["user_id", "category"]
        ):

            group = group.sort_values(
                "date"
            ).reset_index(
                drop=True
            )

            split_idx = int(
                len(group) *
                self.train_ratio
            )

            if split_idx >= len(group):
                continue

            # Include the final 12 months before
            # the split as available history.
            context_start = max(
                0,
                split_idx
                - self.history_months
            )

            evaluation_group = group.iloc[
                context_start:
            ].copy()

            if len(evaluation_group) < (
                self.history_months
                + self.forecast_months
            ):
                continue

            feature_values = (
                evaluation_group[
                    self.feature_columns
                ]
                .values
                .astype(np.float32)
            )

            target_values = (
                evaluation_group[
                    "monthly_spending"
                ]
                .values
                .astype(np.float32)
            )

            dates = evaluation_group[
                "date"
            ].values

            # Every target window must begin
            # at or after the test boundary.
            for i in range(
                self.history_months,
                len(evaluation_group)
                - self.forecast_months
                + 1
            ):

                actual_target_start = (
                    context_start + i
                )

                if actual_target_start < split_idx:
                    continue

                X.append(
                    feature_values[
                        i - self.history_months:
                        i
                    ]
                )

                y.append(
                    target_values[
                        i:
                        i + self.forecast_months
                    ]
                )

                metadata.append({
                    "user_id": user_id,
                    "category": category,
                    "forecast_start": dates[i],
                    "forecast_end": dates[
                        i + self.forecast_months - 1
                    ]
                })

        return (
            np.asarray(
                X,
                dtype=np.float32
            ),
            np.asarray(
                y,
                dtype=np.float32
            ),
            pd.DataFrame(metadata)
        )

    # --------------------------------------------------
    # Scaling
    # --------------------------------------------------

    def fit_scaler(
        self,
        train_df
    ):

        self.scaler = MinMaxScaler(
            feature_range=(0, 1)
        )

        # Scale spending only.
        self.scaler.fit(
            train_df[
                ["monthly_spending"]
            ].values
        )

        return self.scaler

    def transform_features(
        self,
        X
    ):

        X = X.copy()

        spending = X[:, :, 0]

        original_shape = spending.shape

        X[:, :, 0] = (
            self.scaler
            .transform(
                spending.reshape(-1, 1)
            )
            .reshape(original_shape)
        )

        return X

    def transform_targets(
        self,
        y
    ):

        return (
            self.scaler
            .transform(
                y.reshape(-1, 1)
            )
            .reshape(y.shape)
        )

    def inverse_transform_targets(
        self,
        y
    ):

        return (
            self.scaler
            .inverse_transform(
                y.reshape(-1, 1)
            )
            .reshape(y.shape)
        )

    # --------------------------------------------------
    # Save scaler
    # --------------------------------------------------

    def save_scaler(
        self,
        filepath
    ):

        os.makedirs(
            os.path.dirname(filepath),
            exist_ok=True
        )

        joblib.dump(
            self.scaler,
            filepath
        )


# ======================================================
# Main
# ======================================================

def main():

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    input_path = os.path.join(
        base_dir,
        "processed",
        "monthly_category_spending.csv"
    )

    output_dir = os.path.join(
        base_dir,
        "processed",
        "monthly"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    df = pd.read_csv(
        input_path
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    prep = CategorySequencePreparation(
        history_months=12,
        forecast_months=7,
        train_ratio=0.8
    )

    df = prep.sort_data(
        df
    )

    # --------------------------------------------------
    # Training sequences
    # --------------------------------------------------

    X_train, y_train, train_metadata = (
        prep.create_train_sequences(
            df
        )
    )
    

    # --------------------------------------------------
    # Test sequences
    # --------------------------------------------------

    X_test, y_test, test_metadata = (
        prep.create_test_sequences(
            df
        )
    )

    print(
        train_metadata.groupby("category").size()
    )

    print(
        test_metadata.groupby("category").size()
    )

    print(
        "\nRaw sequence shapes:"
    )

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "X_test :",
        X_test.shape
    )

    print(
        "y_test :",
        y_test.shape
    )

    # --------------------------------------------------
    # Fit scaler ONLY using training period
    # --------------------------------------------------

    # Build a dataframe consisting of only
    # pre-split rows for every user/category.
    train_parts = []

    for (
        user_id,
        category
    ), group in df.groupby(
        ["user_id", "category"]
    ):

        group = group.sort_values(
            "date"
        )

        split_idx = int(
            len(group) * prep.train_ratio
        )

        train_parts.append(
            group.iloc[:split_idx]
        )

    train_df = pd.concat(
        train_parts,
        ignore_index=True
    )

    prep.fit_scaler(
        train_df
    )

    # --------------------------------------------------
    # Scale
    # --------------------------------------------------

    X_train_scaled = (
        prep.transform_features(
            X_train
        )
    )

    y_train_scaled = (
        prep.transform_targets(
            y_train
        )
    )

    X_test_scaled = (
        prep.transform_features(
            X_test
        )
    )

    y_test_scaled = (
        prep.transform_targets(
            y_test
        )
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    np.save(
        os.path.join(
            output_dir,
            "X_train_category.npy"
        ),
        X_train_scaled
    )

    np.save(
        os.path.join(
            output_dir,
            "X_test_category.npy"
        ),
        X_test_scaled
    )

    np.save(
        os.path.join(
            output_dir,
            "y_train_category.npy"
        ),
        y_train_scaled
    )

    np.save(
        os.path.join(
            output_dir,
            "y_test_category.npy"
        ),
        y_test_scaled
    )

    np.save(
        os.path.join(
            output_dir,
            "y_train_raw_category.npy"
        ),
        y_train
    )

    np.save(
        os.path.join(
            output_dir,
            "y_test_raw_category.npy"
        ),
        y_test
    )

    train_metadata.to_csv(
        os.path.join(
            output_dir,
            "train_metadata.csv"
        ),
        index=False
    )

    test_metadata.to_csv(
        os.path.join(
            output_dir,
            "test_metadata.csv"
        ),
        index=False
    )

    prep.save_scaler(
        os.path.join(
            output_dir,
            "category_scaler.pkl"
        )
    )

    print(
        "\nSequence preparation complete."
    )


if __name__ == "__main__":
    main()