import os
import pandas as pd
import joblib

from xgboost import XGBRegressor


def main():

    print("=" * 60)
    print("XGBOOST TRAINING")
    print("=" * 60)

    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    processed_dir = os.path.join(
        current_dir,
        "preprocessing",
        "processed"
    )

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    X_train = pd.read_csv(
        os.path.join(
            processed_dir,
            "X_train.csv"
        )
    )

    y_train = pd.read_csv(
        os.path.join(
            processed_dir,
            "y_train.csv"
        )
    ).squeeze()

    print("\nTraining data:")
    print(X_train.shape)

    # --------------------------------------------------
    # Create XGBoost model
    # --------------------------------------------------

    model = XGBRegressor(

        n_estimators=300,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="reg:squarederror",

        random_state=42
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    model_dir = os.path.join(
        current_dir,
        "saved_model"
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    model_path = os.path.join(
        model_dir,
        "xgboost_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print("\nModel saved at:")
    print(model_path)


if __name__ == "__main__":
    main()