import os
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():

    print("=" * 60)
    print("XGBOOST TRAINING")
    print("=" * 60)

    # =========================================================
    # 1. Get the directory containing train_xgboost.py
    # =========================================================

    models_dir = os.path.dirname(os.path.abspath(__file__))

    # =========================================================
    # 2. Locate the XGBoost processed data
    #
    # Actual structure:
    #
    # ml/
    # └── models/
    #     ├── train_xgboost.py
    #     └── xgboost/
    #         └── preprocessing/
    #             └── processed/
    #                 ├── X_train.csv
    #                 ├── X_test.csv
    #                 ├── y_train.csv
    #                 └── y_test.csv
    #
    # =========================================================

    processed_dir = os.path.join(
        models_dir,
        "xgboost",
        "preprocessing",
        "processed"
    )

    print("\nProcessed data directory:")
    print(processed_dir)

    # =========================================================
    # 3. Check processed directory
    # =========================================================

    if not os.path.exists(processed_dir):

        raise FileNotFoundError(
            "\nERROR: Processed directory not found!\n\n"
            f"Expected location:\n{processed_dir}\n"
        )

    # =========================================================
    # 4. Define dataset file paths
    # =========================================================

    X_train_path = os.path.join(
        processed_dir,
        "X_train.csv"
    )

    X_test_path = os.path.join(
        processed_dir,
        "X_test.csv"
    )

    y_train_path = os.path.join(
        processed_dir,
        "y_train.csv"
    )

    y_test_path = os.path.join(
        processed_dir,
        "y_test.csv"
    )

    # =========================================================
    # 5. Check that all files exist
    # =========================================================

    required_files = [
        X_train_path,
        X_test_path,
        y_train_path,
        y_test_path
    ]

    print("\nChecking preprocessing files...")

    for file_path in required_files:

        if not os.path.exists(file_path):

            raise FileNotFoundError(
                f"\nERROR: File not found:\n{file_path}"
            )

        print("Found:", os.path.basename(file_path))

    print("\nAll preprocessing files found successfully.")

    # =========================================================
    # 6. Load datasets
    # =========================================================

    print("\nLoading datasets...")

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)

    # =========================================================
    # 7. Convert target columns to Series
    # =========================================================

    if y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]

    if y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]

    # =========================================================
    # 8. Keep only numeric columns
    # =========================================================

    X_train = X_train.select_dtypes(
        include=[np.number]
    )

    X_test = X_test.select_dtypes(
        include=[np.number]
    )

    # =========================================================
    # 9. Make sure train and test have same columns
    # =========================================================

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    # =========================================================
    # 10. Replace infinite values
    # =========================================================

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # =========================================================
    # 11. Handle missing values
    # =========================================================

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # =========================================================
    # 12. Display dataset information
    # =========================================================

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print("X_train shape :", X_train.shape)
    print("X_test shape  :", X_test.shape)
    print("y_train shape :", y_train.shape)
    print("y_test shape  :", y_test.shape)

    print("\nNumber of features:", X_train.shape[1])

    # =========================================================
    # 13. Create XGBoost model
    # =========================================================

    print("\n" + "=" * 60)
    print("CREATING XGBOOST MODEL")
    print("=" * 60)

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    print("Model created successfully.")

    # =========================================================
    # 14. Train model
    # =========================================================

    print("\n" + "=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)

    model.fit(
        X_train,
        y_train
    )

    print("\nTraining completed successfully.")

    # =========================================================
    # 15. Generate predictions
    # =========================================================

    print("\nGenerating predictions...")

    y_pred = model.predict(
        X_test
    )

    print("Predictions generated successfully.")

    # =========================================================
    # 16. Calculate evaluation metrics
    # =========================================================

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    # =========================================================
    # 17. Display results
    # =========================================================

    print("\n" + "=" * 60)
    print("XGBOOST RESULTS")
    print("=" * 60)

    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    # =========================================================
    # 18. Save trained model
    # =========================================================

    model_path = os.path.join(
        models_dir,
        "xgboost_model.json"
    )

    model.save_model(
        model_path
    )

    print("\nTrained model saved to:")

    print(model_path)

    # =========================================================
    # 19. Save predictions
    # =========================================================

    predictions = pd.DataFrame({

        "actual": y_test.values,

        "predicted": y_pred

    })

    predictions_path = os.path.join(
        processed_dir,
        "xgboost_predictions.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=False
    )

    print("\nPredictions saved to:")

    print(predictions_path)

    # =========================================================
    # 20. Display first 10 predictions
    # =========================================================

    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)

    print(
        predictions.head(10).to_string(
            index=False
        )
    )

    # =========================================================
    # 21. Feature importance
    # =========================================================

    print("\n" + "=" * 60)
    print("TOP FEATURE IMPORTANCE")
    print("=" * 60)

    feature_importance = pd.DataFrame({

        "feature": X_train.columns,

        "importance": model.feature_importances_

    })

    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False
    )

    print(
        feature_importance.head(10).to_string(
            index=False
        )
    )

    # =========================================================
    # 22. Save feature importance
    # =========================================================

    feature_importance_path = os.path.join(
        processed_dir,
        "xgboost_feature_importance.csv"
    )

    feature_importance.to_csv(
        feature_importance_path,
        index=False
    )

    print("\nFeature importance saved to:")

    print(feature_importance_path)

    # =========================================================
    # 23. Finished
    # =========================================================

    print("\n" + "=" * 60)
    print("XGBOOST TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()