import os
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# =========================================================
# Configuration
# =========================================================

SEED = 42

HISTORY_DAYS = 90
FORECAST_DAYS = 7
INPUT_FEATURES = 6

EPOCHS = 5
BATCH_SIZE = 512
LEARNING_RATE = 0.001

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

torch.manual_seed(SEED)
np.random.seed(SEED)


# =========================================================
# Device
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")

if DEVICE.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# =========================================================
# Model
# =========================================================

class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size=INPUT_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=FORECAST_DAYS,
        dropout=DROPOUT
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc1 = nn.Linear(
            hidden_size,
            32
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            32,
            output_size
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        # Last timestep
        out = out[:, -1, :]

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out


# =========================================================
# Load prepared data
# =========================================================

def load_data(base_dir):

    processed_dir = os.path.join(
        base_dir,
        "preprocessing",
        "processed"
    )

    X_train = np.load(
        os.path.join(
            processed_dir,
            "X_train_category.npy"
        )
    )

    X_test = np.load(
        os.path.join(
            processed_dir,
            "X_test_category.npy"
        )
    )

    y_train = np.load(
        os.path.join(
            processed_dir,
            "y_train_category.npy"
        )
    )

    y_test = np.load(
        os.path.join(
            processed_dir,
            "y_test_category.npy"
        )
    )

    y_train_raw = np.load(
        os.path.join(
            processed_dir,
            "y_train_raw_category.npy"
        )
    )

    y_test_raw = np.load(
        os.path.join(
            processed_dir,
            "y_test_raw_category.npy"
        )
    )

    train_metadata = pd.read_csv(
        os.path.join(
            processed_dir,
            "train_metadata.csv"
        )
    )

    test_metadata = pd.read_csv(
        os.path.join(
            processed_dir,
            "test_metadata.csv"
        )
    )

    # Global scaler used during sequence preparation.
    global_scaler = joblib.load(
        os.path.join(
            processed_dir,
            "category_scaler.pkl"
        )
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_raw,
        y_test_raw,
        train_metadata,
        test_metadata,
        global_scaler
    )


# =========================================================
# Recover raw spending from globally-scaled X
# =========================================================

def recover_raw_input_spending(
    X_scaled,
    global_scaler
):

    X_raw = X_scaled.copy()

    spending_scaled = X_scaled[:, :, 0]

    raw_shape = spending_scaled.shape

    spending_raw = global_scaler.inverse_transform(
        spending_scaled.reshape(-1, 1)
    ).reshape(raw_shape)

    X_raw[:, :, 0] = spending_raw

    return X_raw


# =========================================================
# Train one category
# =========================================================

def train_category(
    category,
    category_index,
    X_train_raw,
    X_test_raw,
    y_train_raw,
    y_test_raw,
    output_dir
):

    print("\n" + "=" * 70)
    print(
        f"[{category_index}] TRAINING CATEGORY: {category}"
    )
    print("=" * 70)

    print(
        f"Training sequences: {len(X_train_raw):,}"
    )

    print(
        f"Testing sequences : {len(X_test_raw):,}"
    )

    if len(X_train_raw) == 0:
        print("No training sequences. Skipping.")
        return None

    if len(X_test_raw) == 0:
        print("No testing sequences. Skipping.")
        return None

    # -----------------------------------------------------
    # Category-specific scaler
    # -----------------------------------------------------

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    scaler.fit(
        y_train_raw.reshape(-1, 1)
    )

    # Scale spending feature only.
    X_train_scaled = X_train_raw.copy()
    X_test_scaled = X_test_raw.copy()

    X_train_scaled[:, :, 0] = scaler.transform(
        X_train_raw[:, :, 0].reshape(-1, 1)
    ).reshape(
        X_train_raw[:, :, 0].shape
    )

    X_test_scaled[:, :, 0] = scaler.transform(
        X_test_raw[:, :, 0].reshape(-1, 1)
    ).reshape(
        X_test_raw[:, :, 0].shape
    )

    y_train_scaled = scaler.transform(
        y_train_raw.reshape(-1, 1)
    ).reshape(
        y_train_raw.shape
    )

    y_test_scaled = scaler.transform(
        y_test_raw.reshape(-1, 1)
    ).reshape(
        y_test_raw.shape
    )

    # -----------------------------------------------------
    # Tensor data
    # -----------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train_scaled,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32
    )

    # -----------------------------------------------------
    # DataLoader
    # -----------------------------------------------------

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda")
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = LSTMModel().to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    start_time = time.time()

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        total_loss = 0.0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(
                DEVICE,
                non_blocking=True
            )

            batch_y = batch_y.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            predictions = model(
                batch_X
            )

            loss = criterion(
                predictions,
                batch_y
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * batch_X.size(0)
            )

        epoch_loss = (
            total_loss
            / len(train_dataset)
        )

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch "
                f"{epoch:02d}/{EPOCHS} "
                f"- MSE: {epoch_loss:.6f}"
            )

    training_time = time.time() - start_time

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    model.eval()

    # Use batched inference so a huge test set
    # does not consume unnecessary GPU memory.

    prediction_loader = DataLoader(
        TensorDataset(X_test_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(DEVICE.type == "cuda")
    )

    predictions_scaled = []

    with torch.no_grad():

        for (batch_X,) in prediction_loader:

            batch_X = batch_X.to(
                DEVICE,
                non_blocking=True
            )

            batch_predictions = model(
                batch_X
            )

            predictions_scaled.append(
                batch_predictions.cpu().numpy()
            )

    predictions_scaled = np.concatenate(
        predictions_scaled,
        axis=0
    )

    # -----------------------------------------------------
    # Convert predictions back to INR
    # -----------------------------------------------------

    predictions_raw = scaler.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).reshape(
        predictions_scaled.shape
    )

    # -----------------------------------------------------
    # Metrics in original INR scale
    # -----------------------------------------------------

    mse = mean_squared_error(
        y_test_raw.flatten(),
        predictions_raw.flatten()
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_test_raw.flatten(),
        predictions_raw.flatten()
    )

    r2 = r2_score(
        y_test_raw.flatten(),
        predictions_raw.flatten()
    )

    print("\nRESULTS")

    print(
        f"MSE : INR² {mse:,.4f}"
    )

    print(
        f"RMSE: INR {rmse:,.2f}"
    )

    print(
        f"MAE : INR {mae:,.2f}"
    )

    print(
        f"R²  : {r2:.4f}"
    )

    # -----------------------------------------------------
    # Per-day metrics
    # -----------------------------------------------------

    per_day_metrics = {}

    print("\nPer-day metrics:")

    for day in range(
        FORECAST_DAYS
    ):

        actual = y_test_raw[:, day]

        predicted = predictions_raw[:, day]

        day_mse = mean_squared_error(
            actual,
            predicted
        )

        day_mae = mean_absolute_error(
            actual,
            predicted
        )

        day_r2 = r2_score(
            actual,
            predicted
        )

        per_day_metrics[
            f"Day_{day + 1}"
        ] = {
            "mse": float(day_mse),
            "mae": float(day_mae),
            "r2": float(day_r2)
        }

        print(
            f"Day {day + 1}: "
            f"MSE={day_mse:,.2f}, "
            f"MAE={day_mae:,.2f}, "
            f"R²={day_r2:.4f}"
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    safe_category = (
        str(category)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    category_dir = os.path.join(
        output_dir,
        safe_category
    )

    os.makedirs(
        category_dir,
        exist_ok=True
    )

    # Model
    model_path = os.path.join(
        category_dir,
        "model.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    # Category scaler
    scaler_path = os.path.join(
        category_dir,
        "scaler.pkl"
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    # Predictions
    prediction_df = pd.DataFrame()

    for day in range(
        FORECAST_DAYS
    ):

        prediction_df[
            f"Actual_Day_{day + 1}"
        ] = y_test_raw[:, day]

        prediction_df[
            f"Predicted_Day_{day + 1}"
        ] = predictions_raw[:, day]

    prediction_df.to_csv(
        os.path.join(
            category_dir,
            "predictions.csv"
        ),
        index=False
    )

    # Metrics
    metrics = {
        "category": category,
        "model": "2-layer LSTM",
        "history_days": HISTORY_DAYS,
        "forecast_days": FORECAST_DAYS,
        "input_features": INPUT_FEATURES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "train_sequences": int(
            len(X_train_raw)
        ),
        "test_sequences": int(
            len(X_test_raw)
        ),
        "mse_inr_squared": float(mse),
        "rmse_inr": float(rmse),
        "mae_inr": float(mae),
        "r2": float(r2),
        "training_time_seconds": float(
            training_time
        ),
        "per_day": per_day_metrics
    }

    with open(
        os.path.join(
            category_dir,
            "metrics.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    return metrics


# =========================================================
# Main
# =========================================================

def main():

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    output_dir = os.path.join(
        base_dir,
        "models"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_raw,
        y_test_raw,
        train_metadata,
        test_metadata,
        global_scaler
    ) = load_data(
        base_dir
    )

    print(
        "\nLoaded prepared data:"
    )

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_test : {X_test.shape}"
    )

    print(
        f"y_train: {y_train.shape}"
    )

    print(
        f"y_test : {y_test.shape}"
    )

    # -----------------------------------------------------
    # Recover raw spending from the globally scaled
    # feature matrix.
    # -----------------------------------------------------

    X_train_raw = recover_raw_input_spending(
        X_train,
        global_scaler
    )

    X_test_raw = recover_raw_input_spending(
        X_test,
        global_scaler
    )

    # -----------------------------------------------------
    # Categories
    # -----------------------------------------------------

    categories = sorted(
        train_metadata["category"]
        .dropna()
        .unique()
    )

    print(
        "\nCategories:"
    )

    for category in categories:
        print(f"  - {category}")

    # -----------------------------------------------------
    # Train category-specific models
    # -----------------------------------------------------

    all_metrics = []

    for index, category in enumerate(
        categories,
        start=1
    ):

        train_mask = (
            train_metadata["category"]
            .astype(str)
            .values
            == str(category)
        )

        test_mask = (
            test_metadata["category"]
            .astype(str)
            .values
            == str(category)
        )

        category_X_train = (
            X_train_raw[train_mask]
        )

        category_X_test = (
            X_test_raw[test_mask]
        )

        category_y_train = (
            y_train_raw[train_mask]
        )

        category_y_test = (
            y_test_raw[test_mask]
        )

        metrics = train_category(
            category=category,
            category_index=index,
            X_train_raw=category_X_train,
            X_test_raw=category_X_test,
            y_train_raw=category_y_train,
            y_test_raw=category_y_test,
            output_dir=output_dir
        )

        if metrics is not None:

            all_metrics.append(
                metrics
            )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summary_rows = []

    for metrics in all_metrics:

        summary_rows.append({
            "category":
                metrics["category"],

            "train_sequences":
                metrics["train_sequences"],

            "test_sequences":
                metrics["test_sequences"],

            "MSE":
                metrics["mse_inr_squared"],

            "MAE":
                metrics["mae_inr"],

            "R2":
                metrics["r2"],

            "training_time_seconds":
                metrics[
                    "training_time_seconds"
                ]
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = os.path.join(
        output_dir,
        "category_lstm_metrics.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("CATEGORY-WISE LSTM RESULTS")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved summary to:\n"
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()