import os
import json
import itertools
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import sys
sys.stdout.reconfigure(encoding="utf-8")

# =====================================================
# 1. SETTINGS
# =====================================================

SEED = 42
EPOCHS = 30
PATIENCE = 5

BATCH_SIZES = [32, 64]
HIDDEN_SIZES = [32, 64]
NUM_LAYERS = [1, 2]
DROPOUTS = [0.0, 0.2]
LEARNING_RATES = [0.001, 0.0005]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# =====================================================
# 2. PATHS
# =====================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "..",
        "preprocessing",
        "processed",
        "lstm_daily"
    )
)

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "tuning_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# 3. LOAD DATA
# =====================================================

def load_data():

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))

    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    y_val_raw = np.load(os.path.join(DATA_DIR, "y_val_raw.npy"))
    y_test_raw = np.load(os.path.join(DATA_DIR, "y_test_raw.npy"))

    scaler = joblib.load(
        os.path.join(DATA_DIR, "daily_scaler.pkl")
    )

    print("\nLoaded data:")
    print("Training:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Testing:", X_test.shape, y_test.shape)

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        y_val_raw, y_test_raw,
        scaler
    )


# =====================================================
# 4. LSTM MODEL
# =====================================================

class LSTMModel(nn.Module):

    def __init__(
        self,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 7)

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        x = self.fc1(last_output)
        x = self.relu(x)
        x = self.fc2(x)

        return x


# =====================================================
# 5. TRAIN ONE CONFIGURATION
# =====================================================

def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden_size,
    num_layers,
    dropout,
    learning_rate,
    batch_size
):

    model = LSTMModel(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    criterion = nn.MSELoss()

    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    X_val_tensor = torch.from_numpy(X_val).float().to(DEVICE)
    y_val_tensor = torch.from_numpy(y_val).float().to(DEVICE)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(EPOCHS):

        model.train()
        train_loss_total = 0.0

        for xb, yb in train_loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(xb)
            loss = criterion(predictions, yb)

            loss.backward()

            # Prevent unusually large gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            train_loss_total += loss.item() * len(xb)

        train_loss = train_loss_total / len(train_dataset)

        model.eval()

        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_loss = criterion(
                val_predictions,
                y_val_tensor
            ).item()

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            best_epoch = epoch + 1
            patience_counter = 0

        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)

    return model, best_val_loss, best_epoch


# =====================================================
# 6. HYPERPARAMETER SEARCH
# =====================================================

def main():

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        y_val_raw, y_test_raw,
        scaler
    ) = load_data()

    configurations = list(itertools.product(
        HIDDEN_SIZES,
        NUM_LAYERS,
        DROPOUTS,
        LEARNING_RATES,
        BATCH_SIZES
    ))

    results = []

    print("\nTotal configurations:", len(configurations))

    best_val_mae = float("inf")
    best_config = None
    best_state = None
    best_epoch = None

    # Convert validation labels back to rupees for selection.
    y_val_rupees = y_val_raw.reshape(-1, 1)

    for index, config in enumerate(configurations, start=1):

        hidden_size, num_layers, dropout, lr, batch_size = config

        print("\n" + "=" * 60)
        print(f"Configuration {index}/{len(configurations)}")
        print(
            f"Hidden={hidden_size}, Layers={num_layers}, "
            f"Dropout={dropout}, LR={lr}, Batch={batch_size}"
        )

        model, val_loss, best_epoch_config = train_model(
            X_train, y_train,
            X_val, y_val,
            hidden_size,
            num_layers,
            dropout,
            lr,
            batch_size
        )

        model.eval()

        with torch.no_grad():

            val_pred_scaled = model(
                torch.from_numpy(X_val).float().to(DEVICE)
            ).cpu().numpy()

        val_pred_rupees = scaler.inverse_transform(
            val_pred_scaled.reshape(-1, 1)
        ).reshape(val_pred_scaled.shape)

        val_mae = mean_absolute_error(
            y_val_raw.reshape(-1),
            val_pred_rupees.reshape(-1)
        )

        val_rmse = np.sqrt(mean_squared_error(
            y_val_raw.reshape(-1),
            val_pred_rupees.reshape(-1)
        ))

        print(f"Validation MAE: ₹{val_mae:.2f}")
        print(f"Validation RMSE: ₹{val_rmse:.2f}")

        results.append({
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": lr,
            "batch_size": batch_size,
            "best_epoch": best_epoch_config,
            "validation_loss": val_loss,
            "validation_MAE": val_mae,
            "validation_RMSE": val_rmse
        })

        if val_mae < best_val_mae:

            best_val_mae = val_mae

            best_config = {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "learning_rate": lr,
                "batch_size": batch_size
            }

            best_epoch = best_epoch_config

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    # Save all tuning results.
    results_df = pd.DataFrame(results).sort_values(
        "validation_MAE"
    )

    results_path = os.path.join(
        OUTPUT_DIR,
        "hyperparameter_results.csv"
    )

    results_df.to_csv(results_path, index=False)

    # Save the best model and configuration.
    best_model = LSTMModel(
        hidden_size=best_config["hidden_size"],
        num_layers=best_config["num_layers"],
        dropout=best_config["dropout"]
    )

    best_model.load_state_dict(best_state)

    model_path = os.path.join(
        OUTPUT_DIR,
        "best_lstm_model.pt"
    )

    torch.save(best_model.state_dict(), model_path)

    best_config["best_epoch"] = best_epoch

    config_path = os.path.join(
        OUTPUT_DIR,
        "best_hyperparameters.json"
    )

    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=4)

    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 60)

    print("\nBest configuration:")
    print(json.dumps(best_config, indent=4))

    print(f"\nBest validation MAE: ₹{best_val_mae:.2f}")

    print("\nTop 5 configurations:")
    print(results_df.head(5).to_string(index=False))

    print("\nSaved results:")
    print(results_path)
    print(config_path)
    print(model_path)

    # The test arrays are loaded for the next evaluation stage.
    # Do not use them to select hyperparameters.


if __name__ == "__main__":
    main()
