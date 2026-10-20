import os
import json
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "..",
        "preprocessing",
        "processed",
        "lstm_daily"
    )
)

print("=" * 55)
print("LSTM DATA DIAGNOSTICS")
print("=" * 55)

scaler = joblib.load(
    os.path.join(DATA_DIR, "daily_scaler.pkl")
)

print("\nSCALER")
print("Minimum:", scaler.data_min_)
print("Maximum:", scaler.data_max_)
print("Scale:", scaler.scale_)

for name in [
    "y_train.npy",
    "y_val.npy",
    "y_test.npy",
    "y_train_raw.npy",
    "y_val_raw.npy",
    "y_test_raw.npy"
]:
    path = os.path.join(DATA_DIR, name)
    values = np.load(path)

    print(f"\n{name}")
    print("Shape:", values.shape)
    print("Minimum:", np.min(values))
    print("Maximum:", np.max(values))
    print("Mean:", np.mean(values))
    print("Median:", np.median(values))
    print("99th percentile:", np.percentile(values, 99))
    print("99.9th percentile:", np.percentile(values, 99.9))
    print("NaN values:", np.isnan(values).sum())

print("\nUSER DISTRIBUTION")
users_path = os.path.join(DATA_DIR, "test_users.npy")

if os.path.exists(users_path):
    users = np.load(users_path)

    print("Test sequences:", len(users))
    print("Unique test users:", len(np.unique(users)))

print("\nSAVED BEST CONFIGURATION")
config_path = os.path.join(
    BASE_DIR,
    "tuning_results",
    "best_hyperparameters.json"
)

if os.path.exists(config_path):
    with open(config_path, "r") as file:
        print(json.dumps(json.load(file), indent=4))

print("\nDiagnostic complete. No files modified.")

