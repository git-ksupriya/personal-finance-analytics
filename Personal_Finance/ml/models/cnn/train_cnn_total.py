import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SIZE = 30     # Use previous 30 days
FORECAST_DAYS = 7      # Predict next 7 days

EPOCHS = 50
BATCH_SIZE = 32

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "preprocessing",
    "processed",
    "feature_engineered_dataset.csv"
)

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("1D CNN - TOTAL SPENDING FORECAST")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

print("\nDataset loaded:")
print(df.shape)


# ============================================================
# SELECT EXPENSE TRANSACTIONS
# ============================================================

df = df[df["transaction_type"] == "Expense"].copy()

print("\nExpense records:", len(df))


# ============================================================
# DAILY TOTAL SPENDING
# ============================================================

daily_spending = (
    df.groupby("date")["amount"]
    .sum()
    .reset_index()
)

daily_spending = daily_spending.sort_values("date")


# Create continuous daily dates
date_range = pd.date_range(
    start=daily_spending["date"].min(),
    end=daily_spending["date"].max(),
    freq="D"
)

daily_spending = (
    daily_spending
    .set_index("date")
    .reindex(date_range, fill_value=0)
    .rename_axis("date")
    .reset_index()
)

print("\nDaily spending data:")
print(daily_spending.head())

print("\nNumber of days:", len(daily_spending))


# ============================================================
# EXTRACT VALUES
# ============================================================

values = daily_spending["amount"].values.reshape(-1, 1)


# ============================================================
# CHRONOLOGICAL TRAIN/TEST SPLIT
# ============================================================

split_index = int(len(values) * 0.8)

train_values = values[:split_index]
test_values = values[split_index:]


# ============================================================
# SCALE DATA
# IMPORTANT: FIT ONLY ON TRAIN DATA
# ============================================================

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_values)

# Transform test using training scaler
test_scaled = scaler.transform(test_values)


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(data, window_size, forecast_days):

    X = []
    y = []

    for i in range(
        len(data) - window_size - forecast_days + 1
    ):

        X.append(
            data[i:i + window_size]
        )

        y.append(
            data[
                i + window_size:
                i + window_size + forecast_days
            ]
        )

    return np.array(X), np.array(y)


X_train, y_train = create_sequences(
    train_scaled,
    WINDOW_SIZE,
    FORECAST_DAYS
)

# For test sequences, include the last WINDOW_SIZE
# training observations as context.
test_with_context = np.concatenate(
    [
        train_scaled[-WINDOW_SIZE:],
        test_scaled
    ]
)

X_test, y_test = create_sequences(
    test_with_context,
    WINDOW_SIZE,
    FORECAST_DAYS
)


print("\nSequence shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ============================================================
# 1D CNN MODEL
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(WINDOW_SIZE, 1)
    ),

    tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        padding="causal"
    ),

    tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        padding="causal"
    ),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.2),

    # Seven outputs = seven future days
    tf.keras.layers.Dense(
        FORECAST_DAYS
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
)

print("\nModel summary:")
model.summary()


# ============================================================
# TRAIN
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train.reshape(-1, FORECAST_DAYS),
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)


# ============================================================
# PREDICTION
# ============================================================

y_pred_scaled = model.predict(
    X_test,
    verbose=0
)


# ============================================================
# INVERSE TRANSFORM
# ============================================================

y_test_original = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).reshape(-1, FORECAST_DAYS)

y_pred_original = scaler.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).reshape(-1, FORECAST_DAYS)


# ============================================================
# EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test_original.flatten(),
    y_pred_original.flatten()
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_original.flatten(),
        y_pred_original.flatten()
    )
)

r2 = r2_score(
    y_test_original.flatten(),
    y_pred_original.flatten()
)


print("\n" + "=" * 60)
print("TOTAL SPENDING CNN RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "cnn_total_spending.keras"
)

model.save(model_path)

print("\nModel saved to:")
print(model_path)

# ============================================================
# SAMPLE 7-DAY ACTUAL VS PREDICTED
# ============================================================

print("\n" + "=" * 70)
print("7-DAY TOTAL SPENDING: ACTUAL VS PREDICTED")
print("=" * 70)

print(
    f"{'Day':<10}"
    f"{'Actual':>15}"
    f"{'Predicted':>15}"
    f"{'Difference':>15}"
    f"{'Error %':>12}"
)

print("-" * 67)

for day in range(FORECAST_DAYS):

    actual = y_test_original[0, day]
    predicted = y_pred_original[0, day]

    difference = actual - predicted

    if actual != 0:
        error_percentage = (
            abs(difference) / abs(actual)
        ) * 100
    else:
        error_percentage = 0

    print(
        f"{'Day ' + str(day + 1):<10}"
        f"{'Rs.' + format(actual, '.2f'):>15}"
        f"{'Rs.' + format(predicted, '.2f'):>15}"
        f"{'Rs.' + format(difference, '.2f'):>15}"
        f"{error_percentage:>10.2f}%"
    )

print("=" * 70)

print("\nOverall Metrics")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

print("\nTraining completed successfully.")