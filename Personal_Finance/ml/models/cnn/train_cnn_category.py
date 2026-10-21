import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_SIZE = 14
FORECAST_DAYS = 7

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
print("1D CNN - CATEGORY-WISE SPENDING FORECAST")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df[
    df["transaction_type"] == "Expense"
].copy()

print("\nExpense records:", len(df))


# ============================================================
# FIND CATEGORIES
# ============================================================

categories = sorted(
    df["category"]
    .dropna()
    .unique()
)

print("\nCategories:")
for category in categories:
    print("-", category)

num_categories = len(categories)

print("\nNumber of categories:", num_categories)


# ============================================================
# DAILY CATEGORY SPENDING
# ============================================================

daily_category = (
    df.groupby(
        ["date", "category"]
    )["amount"]
    .sum()
    .unstack(fill_value=0)
)


# Make sure all categories exist
daily_category = daily_category.reindex(
    columns=categories,
    fill_value=0
)


# Create continuous dates
date_range = pd.date_range(
    start=daily_category.index.min(),
    end=daily_category.index.max(),
    freq="D"
)

daily_category = daily_category.reindex(
    date_range,
    fill_value=0
)

daily_category.index.name = "date"


print("\nDaily category matrix:")
print(daily_category.head())

print(
    "\nShape:",
    daily_category.shape
)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

values = daily_category.values.astype(float)


# ============================================================
# CHRONOLOGICAL TRAIN/TEST SPLIT
# ============================================================

split_index = int(
    len(values) * 0.8
)

train_values = values[:split_index]
test_values = values[split_index:]


# ============================================================
# SCALE EACH CATEGORY
# ============================================================

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(
    train_values
)

test_scaled = scaler.transform(
    test_values
)


# ============================================================
# CREATE MULTIVARIATE SEQUENCES
# ============================================================

def create_sequences(
    data,
    window_size,
    forecast_days
):

    X = []
    y = []

    for i in range(
        len(data)
        - window_size
        - forecast_days
        + 1
    ):

        X.append(
            data[
                i:i + window_size
            ]
        )

        y.append(
            data[
                i + window_size:
                i + window_size + forecast_days
            ]
        )

    return (
        np.array(X),
        np.array(y)
    )


X_train, y_train = create_sequences(
    train_scaled,
    WINDOW_SIZE,
    FORECAST_DAYS
)


# Include previous 14 days as context
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
        shape=(
            WINDOW_SIZE,
            num_categories
        )
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

    # 7 days × number of categories
    tf.keras.layers.Dense(
        FORECAST_DAYS * num_categories
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
    y_train.reshape(
        -1,
        FORECAST_DAYS * num_categories
    ),
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)


# ============================================================
# PREDICT
# ============================================================

y_pred_scaled = model.predict(
    X_test,
    verbose=0
)


# Reshape prediction
y_pred_scaled = y_pred_scaled.reshape(
    -1,
    FORECAST_DAYS,
    num_categories
)


# ============================================================
# INVERSE SCALING
# ============================================================

y_test_original = scaler.inverse_transform(
    y_test.reshape(
        -1,
        num_categories
    )
).reshape(
    -1,
    FORECAST_DAYS,
    num_categories
)


y_pred_original = scaler.inverse_transform(
    y_pred_scaled.reshape(
        -1,
        num_categories
    )
).reshape(
    -1,
    FORECAST_DAYS,
    num_categories
)


# ============================================================
# OVERALL METRICS
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


print("\n" + "=" * 60)
print("CATEGORY-WISE CNN RESULTS")
print("=" * 60)

print(f"Overall MAE  : {mae:.2f}")
print(f"Overall RMSE : {rmse:.2f}")


# ============================================================
# CATEGORY-WISE MAE
# ============================================================

print("\nCategory-wise MAE:")

for i, category in enumerate(categories):

    category_mae = mean_absolute_error(
        y_test_original[:, :, i].flatten(),
        y_pred_original[:, :, i].flatten()
    )

    print(
        f"{category:20s}: {category_mae:.2f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "cnn_category_spending.keras"
)

model.save(model_path)

print("\nModel saved to:")
print(model_path)


# ============================================================
# SAMPLE PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE 7-DAY CATEGORY PREDICTION")
print("=" * 60)

for day in range(FORECAST_DAYS):

    print(
        f"\nDay {day + 1}"
    )

    for i, category in enumerate(categories):

       print(
    f"{category:20s}: "
    f"Rs.{y_pred_original[0, day, i]:.2f}"
)


print("\nTraining completed successfully.")