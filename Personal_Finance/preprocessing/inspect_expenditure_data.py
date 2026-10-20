import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(
    base_dir,
    "raw",
    "budgetwise_finance_dataset.csv"
)

df = pd.read_csv(file_path)

print("\n--- Dataset Information ---")
print("Shape:", df.shape)

print("\n--- Column Names ---")
print(df.columns.tolist())

print("\n--- First 10 Rows ---")
print(df.head(10).to_string())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- User Statistics ---")
print("Number of users:", df["user_id"].nunique())

print("\n--- Transactions Per User ---")
print(df.groupby("user_id").size().describe())

print("\n--- Date Range ---")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("Earliest date:", df["date"].min())
print("Latest date:", df["date"].max())

print("\n--- Transactions Per User Per Day ---")
daily_counts = df.groupby(
    ["user_id", df["date"].dt.date]
).size()

print(daily_counts.describe())