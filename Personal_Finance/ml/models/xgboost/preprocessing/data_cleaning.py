
import pandas as pd
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"

PROJECT_ROOT = Path(__file__).resolve().parents[4]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "preprocessing"
    / "raw"
    / "budgetwise_finance_dataset.csv"
)

# Values that look like artificial placeholder amounts.
# Exclude these from modeling until the dataset source confirms otherwise.
PLACEHOLDER_AMOUNTS = {999999, 999999999}


def parse_amount(value):
    """Convert common currency-formatted values into numeric amounts."""
    if pd.isna(value):
        return float("nan")

    value = str(value).strip()

    if not value or value.lower() in {"nan", "none", "null", "n/a"}:
        return float("nan")

    # Remove thousands separators.
    value = value.replace(",", "")

    # Remove common currency symbols and labels.
    value = re.sub(r"(?i)INR|Rs\.?|₹|\$", "", value)
    value = value.strip()

    # Convert only if the remaining value is numeric.
    try:
        return float(value)
    except ValueError:
        return float("nan")


def standardize_category(value):
    """Standardize common spelling and capitalization variations."""
    if pd.isna(value) or not str(value).strip():
        return "Unknown"

    original = str(value).strip()
    key = re.sub(r"[^a-z]", "", original.lower())

    category_map = {
        "food": "Food",
        "foods": "Food",
        "foodd": "Food",
        "fod": "Food",

        "rent": "Rent",
        "rentt": "Rent",
        "rnt": "Rent",

        "travel": "Travel",
        "traval": "Travel",
        "travl": "Travel",

        "entertainment": "Entertainment",
        "entertain": "Entertainment",
        "entertainmentt": "Entertainment",
        "entrtnmnt": "Entertainment",

        "utility": "Utilities",
        "utilities": "Utilities",
        "utilties": "Utilities",
        "utlities": "Utilities",

        "education": "Education",
        "educaton": "Education",
        "edu": "Education",

        "health": "Health",
        "helth": "Health",

        "saving": "Savings",
        "savings": "Savings",

        "other": "Other",
        "others": "Other",
        "misc": "Other",
    }

    return category_map.get(key, original.title())


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Original dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)
    df.columns = df.columns.str.strip().str.lower()

    required = ["user_id", "date", "amount"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    original_rows = len(df)

    # Parse dates and amounts.
    df["date"] = pd.to_datetime(
        df["date"], errors="coerce", format="mixed", dayfirst=False
    )
    df["amount"] = df["amount"].apply(parse_amount)

    missing_amounts = df["amount"].isna().sum()
    invalid_dates = df["date"].isna().sum()
    # Remove exact duplicate records after amount normalization.
    before_dedup = len(df)
    df = df.drop_duplicates().copy()
    exact_duplicates_removed = before_dedup - len(df)

    # Missing amounts and dates cannot be used for time-series modeling.
    df = df.dropna(subset=["user_id", "date", "amount"]).copy()
    # Exclude suspicious placeholder values from modeling.
    placeholder_mask = df["amount"].abs().isin(PLACEHOLDER_AMOUNTS)
    placeholder_count = int(placeholder_mask.sum())
    df = df.loc[~placeholder_mask].copy()

    df["user_id"] = df["user_id"].astype(str).str.strip()

    if "transaction_id" not in df.columns:
        df["transaction_id"] = [
            f"ROW_{i}" for i in range(len(df))
        ]

    # Normalize common text columns.
    for col in [
        "transaction_type",
        "category",
        "payment_mode",
        "location",
    ]:
        if col not in df.columns:
            df[col] = "Unknown"

        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

    df["transaction_type"] = df["transaction_type"].str.title()

    df["category"] = df["category"].apply(standardize_category)

    if "notes" in df.columns:
        df["notes"] = (
            df["notes"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # Do not remove repeated transaction IDs automatically.
    # They require a separate investigation.
    df = df.sort_values(
        ["user_id", "date", "transaction_id"],
        kind="stable",
    )

    output_path = PROCESSED_DIR / "cleaned_transactions.csv"
    df.to_csv(output_path, index=False)

    print("Cleaning completed.")
    print("Original rows:", original_rows)
    print("Missing/unparseable amounts:", missing_amounts)
    print("Invalid dates:", invalid_dates)
    print("Placeholder rows excluded:", placeholder_count)
    print("Exact duplicate rows removed:", exact_duplicates_removed)
    print("Final dataset shape:", df.shape)
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
