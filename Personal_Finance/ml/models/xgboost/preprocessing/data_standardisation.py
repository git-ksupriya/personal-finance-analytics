import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"


def main():
    input_path = PROCESSED_DIR / "cleaned_transactions.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            "Run data_cleaning.py first."
        )

    df = pd.read_csv(input_path, parse_dates=["date"])

    df["amount"] = pd.to_numeric(
        df["amount"], errors="coerce"
    )

    df = df.dropna(
        subset=["user_id", "date", "amount"]
    ).copy()

    for col in [
        "transaction_type",
        "category",
        "payment_mode",
        "location"
    ]:
        df[col] = df[col].fillna("Unknown").astype(str)

    df = df.sort_values(
        ["user_id", "date", "transaction_id"],
        kind="stable"
    )

    output_path = PROCESSED_DIR / "standardized_transactions.csv"
    df.to_csv(output_path, index=False)

    print("Data preparation completed.")
    print("Shape:", df.shape)
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()