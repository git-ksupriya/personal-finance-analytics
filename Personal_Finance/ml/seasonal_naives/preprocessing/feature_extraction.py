from pathlib import Path

from ml.lstm_categories.preprocessing.feature_extraction import (
    add_calendar_features,
    aggregate_daily_category_spending,
    create_complete_daily_series,
    load_dataset,
    prepare_expense_data,
)


PROJECT_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "preprocessed"
INPUT_PATH = (
    PROJECT_DIR
    / "preprocessing"
    / "processed"
    / "standardised_dataset.csv"
)
OUTPUT_PATH = OUTPUT_DIR / "daily_category_spending.csv"


def create_daily_category_data(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    transactions = load_dataset(input_path)
    expenses = prepare_expense_data(transactions)

    daily = aggregate_daily_category_spending(expenses)
    daily = create_complete_daily_series(daily)
    daily = add_calendar_features(daily)
    daily = daily.sort_values(
        ["user_id", "category", "date"]
    ).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_path, index=False)

    print(f"Saved daily category series: {output_path}")
    print(f"Rows: {len(daily):,}")
    print(f"Users: {daily['user_id'].nunique():,}")
    print(f"Categories: {daily['category'].nunique():,}")
    return daily


if __name__ == "__main__":
    create_daily_category_data()
