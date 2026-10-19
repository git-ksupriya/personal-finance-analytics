import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
MODEL_PATH = BASE_DIR.parent / "xgboost_model.json"


def main():
    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))

    X_test = pd.read_csv(
        PROCESSED_DIR / "X_test.csv"
    )

    predictions = model.predict(X_test)

    output = pd.DataFrame({
        "predicted_next_transaction_amount": predictions
    })

    output_path = PROCESSED_DIR / "new_predictions.csv"
    output.to_csv(output_path, index=False)

    print(output.head(20).to_string(index=False))
    print("\nPredictions saved to:", output_path)


if __name__ == "__main__":
    main()