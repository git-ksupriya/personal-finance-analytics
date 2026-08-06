import pandas as pd

def inspect_unique_values(df):
    columns = [
        "category",
        "payment_mode",
        "transaction_type",
        "location"
    ]

    for col in columns:
        print(f"\n{col.upper()}")
        print(df[col].unique())
        print(f"Unique values: {df[col].nunique(dropna=False)}")



def main():
    df = pd.read_csv("processed/cleaned_dataset.csv")
    inspect_unique_values(df)


if __name__ == "__main__":
    main()