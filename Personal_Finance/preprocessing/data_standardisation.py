'''
Standardize date formats
Standardize category names
Standardize payment modes
Standardize locations
Standardize transaction types
'''

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


def standardise_category(df):
    CATEGORY_MAPPING = {
    "FOOD": "Food",
    "food": "Food",
    "Foods": "Food",
    "Foodd": "Food",
    "Fod": "Food",

    "rent": "Rent",
    "RENT": "Rent",
    "Rentt": "Rent",
    "Rnt": "Rent",

    "education": "Education",
    "Educaton": "Education",
    "EDU": "Education",

    "Utilties":"Utilities",
    "Utility":"Utilities",
    "Utlities":"Utilities",
    "utilities":"Utilities",

    "entertainment":"Entertainment",
    "Entertain":"Entertainment",
    "Entrtnmnt":"Entertainment",

    "HEALTH":"Health",
    "Helth":"Health",
    "health":"Health",

    "Travl":"Travel",
    "Traval":"Travel",
    "TRAVEL":"Travel",
    "travel":"Travel",

    "Saving":"Savings",
    "savings":"Savings",
    "SAVINGS":"Savings",

    "Other":"Others",
    "OTHERS":"Others",
    "others":"Others",
    "Misc":"Others"

    }

    df["category"] = df["category"].replace(CATEGORY_MAPPING)
    return df



def main():
    df = pd.read_csv("processed/cleaned_dataset.csv")

    # print(df["category"].isna().sum())  said 259 nan vals
    df["category"] = df["category"].fillna("Others")
    df = standardise_category(df)



    inspect_unique_values(df)


if __name__ == "__main__":
    main()