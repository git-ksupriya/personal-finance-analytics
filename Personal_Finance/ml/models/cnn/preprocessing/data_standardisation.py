import pandas as pd


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

    "Utilties": "Utilities",
    "Utility": "Utilities",
    "Utlities": "Utilities",
    "utilities": "Utilities",

    "entertainment": "Entertainment",
    "Entertain": "Entertainment",
    "Entrtnmnt": "Entertainment",

    "HEALTH": "Health",
    "Helth": "Health",
    "health": "Health",

    "Travl": "Travel",
    "Traval": "Travel",
    "TRAVEL": "Travel",
    "travel": "Travel",

    "Saving": "Savings",
    "savings": "Savings",
    "SAVINGS": "Savings",

    "Other": "Others",
    "OTHERS": "Others",
    "others": "Others",
    "Misc": "Others"
}


LOCATION_MAPPING = {

    "BAN": "Bangalore",
    "BANGALORE": "Bangalore",
    "bangalore": "Bangalore",

    "CHENNAI": "Chennai",
    "CHE": "Chennai",
    "chennai": "Chennai",

    "PUNE": "Pune",
    "pune": "Pune",
    "PUN": "Pune",

    "LUC": "Lucknow",
    "lucknow": "Lucknow",
    "LUCKNOW": "Lucknow",

    "hyderabad": "Hyderabad",
    "HYDERABAD": "Hyderabad",
    "HYD": "Hyderabad",

    "JAI": "Jaipur",
    "JAIPUR": "Jaipur",
    "jaipur": "Jaipur",

    "KOL": "Kolkata",
    "KOLKATA": "Kolkata",
    "kolkata": "Kolkata",

    "AHM": "Ahmedabad",
    "AHMEDABAD": "Ahmedabad",
    "ahmedabad": "Ahmedabad",

    "DELHI": "Delhi",
    "DEL": "Delhi",
    "delhi": "Delhi",

    "MUM": "Mumbai",
    "mumbai": "Mumbai",
    "MUMBAI": "Mumbai"
}


PAYMENT_MAPPING = {

    "card": "Card",
    "CRD": "Card",
    "Crd": "Card",
    "CARD": "Card",

    "Csh": "Cash",
    "csh": "Cash",
    "cash": "Cash",
    "CASH": "Cash",

    "upi": "Upi",
    "UPI": "Upi",
    "UPi": "Upi",

    "Bank Transfer": "Bank_Transfer",
    "bank transfer": "Bank_Transfer",
    "BankTransfer": "Bank_Transfer",
    "Bank Transfr": "Bank_Transfer"
}


def standardise_category(df):

    df["category"] = df["category"].fillna("Others")

    df["category"] = df["category"].replace(
        CATEGORY_MAPPING
    )

    return df


def standardise_locations(df):

    df["location"] = df["location"].fillna("Unknown")

    df["location"] = df["location"].replace(
        LOCATION_MAPPING
    )

    return df


def standardise_payment_modes(df):

    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    df["payment_mode"] = df["payment_mode"].replace(
        PAYMENT_MAPPING
    )

    return df


def standardise_dates(df):

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["date"] = df["date"].dt.normalize()

    return df


def standardise_transaction_type(df):

    df["transaction_type"] = (
        df["transaction_type"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    return df


def main():

    df = pd.read_csv(
        "processed/cleaned_dataset.csv"
    )

    df = standardise_category(df)

    df = standardise_locations(df)

    df = standardise_payment_modes(df)

    df = standardise_dates(df)

    df = standardise_transaction_type(df)

    df.to_csv(
        "processed/standardised_dataset.csv",
        index=False
    )

    print("Standardised dataset saved.")


if __name__ == "__main__":
    main()