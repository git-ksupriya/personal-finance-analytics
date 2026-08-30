import pandas as pd
from pathlib import Path


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Current directory:
# cnn/preprocessing/

CURRENT_DIR = Path(__file__).resolve().parent

# CNN preprocessing directory
PROCESSED_DIR = CURRENT_DIR.parent / "processed"

# Make sure processed directory exists
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CATEGORY MAPPING
# ============================================================

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


# ============================================================
# LOCATION MAPPING
# ============================================================

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


# ============================================================
# PAYMENT MODE MAPPING
# ============================================================

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


# ============================================================
# STANDARDISE CATEGORY
# ============================================================

def standardise_category(df):

    df["category"] = df["category"].fillna("Others")

    df["category"] = df["category"].replace(
        CATEGORY_MAPPING
    )

    return df


# ============================================================
# STANDARDISE LOCATIONS
# ============================================================

def standardise_locations(df):

    df["location"] = df["location"].fillna("Unknown")

    df["location"] = df["location"].replace(
        LOCATION_MAPPING
    )

    return df


# ============================================================
# STANDARDISE PAYMENT MODES
# ============================================================

def standardise_payment_modes(df):

    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    df["payment_mode"] = df["payment_mode"].replace(
        PAYMENT_MAPPING
    )

    return df


# ============================================================
# STANDARDISE DATES
# ============================================================

def standardise_dates(df):

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["date"] = df["date"].dt.normalize()

    return df


# ============================================================
# STANDARDISE TRANSACTION TYPE
# ============================================================

def standardise_transaction_type(df):

    df["transaction_type"] = (
        df["transaction_type"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CNN DATA STANDARDISATION")
    print("=" * 60)

    # Input from CNN preprocessing folder
    input_path = PROCESSED_DIR / "cleaned_dataset.csv"

    print("\nReading:")
    print(input_path)

    df = pd.read_csv(input_path)

    print(f"\nRows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    # Standardisation
    df = standardise_category(df)

    df = standardise_locations(df)

    df = standardise_payment_modes(df)

    df = standardise_dates(df)

    df = standardise_transaction_type(df)

    # Output
    output_path = PROCESSED_DIR / "standardised_dataset.csv"

    df.to_csv(
        output_path,
        index=False
    )

    print("\nStandardised dataset saved to:")
    print(output_path)


if __name__ == "__main__":
    main()