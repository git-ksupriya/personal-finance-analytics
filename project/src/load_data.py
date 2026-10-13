import io
import time
import pandas as pd
import psycopg2
from config import DATA_DIR, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB

def get_connection():
    return psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT
    )

def copy_from_df(conn, df: pd.DataFrame, table_name: str, columns: list):
    """Fast copy of a pandas DataFrame to PostgreSQL table using in-memory CSV and COPY."""
    output = io.StringIO()
    df.to_csv(output, sep='\t', header=False, index=False, na_rep='\\N')
    output.seek(0)
    with conn.cursor() as cur:
        cur.copy_from(output, table_name, null='\\N', columns=columns)
    conn.commit()

def load_data():
    start_time = time.time()
    print("Reading fraudTrain.csv and fraudTest.csv...")
    train_path = DATA_DIR / "fraudTrain.csv"
    test_path = DATA_DIR / "fraudTest.csv"

    cols_to_use = [
        'trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt',
        'first', 'last', 'gender', 'street', 'city', 'state', 'zip',
        'lat', 'long', 'city_pop', 'job', 'dob', 'trans_num', 'unix_time',
        'merch_lat', 'merch_long', 'is_fraud'
    ]

    df_train = pd.read_csv(train_path, usecols=cols_to_use)
    df_test = pd.read_csv(test_path, usecols=cols_to_use)
    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"Total raw records loaded: {len(df):,}")

    # Strip 'fraud_' prefix from merchant names
    df['merchant'] = df['merchant'].astype(str).str.replace(r'^fraud_', '', regex=True)

    # Standardize data types
    df['is_fraud'] = df['is_fraud'].astype(int) == 1
    df['amt'] = df['amt'].round(2)
    df['trans_ts'] = pd.to_datetime(df['trans_date_trans_time'])

    conn = get_connection()
    print("Connected to PostgreSQL successfully.")

    # 1. Deduplicate and load users
    print("Extracting and inserting unique users...")
    user_cols = ['cc_num', 'first', 'last', 'gender', 'dob', 'job', 'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop']
    users_df = df[user_cols].drop_duplicates(subset=['cc_num']).copy()
    users_df['dob'] = pd.to_datetime(users_df['dob']).dt.date
    copy_from_df(conn, users_df, 'users', user_cols)
    print(f"Loaded {len(users_df):,} users.")

    # Fetch surrogate user_id mapping
    with conn.cursor() as cur:
        cur.execute("SELECT user_id, cc_num FROM users;")
        user_map = {row[1]: row[0] for row in cur.fetchall()}

    # 2. Deduplicate and load merchants
    print("Extracting and inserting unique merchants...")
    merch_cols = ['merchant', 'category', 'merch_lat', 'merch_long']
    merchants_df = df[merch_cols].drop_duplicates(subset=['merchant']).copy()
    merchants_df.rename(columns={'merchant': 'merchant_name'}, inplace=True)
    copy_from_df(conn, merchants_df, 'merchants', ['merchant_name', 'category', 'merch_lat', 'merch_long'])
    print(f"Loaded {len(merchants_df):,} merchants.")

    # Fetch merchant_id mapping
    with conn.cursor() as cur:
        cur.execute("SELECT merchant_id, merchant_name FROM merchants;")
        merch_map = {row[1]: row[0] for row in cur.fetchall()}

    # 3. Prepare and load transactions
    print("Mapping foreign keys and loading transactions with COPY...")
    df['user_id'] = df['cc_num'].map(user_map)
    df['merchant_id'] = df['merchant'].map(merch_map)

    # Deduplicate transactions if any duplicated trans_num
    df_txns = df.drop_duplicates(subset=['trans_num']).copy()
    txn_cols = ['trans_num', 'user_id', 'merchant_id', 'category', 'amt', 'trans_ts', 'is_fraud']
    copy_from_df(conn, df_txns[txn_cols], 'transactions', txn_cols)
    print(f"Loaded {len(df_txns):,} transactions.")

    # 4. Run weekly aggregations SQL
    print("Running weekly zero-filled aggregations...")
    agg_sql_path = DATA_DIR.parent / "personal-finance-analytics - Copy" / "project" / "sql" / "aggregations.sql"
    if not agg_sql_path.exists():
        agg_sql_path = Path(__file__).resolve().parent.parent / "sql" / "aggregations.sql"
    
    with open(agg_sql_path, 'r') as f:
        agg_sql = f.read()

    with conn.cursor() as cur:
        cur.execute(agg_sql)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM weekly_category_spend;")
        total_weekly = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM weekly_category_spend WHERE total_amt > 0;")
        nonzero_weekly = cur.fetchone()[0]

    print(f"Weekly spend records: {total_weekly:,} (Non-zero weeks: {nonzero_weekly:,}, Zero rate: {(1 - nonzero_weekly/total_weekly)*100:.2f}%)")
    print(f"Data loading complete in {time.time() - start_time:.2f} seconds.")
    conn.close()

if __name__ == "__main__":
    load_data()
