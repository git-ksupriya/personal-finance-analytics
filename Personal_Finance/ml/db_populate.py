import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

def populate_database(db_url=None):
    ml_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.abspath(os.path.join(ml_dir, "..", "preprocessing", "processed"))
    dataset_path = os.path.join(processed_dir, "feature_engineered_dataset.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Processed dataset not found at {dataset_path}. Run preprocessing first.")

    print(f"--- Loading Processed Dataset from {dataset_path} ---")
    df = pd.read_csv(dataset_path)

    # Use PostgreSQL if DB_URL provided or available in env, otherwise SQLite file
    db_env_url = db_url or os.getenv("DATABASE_URL") or os.getenv("DB_URL")
    
    use_sqlite = False
    if db_env_url and db_env_url.startswith("postgresql"):
        try:
            import sqlalchemy
            engine = sqlalchemy.create_engine(db_env_url)
            conn = engine.connect()
            print("--- Connected to PostgreSQL Database ---")
        except Exception as e:
            print(f"PostgreSQL Connection Error ({e}). Falling back to SQLite local database.")
            use_sqlite = True
    else:
        use_sqlite = True

    if use_sqlite:
        sqlite_db_path = os.path.join(ml_dir, "personal_finance.db")
        conn = sqlite3.connect(sqlite_db_path)
        print(f"--- Connected to SQLite Database at {sqlite_db_path} ---")

    cursor = conn.cursor() if use_sqlite else None

    # 1. Create Tables Schema (if not exists)
    if use_sqlite:
        cursor.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            phone TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS payment_modes (
            payment_mode_id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode_name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS locations (
            location_id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_transaction_id TEXT,
            user_id TEXT NOT NULL,
            category_id INTEGER NOT NULL,
            payment_mode_id INTEGER,
            location_id INTEGER,
            amount REAL NOT NULL,
            transaction_type TEXT NOT NULL,
            transaction_date TEXT NOT NULL,
            description TEXT,
            source TEXT DEFAULT 'csv',
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (category_id) REFERENCES categories(category_id),
            FOREIGN KEY (payment_mode_id) REFERENCES payment_modes(payment_mode_id),
            FOREIGN KEY (location_id) REFERENCES locations(location_id)
        );

        CREATE TABLE IF NOT EXISTS monthly_summary (
            user_id TEXT,
            month TEXT NOT NULL,
            total_income REAL DEFAULT 0,
            total_expense REAL DEFAULT 0,
            savings_rate REAL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, month),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        CREATE TABLE IF NOT EXISTS category_monthly_summary (
            user_id TEXT,
            category_id INTEGER,
            month TEXT NOT NULL,
            total_spent REAL DEFAULT 0,
            txn_count INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, category_id, month),
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        );

        CREATE TABLE IF NOT EXISTS model_registry (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            blob_storage_path TEXT,
            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metrics_json TEXT
        );

        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            model_id INTEGER NOT NULL,
            prediction_type TEXT NOT NULL,
            category_id INTEGER,
            target_period TEXT NOT NULL,
            predicted_value REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
        );
        """)
        conn.commit()

    # 2. Populate Lookups: Categories, Payment Modes, Locations
    print("--- Populating Lookup Tables ---")
    unique_categories = sorted(df["category"].dropna().unique())
    unique_payment_modes = sorted(df["payment_mode"].dropna().unique())
    unique_locations = sorted(df["location"].dropna().unique())

    if use_sqlite:
        for cat in unique_categories:
            cursor.execute("INSERT OR IGNORE INTO categories (category_name) VALUES (?)", (cat,))
        for pm in unique_payment_modes:
            cursor.execute("INSERT OR IGNORE INTO payment_modes (mode_name) VALUES (?)", (pm,))
        for loc in unique_locations:
            cursor.execute("INSERT OR IGNORE INTO locations (location_name) VALUES (?)", (loc,))
        conn.commit()

        # Build ID lookups
        cat_map = dict(cursor.execute("SELECT category_name, category_id FROM categories").fetchall())
        pm_map = dict(cursor.execute("SELECT mode_name, payment_mode_id FROM payment_modes").fetchall())
        loc_map = dict(cursor.execute("SELECT location_name, location_id FROM locations").fetchall())

    # 3. Populate Users Table
    print("--- Populating Users Table ---")
    unique_users = sorted(df["user_id"].unique())
    user_rows = []
    for uid in unique_users:
        email = f"{uid.lower()}@example.com"
        name = f"User {uid}"
        pwd = f"hash_{uid}"
        user_rows.append((uid, name, email, pwd, "9999999999"))

    if use_sqlite:
        cursor.executemany("""
        INSERT OR IGNORE INTO users (user_id, name, email, password_hash, phone)
        VALUES (?, ?, ?, ?, ?)
        """, user_rows)
        conn.commit()

    # 4. Populate Transactions Table
    print(f"--- Populating Transactions Table ({len(df)} rows) ---")
    txn_rows = []
    for idx, row in df.iterrows():
        txn_id = row.get("transaction_id")
        uid = row["user_id"]
        cat_id = cat_map.get(row.get("category"), cat_map.get("Others", 1))
        pm_id = pm_map.get(row.get("payment_mode"), pm_map.get("Unknown", 1))
        loc_id = loc_map.get(row.get("location"), loc_map.get("Unknown", 1))
        amount = float(row["amount"])
        ttype = str(row.get("transaction_type", "Expense")).strip()
        tdate = str(row["date"])[:10]
        notes = str(row.get("notes", ""))
        txn_rows.append((txn_id, uid, cat_id, pm_id, loc_id, amount, ttype, tdate, notes, 'csv'))

    if use_sqlite:
        cursor.executemany("""
        INSERT INTO transactions (source_transaction_id, user_id, category_id, payment_mode_id, location_id, amount, transaction_type, transaction_date, description, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, txn_rows)
        conn.commit()

    # 5. Populate Monthly Summaries
    print("--- Populating Monthly Summary & Category Summaries ---")
    df["year_month_first"] = df["year_month"].astype(str) + "-01"
    
    monthly_grp = df.groupby(["user_id", "year_month_first"])
    summary_rows = []
    for (uid, ym), group in monthly_grp:
        inc = float(group[group["transaction_type"] == "Income"]["amount"].sum())
        exp = float(group[group["transaction_type"] == "Expense"]["amount"].sum())
        srate = round(((inc - exp) / inc * 100), 2) if inc > 0 else 0.0
        summary_rows.append((uid, ym, inc, exp, srate))

    if use_sqlite:
        cursor.executemany("""
        INSERT OR REPLACE INTO monthly_summary (user_id, month, total_income, total_expense, savings_rate)
        VALUES (?, ?, ?, ?, ?)
        """, summary_rows)

    cat_grp = df[df["transaction_type"] == "Expense"].groupby(["user_id", "category", "year_month_first"])
    cat_summary_rows = []
    for (uid, cat_name, ym), group in cat_grp:
        c_id = cat_map.get(cat_name, 1)
        tot = float(group["amount"].sum())
        cnt = int(len(group))
        cat_summary_rows.append((uid, c_id, ym, tot, cnt))

    if use_sqlite:
        cursor.executemany("""
        INSERT OR REPLACE INTO category_monthly_summary (user_id, category_id, month, total_spent, txn_count)
        VALUES (?, ?, ?, ?, ?)
        """, cat_summary_rows)
        conn.commit()

    # 6. Populate Model Registry
    print("--- Populating Model Registry Table ---")
    lstm_metrics_file = os.path.join(ml_dir, "lstm", "lstm_metrics.json")
    rnn_metrics_file = os.path.join(ml_dir, "rnn", "rnn_metrics.json")

    lstm_metrics = json.load(open(lstm_metrics_file, encoding="utf-8")) if os.path.exists(lstm_metrics_file) else {"raw_overall_rmse": 0.0}
    rnn_metrics = json.load(open(rnn_metrics_file, encoding="utf-8")) if os.path.exists(rnn_metrics_file) else {"raw_overall_rmse": 0.0}

    model_registry_rows = [
        ("Multi-Step 7-Day LSTM Spending Predictor", "LSTM", os.path.join(ml_dir, "lstm", "lstm_model.pt"), json.dumps(lstm_metrics)),
        ("Multi-Step 7-Day SimpleRNN Spending Predictor", "RNN", os.path.join(ml_dir, "rnn", "rnn_model.pt"), json.dumps(rnn_metrics))
    ]

    if use_sqlite:
        cursor.executemany("""
        INSERT INTO model_registry (model_name, model_type, blob_storage_path, metrics_json)
        VALUES (?, ?, ?, ?)
        """, model_registry_rows)
        conn.commit()

        lstm_model_id = cursor.execute("SELECT model_id FROM model_registry WHERE model_type = 'LSTM' ORDER BY model_id DESC LIMIT 1").fetchone()[0]
    else:
        lstm_model_id = 1

    # 7. Generate & Populate Predictions Table
    print("--- Populating Predictions Table (7-Day Forecasts) ---")
    try:
        from predict import SpendingPredictor
        predictor = SpendingPredictor(model_type="lstm")
        
        prediction_rows = []
        user_grouped = df.sort_values(["user_id", "date"]).groupby("user_id")
        for uid, user_data in user_grouped:
            amounts = user_data["amount"].values
            if len(amounts) >= 10:
                past_seq = amounts[-10:].tolist()
                preds_dict = predictor.predict_next_7_days(past_seq)
                for day_label, pred_val in preds_dict.items():
                    prediction_rows.append((uid, lstm_model_id, "next_7_days_spend", None, day_label, pred_val))

        if use_sqlite:
            cursor.executemany("""
            INSERT INTO predictions (user_id, model_id, prediction_type, category_id, target_period, predicted_value)
            VALUES (?, ?, ?, ?, ?, ?)
            """, prediction_rows)
            conn.commit()
            print(f"Generated {len(prediction_rows)} 7-day spending predictions across users.")
    except Exception as e:
        print(f"Prediction Generation Note ({e})")

    # 8. Print DB Table Summary
    print("\n=======================================================")
    print("DATABASE POPULATION COMPLETE SUMMARY")
    print("=======================================================")
    if use_sqlite:
        tables = ["users", "categories", "payment_modes", "locations", "transactions", "monthly_summary", "category_monthly_summary", "model_registry", "predictions"]
        for tbl in tables:
            cnt = cursor.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            print(f"Table '{tbl:25s}': {cnt:6d} rows")
        conn.close()

if __name__ == "__main__":
    populate_database()
