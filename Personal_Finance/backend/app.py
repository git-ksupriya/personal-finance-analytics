import os
import sys
import io
import csv
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

import psycopg2
from psycopg2.extras import RealDictCursor
from passlib.context import CryptContext

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel

# =========================================================
# PATHS & CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

sys.path.insert(0, str(BASE_DIR))
from inference import InferenceEngine

import bcrypt

DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-please-change-in-production")

def hash_password(password: str) -> str:
    pwd_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        pwd_bytes = plain_password.encode("utf-8")[:72]
        hash_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(pwd_bytes, hash_bytes)
    except Exception:
        return False

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Personal Finance Analytics",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    session_cookie="session",
    max_age=86400 * 7,  # 7 days
    same_site="lax"
)

inference_engine = InferenceEngine.get_instance()

# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url)
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "personal_finance"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "Shiva@138")
    )


# =========================================================
# AUTH UTILITIES
# =========================================================

def get_current_user_id(request: Request) -> str:
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_id


def generate_next_user_id(cursor) -> str:
    cursor.execute("""
        SELECT user_id FROM users
        WHERE user_id ~ '^U[0-9]+$'
        ORDER BY CAST(SUBSTRING(user_id FROM 2) AS INTEGER) DESC
        LIMIT 1;
    """)
    row = cursor.fetchone()
    if row:
        last_id = row["user_id"] if isinstance(row, dict) else row[0]
        num = int(last_id[1:]) + 1
    else:
        num = 1
    return f"U{num:03d}" if num < 100 else f"U{num}"


# =========================================================
# PYDANTIC SCHEMAS
# =========================================================

class SignupRequest(BaseModel):
    username: str
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class BudgetSetting(BaseModel):
    category_id: Optional[int] = None
    category_name: Optional[str] = None
    weekly_amount: float

class OnboardingBudgetsRequest(BaseModel):
    weekly_overall: float
    category_budgets: List[BudgetSetting] = []

class TransactionCreateRequest(BaseModel):
    date: Optional[str] = None
    amount: float
    category: str
    payment_mode: str
    location: Optional[str] = "Unknown"
    description: Optional[str] = ""


# =========================================================
# BASIC & STATIC FILE ROUTES
# =========================================================

@app.get("/")
def home(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url="/pages/login.html")
    
    # Check if onboarded
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT onboarded FROM users WHERE user_id = %s;", (user_id,))
            user = cur.fetchone()
        conn.close()
        if user and user.get("onboarded"):
            return RedirectResponse(url="/pages/analytics.html")
        return RedirectResponse(url="/pages/onboarding.html")
    except Exception:
        return RedirectResponse(url="/pages/login.html")


@app.get("/pages/{page_name}")
def serve_page(page_name: str, request: Request):
    allowed_pages = {
        "login.html",
        "onboarding.html",
        "analytics.html",
        "categories.html",
        "category-summary.html",
        "highest-category.html",
        "top-transactions.html",
        "transaction-details.html",
        "user-spending.html",
    }

    if page_name not in allowed_pages:
        raise HTTPException(status_code=404, detail="Page not found")

    # Guard protected pages
    if page_name in {"onboarding.html", "analytics.html"}:
        user_id = request.session.get("user_id")
        if not user_id:
            return RedirectResponse(url="/pages/login.html")

    return FileResponse(FRONTEND_DIR / "pages" / page_name)


@app.get("/style.css")
def style():
    return FileResponse(FRONTEND_DIR / "style.css")


@app.get("/script.js")
def script():
    return FileResponse(FRONTEND_DIR / "script.js")


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/api/health")
def health():
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT current_database();")
            database = cursor.fetchone()[0]
        conn.close()
        return {"status": "connected", "database": database}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# =========================================================
# AUTHENTICATION ENDPOINTS
# =========================================================

@app.post("/api/signup")
def signup(payload: SignupRequest, request: Request):
    username = payload.username.strip()
    email = payload.email.strip().lower()
    name = payload.name.strip()
    raw_password = payload.password.strip()

    if not username or not email or not raw_password:
        raise HTTPException(status_code=400, detail="Username, email, and password are required")

    hashed_pw = hash_password(raw_password)

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check existing username or email
            cur.execute("SELECT user_id FROM users WHERE username = %s OR email = %s;", (username, email))
            existing = cur.fetchone()
            if existing:
                raise HTTPException(status_code=400, detail="Username or email already registered")

            new_user_id = generate_next_user_id(cur)
            cur.execute(
                """
                INSERT INTO users (user_id, username, name, email, password_hash, onboarded)
                VALUES (%s, %s, %s, %s, %s, FALSE)
                RETURNING user_id, username, name, email, onboarded;
                """,
                (new_user_id, username, name, email, hashed_pw)
            )
            user = cur.fetchone()
        conn.commit()

        request.session["user_id"] = user["user_id"]
        return {
            "status": "success",
            "user": user,
            "redirect": "/pages/onboarding.html"
        }
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")
    finally:
        conn.close()


@app.post("/api/login")
def login(payload: LoginRequest, request: Request):
    identifier = payload.username.strip()
    password = payload.password.strip()

    if not identifier or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT user_id, username, name, email, password_hash, onboarded FROM users WHERE username = %s OR email = %s;",
                (identifier, identifier.lower())
            )
            user = cur.fetchone()

        if not user or not user["password_hash"]:
            raise HTTPException(status_code=401, detail="Invalid username or password")

        if not verify_password(password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid username or password")

        request.session["user_id"] = user["user_id"]

        redirect_url = "/pages/analytics.html" if user["onboarded"] else "/pages/onboarding.html"
        return {
            "status": "success",
            "user": {
                "user_id": user["user_id"],
                "username": user["username"],
                "name": user["name"],
                "email": user["email"],
                "onboarded": user["onboarded"]
            },
            "redirect": redirect_url
        }
    finally:
        conn.close()


@app.post("/api/logout")
def logout(request: Request):
    request.session.clear()
    return {"status": "success", "message": "Logged out"}


@app.get("/api/me")
def get_current_user(request: Request):
    user_id = get_current_user_id(request)
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT user_id, username, name, email, onboarded FROM users WHERE user_id = %s;",
                (user_id,)
            )
            user = cur.fetchone()
        if not user:
            request.session.clear()
            raise HTTPException(status_code=401, detail="User not found")
        return user
    finally:
        conn.close()


# =========================================================
# ONBOARDING ENDPOINTS
# =========================================================

@app.get("/api/lookups")
def get_lookups():
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT category_id, category_name FROM categories ORDER BY category_id;")
            cats = cur.fetchall()
            cur.execute("SELECT payment_mode_id, mode_name FROM payment_modes ORDER BY payment_mode_id;")
            pms = cur.fetchall()
        return {"categories": cats, "payment_modes": pms}
    finally:
        conn.close()


@app.post("/api/onboarding/budgets")
def set_onboarding_budgets(payload: OnboardingBudgetsRequest, request: Request):
    user_id = get_current_user_id(request)
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1. Upsert Overall Weekly Budget
            cur.execute(
                """
                INSERT INTO weekly_budgets (user_id, category_id, weekly_amount)
                VALUES (%s, NULL, %s)
                ON CONFLICT (user_id) WHERE category_id IS NULL
                DO UPDATE SET weekly_amount = EXCLUDED.weekly_amount, updated_at = CURRENT_TIMESTAMP;
                """,
                (user_id, payload.weekly_overall)
            )

            # 2. Upsert Category Budgets
            # Map category_name to id if provided
            cur.execute("SELECT category_id, category_name FROM categories;")
            cat_map = {r["category_name"].lower(): r["category_id"] for r in cur.fetchall()}

            for cb in payload.category_budgets:
                cid = cb.category_id
                if cid is None and cb.category_name:
                    cid = cat_map.get(cb.category_name.strip().lower())
                if cid is not None:
                    cur.execute(
                        """
                        INSERT INTO weekly_budgets (user_id, category_id, weekly_amount)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (user_id, category_id) WHERE category_id IS NOT NULL
                        DO UPDATE SET weekly_amount = EXCLUDED.weekly_amount, updated_at = CURRENT_TIMESTAMP;
                        """,
                        (user_id, cid, cb.weekly_amount)
                    )

            # 3. Mark user onboarded
            cur.execute("UPDATE users SET onboarded = TRUE WHERE user_id = %s;", (user_id,))
        conn.commit()

        # Run inference refresh
        inference_engine.refresh_predictions(conn, user_id)

        return {"status": "success", "message": "Budgets saved and user onboarded"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save budgets: {str(e)}")
    finally:
        conn.close()


@app.post("/api/onboarding/upload-csv")
async def upload_transactions_csv(request: Request, file: UploadFile = File(...)):
    user_id = get_current_user_id(request)
    contents = await file.read()
    
    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = contents.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV file is empty or missing headers")

    # Map headers case-insensitively
    field_map = {col.strip().lower(): col for col in reader.fieldnames}

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT category_id, category_name FROM categories;")
            cat_lookup = {r["category_name"].lower(): r["category_id"] for r in cur.fetchall()}
            default_cat_id = cat_lookup.get("others", 8)

            cur.execute("SELECT payment_mode_id, mode_name FROM payment_modes;")
            pm_lookup = {r["mode_name"].lower(): r["payment_mode_id"] for r in cur.fetchall()}
            default_pm_id = pm_lookup.get("cash", 1)

            cur.execute("SELECT location_id, location_name FROM locations;")
            loc_lookup = {r["location_name"].lower(): r["location_id"] for r in cur.fetchall()}
            default_loc_id = loc_lookup.get("unknown", 1)

            # Load existing transactions to detect duplicates
            cur.execute(
                """
                SELECT transaction_date, amount, category_id, transaction_type
                FROM transactions WHERE user_id = %s;
                """,
                (user_id,)
            )
            existing_txns = set(
                (str(r["transaction_date"]), float(r["amount"]), int(r["category_id"]), str(r["transaction_type"]))
                for r in cur.fetchall()
            )

            inserted = 0
            skipped = 0
            bad_rows = 0

            for row in reader:
                # Helper to get field case-insensitively
                def get_val(key):
                    for k in field_map:
                        if k == key.lower():
                            return row.get(field_map[k], "").strip()
                    return ""

                date_str = get_val("date")
                amount_str = get_val("amount")
                cat_str = get_val("category").lower()
                pm_str = get_val("payment_mode").lower()
                loc_str = get_val("location").lower()
                ttype_str = get_val("transaction_type") or "Expense"
                desc_str = get_val("description") or get_val("notes") or ""

                if not date_str or not amount_str:
                    bad_rows += 1
                    continue

                try:
                    # Parse and format date
                    dt = pd.to_datetime(date_str, errors="coerce")
                    if pd.isna(dt):
                        bad_rows += 1
                        continue
                    clean_date = dt.strftime("%Y-%m-%d")
                    clean_amount = round(abs(float(amount_str)), 2)
                    if clean_amount <= 0:
                        bad_rows += 1
                        continue
                except Exception:
                    bad_rows += 1
                    continue

                clean_ttype = "Income" if "income" in ttype_str.lower() else "Expense"
                cid = cat_lookup.get(cat_str, default_cat_id)
                pmid = pm_lookup.get(pm_str, default_pm_id)
                lid = loc_lookup.get(loc_str, default_loc_id)

                # Duplicate check
                txn_key = (clean_date, clean_amount, cid, clean_ttype)
                if txn_key in existing_txns:
                    skipped += 1
                    continue

                cur.execute(
                    """
                    INSERT INTO transactions (source_transaction_id, user_id, category_id, payment_mode_id, location_id, amount, transaction_type, transaction_date, description, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'csv');
                    """,
                    (None, user_id, cid, pmid, lid, clean_amount, clean_ttype, clean_date, desc_str)
                )
                existing_txns.add(txn_key)
                inserted += 1

        conn.commit()

        # Run inference refresh for user
        inference_engine.refresh_predictions(conn, user_id)

        return {
            "status": "success",
            "inserted": inserted,
            "skipped": skipped,
            "bad_rows": bad_rows
        }
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"CSV upload failed: {str(e)}")
    finally:
        conn.close()


# =========================================================
# ANALYTICS & LOGGING TRANSACTIONS
# =========================================================

@app.get("/api/analytics/weekly-spend")
def get_weekly_spend(request: Request):
    user_id = get_current_user_id(request)
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=6)

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1. Total actual spend in last 7 days
            cur.execute(
                """
                SELECT COALESCE(SUM(amount), 0) AS total_actual
                FROM transactions
                WHERE user_id = %s
                  AND transaction_type = 'Expense'
                  AND transaction_date BETWEEN %s AND %s;
                """,
                (user_id, start_date, today)
            )
            overall_actual = float(cur.fetchone()["total_actual"])

            # 2. Category actual spend in last 7 days
            cur.execute(
                """
                SELECT c.category_id, c.category_name, COALESCE(SUM(t.amount), 0) AS actual_amount
                FROM transactions t
                JOIN categories c ON t.category_id = c.category_id
                WHERE t.user_id = %s
                  AND t.transaction_type = 'Expense'
                  AND t.transaction_date BETWEEN %s AND %s
                GROUP BY c.category_id, c.category_name
                ORDER BY actual_amount DESC;
                """,
                (user_id, start_date, today)
            )
            cat_actuals = cur.fetchall()

            # 3. Weekly budgets
            cur.execute(
                """
                SELECT b.category_id, c.category_name, b.weekly_amount
                FROM weekly_budgets b
                LEFT JOIN categories c ON b.category_id = c.category_id
                WHERE b.user_id = %s;
                """,
                (user_id,)
            )
            budget_rows = cur.fetchall()

            overall_budget = None
            category_budgets = {}
            for b in budget_rows:
                if b["category_id"] is None:
                    overall_budget = float(b["weekly_amount"])
                else:
                    category_budgets[b["category_name"]] = float(b["weekly_amount"])

        return {
            "period": {
                "start_date": str(start_date),
                "end_date": str(today)
            },
            "overall": {
                "actual": round(overall_actual, 2),
                "budget": overall_budget
            },
            "categories": [
                {
                    "category_id": r["category_id"],
                    "category_name": r["category_name"],
                    "actual": round(float(r["actual_amount"]), 2),
                    "budget": category_budgets.get(r["category_name"])
                }
                for r in cat_actuals
            ]
        }
    finally:
        conn.close()


@app.post("/api/transactions")
def log_transaction(payload: TransactionCreateRequest, request: Request):
    user_id = get_current_user_id(request)
    date_str = payload.date.strip() if payload.date else str(datetime.date.today())
    amount = round(abs(payload.amount), 2)
    cat_input = payload.category.strip()
    pm_input = payload.payment_mode.strip()
    loc_input = (payload.location or "Unknown").strip()
    desc = (payload.description or "").strip()

    if amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be greater than zero")

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Map category
            cur.execute("SELECT category_id, category_name FROM categories;")
            cats = cur.fetchall()
            cat_id = None
            for c in cats:
                if str(c["category_id"]) == cat_input or c["category_name"].lower() == cat_input.lower():
                    cat_id = c["category_id"]
                    break
            if not cat_id:
                cat_id = 8  # Others default

            # Map payment mode
            cur.execute("SELECT payment_mode_id, mode_name FROM payment_modes;")
            pms = cur.fetchall()
            pm_id = None
            for p in pms:
                if str(p["payment_mode_id"]) == pm_input or p["mode_name"].lower() == pm_input.lower():
                    pm_id = p["payment_mode_id"]
                    break
            if not pm_id:
                pm_id = 1

            # Map location
            cur.execute("SELECT location_id, location_name FROM locations;")
            locs = cur.fetchall()
            loc_id = None
            for l in locs:
                if l["location_name"].lower() == loc_input.lower():
                    loc_id = l["location_id"]
                    break
            if not loc_id:
                loc_id = 1

            cur.execute(
                """
                INSERT INTO transactions (source_transaction_id, user_id, category_id, payment_mode_id, location_id, amount, transaction_type, transaction_date, description, source)
                VALUES (%s, %s, %s, %s, %s, %s, 'Expense', %s, %s, 'manual')
                RETURNING transaction_id;
                """,
                (None, user_id, cat_id, pm_id, loc_id, amount, date_str, desc)
            )
            new_txn_id = cur.fetchone()["transaction_id"]
        conn.commit()

        # Immediately refresh predictions
        refresh_res = inference_engine.refresh_predictions(conn, user_id)

        return {
            "status": "success",
            "transaction_id": new_txn_id,
            "refreshed_predictions": refresh_res
        }
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to log transaction: {str(e)}")
    finally:
        conn.close()


@app.get("/api/predictions")
def get_predictions(request: Request):
    user_id = get_current_user_id(request)
    conn = get_connection()
    try:
        res = inference_engine.refresh_predictions(conn, user_id)
        return res
    finally:
        conn.close()


@app.get("/api/alerts")
def get_alerts(request: Request):
    user_id = get_current_user_id(request)
    conn = get_connection()
    try:
        res = inference_engine.refresh_predictions(conn, user_id)
        return {
            "alerts": res.get("alerts", []),
            "data_status": res.get("data_status", [])
        }
    finally:
        conn.close()


# =========================================================
# EXISTING DEMO SQL ANALYTICAL ROUTES (Preserved intact)
# =========================================================

@app.get("/api/categories")
def get_categories():
    query = """
        SELECT DISTINCT c.category_name AS category
        FROM transactions t
        JOIN categories c ON t.category_id = c.category_id
        ORDER BY c.category_name;
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            return cursor.fetchall()
    finally:
        conn.close()


@app.get("/api/top-transactions")
def get_top_transactions():
    query = """
        SELECT t.transaction_id, t.user_id, c.category_name AS category, t.amount, t.transaction_type, t.transaction_date AS date, t.description
        FROM transactions t
        LEFT JOIN categories c ON t.category_id = c.category_id
        ORDER BY t.amount DESC LIMIT 5;
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            return cursor.fetchall()
    finally:
        conn.close()


@app.get("/api/category-summary")
def get_category_summary():
    query = """
        SELECT c.category_name AS category, COUNT(t.transaction_id) AS transaction_count, ROUND(SUM(t.amount), 2) AS total_amount
        FROM transactions t
        JOIN categories c ON t.category_id = c.category_id
        GROUP BY c.category_name ORDER BY total_amount DESC;
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            return cursor.fetchall()
    finally:
        conn.close()


@app.get("/api/transaction-details")
def get_transaction_details():
    query = """
        SELECT t.transaction_id, t.user_id, t.amount, t.transaction_type, t.transaction_date AS date, t.description, c.category_name AS category
        FROM transactions t
        LEFT JOIN categories c ON t.category_id = c.category_id
        ORDER BY t.transaction_date DESC LIMIT 20;
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            return cursor.fetchall()
    finally:
        conn.close()


@app.get("/api/user-spending")
def get_user_spending():
    query = """
        SELECT user_id, transaction_date AS date, amount, transaction_type
        FROM (
            SELECT user_id, transaction_date, amount, transaction_type,
                   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount DESC) AS rn
            FROM transactions WHERE transaction_type = 'Expense'
        ) ranked
        WHERE rn = 1 ORDER BY amount DESC LIMIT 20;
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            return cursor.fetchall()
    finally:
        conn.close()


@app.get("/api/max-category/{user_id}")
def get_max_category(user_id: str):
    query = """
        SELECT c.category_name AS category, ROUND(SUM(t.amount), 2) AS amount
        FROM transactions t
        JOIN categories c ON t.category_id = c.category_id
        WHERE t.user_id = %s AND t.transaction_type = 'Expense'
        GROUP BY c.category_name ORDER BY amount DESC LIMIT 1;
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (user_id,))
            data = cursor.fetchone()
        if not data:
            raise HTTPException(status_code=404, detail="No expense data found for this user")
        return data
    finally:
        conn.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)