import os
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Personal Finance Analytics",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# DATABASE
# =========================================================

def get_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="personal_finance",
        user="postgres",
        password="Shiva@138"
    )


# =========================================================
# BASIC ROUTES
# =========================================================

@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/pages/{page_name}")
def serve_page(page_name: str):

    allowed_pages = {
        "categories.html",
        "category-summary.html",
        "highest-category.html",
        "top-transactions.html",
        "transaction-details.html",
        "user-spending.html",
    }

    if page_name not in allowed_pages:
        raise HTTPException(status_code=404, detail="Page not found")

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

        return {
            "status": "connected",
            "database": database
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


# =========================================================
# 1. DISTINCT CATEGORIES
# =========================================================

@app.get("/api/categories")
def get_categories():

    query = """
        SELECT DISTINCT
            c.category_name AS category
        FROM transactions t
        JOIN categories c
            ON t.category_id = c.category_id
        ORDER BY c.category_name;
    """

    try:

        conn = get_connection()

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            data = cursor.fetchall()

        conn.close()

        return data

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# 2. TOP 5 TRANSACTIONS
# =========================================================

@app.get("/api/top-transactions")
def get_top_transactions():

    query = """
        SELECT
            t.transaction_id,
            t.user_id,
            c.category_name AS category,
            t.amount,
            t.transaction_type,
            t.transaction_date AS date,
            t.description
        FROM transactions t
        LEFT JOIN categories c
            ON t.category_id = c.category_id
        ORDER BY t.amount DESC
        LIMIT 5;
    """

    try:

        conn = get_connection()

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            data = cursor.fetchall()

        conn.close()

        return data

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# 3. CATEGORY GROUP BY
# =========================================================

@app.get("/api/category-summary")
def get_category_summary():

    query = """
        SELECT
            c.category_name AS category,
            COUNT(t.transaction_id) AS transaction_count,
            ROUND(SUM(t.amount), 2) AS total_amount
        FROM transactions t
        JOIN categories c
            ON t.category_id = c.category_id
        GROUP BY c.category_name
        ORDER BY total_amount DESC;
    """

    try:

        conn = get_connection()

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            data = cursor.fetchall()

        conn.close()

        return data

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# 4. SELECT FEW COLUMNS
# =========================================================

@app.get("/api/transaction-details")
def get_transaction_details():

    query = """
        SELECT
            t.transaction_id,
            t.user_id,
            t.amount,
            t.transaction_type,
            t.transaction_date AS date,
            t.description,
            c.category_name AS category
        FROM transactions t
        LEFT JOIN categories c
            ON t.category_id = c.category_id
        ORDER BY t.transaction_date DESC
        LIMIT 20;
    """

    try:

        conn = get_connection()

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            data = cursor.fetchall()

        conn.close()

        return data

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# 5. TOP SPENDING OF EACH USER
# =========================================================

@app.get("/api/user-spending")
def get_user_spending():

    query = """
        SELECT
            user_id,
            transaction_date AS date,
            amount,
            transaction_type
        FROM (
            SELECT
                user_id,
                transaction_date,
                amount,
                transaction_type,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY amount DESC
                ) AS rn
            FROM transactions
            WHERE transaction_type = 'Expense'
        ) ranked
        WHERE rn = 1
        ORDER BY amount DESC
        LIMIT 20;
    """

    try:

        conn = get_connection()

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            data = cursor.fetchall()

        conn.close()

        return data

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# 6. CATEGORY WITH MAX SPENDING FOR A USER
# =========================================================

@app.get("/api/max-category/{user_id}")
def get_max_category(user_id: str):

    query = """
        SELECT
            c.category_name AS category,
            ROUND(SUM(t.amount), 2) AS amount
        FROM transactions t
        JOIN categories c
            ON t.category_id = c.category_id
        WHERE t.user_id = %s
          AND t.transaction_type = 'Expense'
        GROUP BY c.category_name
        ORDER BY amount DESC
        LIMIT 1;
    """

    try:

        conn = get_connection()

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (user_id,))
            data = cursor.fetchone()

        conn.close()

        if data is None:
            raise HTTPException(
                status_code=404,
                detail="No expense data found for this user"
            )

        return data

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
   # =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )