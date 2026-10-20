import calendar
import hmac
import logging
import os
import secrets
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from functools import wraps
from pathlib import Path

import pandas as pd
import psycopg2
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from psycopg2.extras import RealDictCursor, execute_batch
from werkzeug.security import check_password_hash, generate_password_hash

from backend.inference import (
    predict_category_daily,
    predict_overall_transaction_steps,
)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR),
    static_folder=None,
)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "false").lower()
    == "true",
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
    MAX_CONTENT_LENGTH=int(os.environ.get("MAX_CONTENT_LENGTH", 12 * 1024 * 1024)),
)


@contextmanager
def database_connection():
    settings = {
        "host": os.environ.get("DB_HOST", "localhost"),
        "port": int(os.environ.get("DB_PORT", "5432")),
        "dbname": os.environ.get("DB_NAME", "personal_finance"),
        "user": os.environ.get("DB_USER", "postgres"),
        "password": os.environ.get("DB_PASSWORD", ""),
        "connect_timeout": 5,
    }
    connection = psycopg2.connect(**settings)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def api_error(message, status):
    return jsonify({"error": message}), status


def csrf_token():
    token = session.get("_csrf_token")
    if token is None:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


app.jinja_env.globals["csrf_token"] = csrf_token


@app.before_request
def validate_csrf():
    if request.method != "POST":
        return None
    expected = session.get("_csrf_token", "")
    supplied = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token", "")
    if not expected or not supplied or not hmac.compare_digest(expected, supplied):
        if request.path.startswith("/api/"):
            return api_error("Request token is missing or expired. Refresh and try again.", 400)
        abort(400)
    return None


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            if request.path.startswith("/api/"):
                return api_error("Please log in to continue.", 401)
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


def user_is_onboarded(user_id):
    with database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM user_preferences WHERE user_id = %s",
                (user_id,),
            )
            return cursor.fetchone() is not None


def require_onboarding(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if not user_is_onboarded(session["user_id"]):
            if request.path.startswith("/api/"):
                return api_error("Complete onboarding before accessing the dashboard.", 403)
            return redirect(url_for("onboarding"))
        return view(*args, **kwargs)

    return wrapped


@app.get("/style.css")
def style():
    return send_from_directory(FRONTEND_DIR, "style.css")


@app.get("/app.js")
def javascript():
    return send_from_directory(FRONTEND_DIR, "app.js")


@app.get("/")
def index():
    if not session.get("user_id"):
        return redirect(url_for("login"))
    return redirect(
        url_for("dashboard")
        if user_is_onboarded(session["user_id"])
        else url_for("onboarding")
    )


@app.get("/login")
def login():
    if session.get("user_id"):
        return index()
    return render_template("login.html")


@app.get("/register")
def register_page():
    if session.get("user_id"):
        return index()
    return render_template("register.html")


@app.get("/onboarding")
@login_required
def onboarding():
    if user_is_onboarded(session["user_id"]):
        return redirect(url_for("dashboard"))
    return render_template("onboarding.html")


@app.get("/dashboard")
@require_onboarding
def dashboard():
    return render_template("dashboard.html", username=session.get("username", ""))


@app.get("/api/health")
def health():
    try:
        with database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_database()")
                database = cursor.fetchone()[0]
        return jsonify({"status": "connected", "database": database})
    except psycopg2.Error:
        logger.exception("Database health check failed")
        return api_error("Database connection failed.", 503)


@app.post("/api/register")
def register():
    data = request.get_json(silent=True) or request.form
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))
    if not 3 <= len(username) <= 64:
        return api_error("Username must be between 3 and 64 characters.", 400)
    if len(password) < 8 or len(password) > 256:
        return api_error("Password must be between 8 and 256 characters.", 400)

    user_id = f"U{uuid.uuid4().hex[:9].upper()}"
    email = f"{user_id.lower()}@accounts.local"
    password_hash = generate_password_hash(password)
    try:
        with database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO users (user_id, username, name, email, password_hash)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (user_id, username, username, email, password_hash),
                )
        session.clear()
        session.permanent = True
        session["user_id"] = user_id
        session["username"] = username
        csrf_token()
        return jsonify({"redirect": url_for("onboarding")}), 201
    except psycopg2.errors.UniqueViolation:
        return api_error("That username is already taken.", 409)
    except psycopg2.Error:
        logger.exception("Account registration failed")
        return api_error("Unable to create the account. Please try again.", 500)


@app.post("/api/login")
def login_api():
    data = request.get_json(silent=True) or request.form
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))
    if not username or not password:
        return api_error("Enter both username and password.", 400)

    try:
        with database_connection() as connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT user_id, username, password_hash
                    FROM users
                    WHERE LOWER(username) = LOWER(%s)
                    """,
                    (username,),
                )
                user = cursor.fetchone()
        if user is None or not check_password_hash(user["password_hash"], password):
            return api_error("Invalid username or password.", 401)
        session.clear()
        session.permanent = True
        session["user_id"] = user["user_id"]
        session["username"] = user["username"]
        csrf_token()
        destination = (
            url_for("dashboard")
            if user_is_onboarded(user["user_id"])
            else url_for("onboarding")
        )
        return jsonify({"redirect": destination})
    except psycopg2.Error:
        logger.exception("Login query failed")
        return api_error("Unable to log in right now. Please try again.", 503)


@app.post("/api/logout")
@login_required
def logout():
    session.clear()
    return jsonify({"redirect": url_for("login")})


@app.post("/api/account/delete")
@login_required
def delete_account():
    user_id = session["user_id"]
    try:
        with database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM goal_progress
                    WHERE goal_id IN (
                        SELECT goal_id FROM goals WHERE user_id = %s
                    )
                    """,
                    (user_id,),
                )
                cursor.execute(
                    "DELETE FROM monthly_summary WHERE user_id = %s",
                    (user_id,),
                )
                cursor.execute(
                    "DELETE FROM category_monthly_summary WHERE user_id = %s",
                    (user_id,),
                )
                cursor.execute(
                    "DELETE FROM detected_recurring_payments WHERE user_id = %s",
                    (user_id,),
                )
                cursor.execute(
                    "DELETE FROM users WHERE user_id = %s",
                    (user_id,),
                )
        session.clear()
        return jsonify({"redirect": url_for("login")})
    except psycopg2.Error:
        logger.exception("Account deletion failed for user %s", user_id)
        return api_error("Unable to delete your account right now. Please try again.", 503)


@app.get("/api/me")
@login_required
def current_user():
    return jsonify(
        {
            "username": session.get("username"),
            "onboarded": user_is_onboarded(session["user_id"]),
        }
    )


@app.get("/api/categories")
def categories():
    try:
        with database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT category_name FROM categories ORDER BY category_name")
                names = [row[0] for row in cursor.fetchall()]
        return jsonify(names)
    except psycopg2.Error:
        logger.exception("Category lookup failed")
        return api_error("Unable to load categories.", 503)


def parse_import(file_storage, category_ids, payment_mode_ids, location_ids):
    if file_storage is None or not file_storage.filename:
        return []
    if not file_storage.filename.lower().endswith(".csv"):
        raise ValueError("Upload a CSV file.")
    try:
        frame = pd.read_csv(file_storage, dtype={"transaction_id": str})
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Could not read this CSV: {error}") from error
    if frame.empty:
        raise ValueError("The CSV has no transaction rows.")

    aliases = {
        "date": ("date", "txn_date", "transaction_date"),
        "category": ("category",),
        "amount": ("amount",),
        "transaction_type": ("transaction_type", "type"),
        "description": ("description", "notes"),
        "source_transaction_id": ("transaction_id", "source_transaction_id"),
        "payment_mode": ("payment_mode",),
        "location": ("location",),
    }
    columns = {str(name).strip().lower(): name for name in frame.columns}
    selected = {
        target: next((columns[key] for key in options if key in columns), None)
        for target, options in aliases.items()
    }
    missing = [
        name for name in ("date", "category", "amount") if selected[name] is None
    ]
    if missing:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(missing)
        )

    frame["_parsed_date"] = pd.to_datetime(
        frame[selected["date"]],
        errors="coerce",
        dayfirst=False,
    )
    frame["_parsed_amount"] = pd.to_numeric(
        frame[selected["amount"]],
        errors="coerce",
    )
    category_key = {name.casefold(): (name, category_id) for name, category_id in category_ids.items()}

    imported = []
    errors = []
    today = date.today()
    for row_number, (_, row) in enumerate(frame.iterrows(), start=2):
        parsed_date = row["_parsed_date"]
        amount = row["_parsed_amount"]
        category_value = row[selected["category"]]
        if pd.isna(parsed_date):
            errors.append(f"row {row_number}: invalid date")
            continue
        if pd.isna(amount) or not np_is_finite_positive(amount):
            errors.append(f"row {row_number}: amount must be a positive number")
            continue
        if parsed_date.date() > today:
            errors.append(f"row {row_number}: transaction date cannot be in the future")
            continue
        if pd.isna(category_value) or str(category_value).strip().casefold() not in category_key:
            errors.append(f"row {row_number}: category is not in the application category list")
            continue
        category_name, category_id = category_key[str(category_value).strip().casefold()]

        transaction_type = "Expense"
        if selected["transaction_type"] is not None and not pd.isna(row[selected["transaction_type"]]):
            transaction_type = str(row[selected["transaction_type"]]).strip().title()
        if transaction_type not in ("Expense", "Income"):
            errors.append(f"row {row_number}: transaction_type must be Expense or Income")
            continue

        source_id = None
        if selected["source_transaction_id"] is not None:
            value = row[selected["source_transaction_id"]]
            if not pd.isna(value):
                source_id = str(value).strip()
                if len(source_id) > 20:
                    errors.append(f"row {row_number}: transaction_id exceeds 20 characters")
                    continue

        description = None
        if selected["description"] is not None:
            value = row[selected["description"]]
            if not pd.isna(value):
                description = str(value).strip() or None

        def lookup_id(source_column, mapping):
            if source_column is None or pd.isna(row[source_column]):
                return None
            return mapping.get(str(row[source_column]).strip().casefold())

        payment_mode_id = lookup_id(selected["payment_mode"], payment_mode_ids)
        location_id = lookup_id(selected["location"], location_ids)
        imported.append(
            (
                source_id,
                category_id,
                payment_mode_id,
                location_id,
                round(float(amount), 2),
                transaction_type,
                parsed_date.date(),
                description,
            )
        )

    if errors:
        preview = "; ".join(errors[:8])
        suffix = f" (and {len(errors) - 8} more errors)" if len(errors) > 8 else ""
        raise ValueError(f"CSV validation failed: {preview}{suffix}")
    return imported


def np_is_finite_positive(value):
    try:
        return float(value) > 0 and pd.notna(value) and float(value) != float("inf")
    except (TypeError, ValueError, OverflowError):
        return False


@app.post("/api/onboarding")
@login_required
def save_onboarding():
    overall_budget_value = request.form.get("monthly_budget", "").strip()
    try:
        overall_budget = float(overall_budget_value)
        if not pd.notna(overall_budget) or overall_budget < 0 or overall_budget == float("inf"):
            raise ValueError
    except (TypeError, ValueError):
        return api_error("Enter a valid non-negative monthly budget.", 400)

    category_budgets = {}
    for key, value in request.form.items():
        if not key.startswith("category_budget[") or not key.endswith("]"):
            continue
        category_name = key[len("category_budget["):-1].strip()
        if not value.strip():
            continue
        try:
            amount = float(value)
            if pd.isna(amount) or amount < 0 or amount == float("inf"):
                raise ValueError
        except (TypeError, ValueError):
            return api_error(f"Enter a valid non-negative budget for {category_name}.", 400)
        category_budgets[category_name.casefold()] = amount

    try:
        with database_connection() as connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT category_id, category_name FROM categories"
                )
                rows = cursor.fetchall()
                category_ids = {row["category_name"]: row["category_id"] for row in rows}
                category_ids_by_key = {
                    row["category_name"].casefold(): row["category_id"] for row in rows
                }
                cursor.execute(
                    "SELECT mode_name, payment_mode_id FROM payment_modes"
                )
                payment_mode_ids = {
                    name.casefold(): identifier
                    for name, identifier in cursor.fetchall()
                }
                cursor.execute(
                    "SELECT location_name, location_id FROM locations"
                )
                location_ids = {
                    name.casefold(): identifier
                    for name, identifier in cursor.fetchall()
                }
                uploaded_rows = parse_import(
                    request.files.get("history_csv"),
                    category_ids,
                    payment_mode_ids,
                    location_ids,
                )

                unknown_budgets = set(category_budgets) - set(category_ids_by_key)
                if unknown_budgets:
                    return api_error(
                        "Unknown budget categories: " + ", ".join(sorted(unknown_budgets)),
                        400,
                    )

                today = date.today()
                month_start = today.replace(day=1)
                cursor.execute(
                    """
                    INSERT INTO user_preferences (user_id, monthly_budget)
                    VALUES (%s, %s)
                    ON CONFLICT (user_id) DO UPDATE
                    SET monthly_budget = EXCLUDED.monthly_budget
                    """,
                    (session["user_id"], overall_budget),
                )
                for category, budget_amount in category_budgets.items():
                    cursor.execute(
                        """
                        INSERT INTO budgets (user_id, category_id, month, budget_amount)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (user_id, category_id, month)
                        DO UPDATE SET budget_amount = EXCLUDED.budget_amount
                        """,
                        (
                            session["user_id"],
                            category_ids_by_key[category],
                            month_start,
                            budget_amount,
                        ),
                    )
                if uploaded_rows:
                    execute_batch(
                        cursor,
                        """
                        INSERT INTO transactions (
                            source_transaction_id, user_id, category_id,
                            payment_mode_id, location_id, amount,
                            transaction_type, transaction_date, description, source
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'csv')
                        """,
                        [
                            (
                                row[0],
                                session["user_id"],
                                *row[1:],
                            )
                            for row in uploaded_rows
                        ],
                    )
        session["onboarded"] = True
        return jsonify(
            {
                "redirect": url_for("dashboard"),
                "imported_transactions": len(uploaded_rows),
            }
        )
    except ValueError as error:
        return api_error(str(error), 400)
    except psycopg2.Error:
        logger.exception("Onboarding persistence failed")
        return api_error("Unable to save onboarding details. Please try again.", 500)


@app.post("/api/transactions")
@require_onboarding
def add_transaction():
    data = request.get_json(silent=True) or request.form
    category = str(data.get("category", "")).strip()
    description = str(data.get("description", "")).strip()
    date_value = str(data.get("date", "")).strip()
    amount_value = str(data.get("amount", "")).strip()
    try:
        transaction_date = date.fromisoformat(date_value)
        amount = float(amount_value)
        if not pd.notna(amount) or amount <= 0 or amount == float("inf"):
            raise ValueError
        if transaction_date > date.today():
            return api_error("Transaction date cannot be in the future.", 400)
    except (TypeError, ValueError):
        return api_error("Enter a valid date and positive amount.", 400)
    if len(description) > 500:
        return api_error("Description must be 500 characters or fewer.", 400)

    try:
        with database_connection() as connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT category_id, category_name
                    FROM categories
                    WHERE LOWER(category_name) = LOWER(%s)
                    """,
                    (category,),
                )
                selected_category = cursor.fetchone()
                if selected_category is None:
                    return api_error("Select a valid category.", 400)
                cursor.execute(
                    """
                    INSERT INTO transactions (
                        user_id, category_id, amount, transaction_type,
                        transaction_date, description, source
                    )
                    VALUES (%s, %s, %s, 'Expense', %s, %s, 'manual')
                    RETURNING transaction_id
                    """,
                    (
                        session["user_id"],
                        selected_category["category_id"],
                        round(amount, 2),
                        transaction_date,
                        description or None,
                    ),
                )
                transaction_id = cursor.fetchone()["transaction_id"]
        return jsonify({"transaction_id": transaction_id}), 201
    except psycopg2.Error:
        logger.exception("Transaction insert failed")
        return api_error("Unable to save this transaction.", 500)


def model_id(cursor, model_name, model_type, model_path):
    cursor.execute(
        """
        SELECT model_id FROM model_registry
        WHERE model_name = %s AND model_type = %s
        ORDER BY model_id LIMIT 1
        """,
        (model_name, model_type),
    )
    row = cursor.fetchone()
    if row:
        return row[0] if not isinstance(row, dict) else row["model_id"]
    cursor.execute(
        """
        INSERT INTO model_registry (model_name, model_type, blob_storage_path)
        VALUES (%s, %s, %s)
        RETURNING model_id
        """,
        (model_name, model_type, str(model_path)),
    )
    row = cursor.fetchone()
    return row[0] if not isinstance(row, dict) else row["model_id"]


def persist_predictions(user_id, transactions, overall, category_predictions, as_of):
    if not overall.get("available") and not category_predictions.get("categories"):
        return
    try:
        with database_connection() as connection:
            with connection.cursor() as cursor:
                overall_id = None
                if overall.get("available"):
                    overall_id = model_id(
                        cursor,
                        "Overall seven-step LSTM",
                        "LSTM_OVERALL",
                        PROJECT_DIR / "ml" / "lstm" / "lstm_model.pt",
                    )
                category_id = None
                if category_predictions.get("categories"):
                    category_id = model_id(
                        cursor,
                        "Category seven-day LSTM",
                        "LSTM_CATEGORY",
                        PROJECT_DIR / "ml" / "lstm_categories" / "models",
                    )

                cursor.execute(
                    """
                    DELETE FROM predictions
                    WHERE user_id = %s
                      AND (
                        prediction_type LIKE 'overall_transaction_step_%%'
                        OR prediction_type = 'category_daily'
                      )
                    """,
                    (user_id,),
                )

                if overall_id is not None:
                    cursor.executemany(
                        """
                        INSERT INTO predictions (
                            user_id, model_id, prediction_type, target_period,
                            predicted_value
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                user_id,
                                overall_id,
                                f"overall_transaction_step_{step}",
                                as_of,
                                value,
                            )
                            for step, value in enumerate(overall["values"], start=1)
                        ],
                    )

                if category_id is not None:
                    cursor.execute(
                        """
                        SELECT category_id, category_name FROM categories
                        """
                    )
                    ids = {name: identifier for identifier, name in cursor.fetchall()}
                    category_rows = []
                    for name, values in category_predictions["categories"].items():
                        if name not in ids:
                            continue
                        for offset, value in enumerate(values, start=1):
                            category_rows.append(
                                (
                                    user_id,
                                    category_id,
                                    "category_daily",
                                    ids[name],
                                    as_of + timedelta(days=offset),
                                    value,
                                )
                            )
                    cursor.executemany(
                        """
                        INSERT INTO predictions (
                            user_id, model_id, prediction_type, category_id,
                            target_period, predicted_value
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        category_rows,
                    )
    except psycopg2.Error:
        logger.exception("Could not persist generated model predictions")


def build_dashboard(user_id):
    today = date.today()
    month_start = today.replace(day=1)
    six_months_ago = (pd.Timestamp(month_start) - pd.DateOffset(months=5)).date()

    with database_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT t.transaction_id, t.user_id, t.amount,
                       t.transaction_type, t.transaction_date, t.description,
                       c.category_id, c.category_name AS category
                FROM transactions t
                JOIN categories c ON c.category_id = t.category_id
                WHERE t.user_id = %s
                ORDER BY t.transaction_date, t.transaction_id
                """,
                (user_id,),
            )
            transactions = cursor.fetchall()
            cursor.execute(
                """
                SELECT monthly_budget FROM user_preferences WHERE user_id = %s
                """,
                (user_id,),
            )
            preferences = cursor.fetchone()
            cursor.execute(
                """
                SELECT b.budget_amount, c.category_id, c.category_name AS category
                FROM budgets b
                JOIN categories c ON c.category_id = b.category_id
                WHERE b.user_id = %s AND b.month = %s
                """,
                (user_id, month_start),
            )
            budget_rows = cursor.fetchall()
            cursor.execute(
                """
                SELECT c.category_name AS category, SUM(t.amount) AS amount
                FROM transactions t
                JOIN categories c ON c.category_id = t.category_id
                WHERE t.user_id = %s AND t.transaction_type = 'Expense'
                  AND t.transaction_date >= %s AND t.transaction_date <= %s
                GROUP BY c.category_name
                ORDER BY amount DESC
                """,
                (user_id, six_months_ago, today),
            )
            pie_rows = cursor.fetchall()

    overall = {}
    category_predictions = {
        "available": False,
        "reason": "No expense history is available.",
        "categories": {},
        "errors": [],
    }
    inference_errors = []
    try:
        overall = predict_overall_transaction_steps(transactions)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        logger.exception("Overall LSTM inference failed")
        overall = {"available": False, "reason": str(error), "values": []}
        inference_errors.append(str(error))
    try:
        category_predictions = predict_category_daily(transactions, today)
    except (FileNotFoundError, ValueError, RuntimeError, OSError, KeyError) as error:
        logger.exception("Category LSTM inference failed")
        category_predictions = {
            "available": False,
            "reason": str(error),
            "categories": {},
            "errors": [str(error)],
        }
        inference_errors.append(str(error))

    actual_month = sum(
        float(row["amount"])
        for row in transactions
        if row["transaction_type"] == "Expense"
        and month_start <= row["transaction_date"] <= today
    )
    category_actual = {}
    for row in transactions:
        if (
            row["transaction_type"] == "Expense"
            and month_start <= row["transaction_date"] <= today
        ):
            category_actual[row["category"]] = (
                category_actual.get(row["category"], 0.0) + float(row["amount"])
            )

    category_budgets = {
        row["category"]: float(row["budget_amount"]) for row in budget_rows
    }
    days_remaining = calendar.monthrange(today.year, today.month)[1] - today.day
    predicted_categories = {}
    for name, daily_values in category_predictions.get("categories", {}).items():
        actual_amount = category_actual.get(name, 0.0)
        forecast_mean = sum(daily_values) / len(daily_values)
        projected_month = actual_amount + forecast_mean * days_remaining
        predicted_categories[name] = {
            "next_7_days": daily_values,
            "actual_month": round(actual_amount, 2),
            "budget": category_budgets.get(name),
            "projected_month": round(projected_month, 2),
        }

    expense_history_categories = {
        row["category"]
        for row in transactions
        if row["transaction_type"] == "Expense"
    }
    relevant_categories = expense_history_categories | {
        name for name, budget in category_budgets.items() if budget > 0
    }
    overall_projection_available = (
        bool(predicted_categories)
        and relevant_categories.issubset(predicted_categories)
    )
    predicted_month = (
        round(sum(item["projected_month"] for item in predicted_categories.values()), 2)
        if overall_projection_available
        else None
    )
    overall_budget = (
        float(preferences["monthly_budget"]) if preferences is not None else None
    )

    alerts = []
    if overall_budget is not None:
        if actual_month > overall_budget:
            alerts.append(
                {
                    "severity": "red",
                    "scope": "Overall budget",
                    "message": f"Actual spending ₹{actual_month:,.2f} exceeded "
                    f"the ₹{overall_budget:,.2f} monthly budget.",
                }
            )
        elif predicted_month is not None and predicted_month > overall_budget:
            alerts.append(
                {
                    "severity": "yellow",
                    "scope": "Overall budget",
                    "message": f"Projected month-end spending ₹{predicted_month:,.2f} "
                    f"may exceed the ₹{overall_budget:,.2f} monthly budget.",
                }
            )

    for name, budget in category_budgets.items():
        actual_amount = category_actual.get(name, 0.0)
        projection = predicted_categories.get(name, {}).get("projected_month")
        if actual_amount > budget:
            alerts.append(
                {
                    "severity": "red",
                    "scope": name,
                    "message": f"Actual {name} spending ₹{actual_amount:,.2f} "
                    f"exceeded the ₹{budget:,.2f} category budget.",
                }
            )
        elif projection is not None and projection > budget:
            alerts.append(
                {
                    "severity": "yellow",
                    "scope": name,
                    "message": f"Projected {name} spending ₹{projection:,.2f} "
                    f"may exceed the ₹{budget:,.2f} category budget.",
                }
            )

    pie_total = sum(float(row["amount"]) for row in pie_rows)
    pie = [
        {
            "category": row["category"],
            "amount": round(float(row["amount"]), 2),
            "percentage": round(float(row["amount"]) / pie_total * 100, 1)
            if pie_total
            else 0,
        }
        for row in pie_rows
    ]
    transactions_recent = [
        {
            "date": row["transaction_date"].isoformat(),
            "description": row["description"] or "",
            "category": row["category"],
            "amount": round(float(row["amount"]), 2),
            "type": row["transaction_type"],
        }
        for row in reversed(transactions[-30:])
    ]

    persist_predictions(
        user_id,
        transactions,
        overall,
        category_predictions,
        today,
    )
    return {
        "month": month_start.isoformat(),
        "currency": "INR",
        "actual_monthly_spending": round(actual_month, 2),
        "monthly_budget": round(overall_budget, 2) if overall_budget is not None else None,
        "projected_monthly_spending": predicted_month,
        "projection_method": (
            "Actual month-to-date spending plus the mean of the category LSTMs' "
            "next-seven-day forecasts for each remaining calendar day."
            if predicted_month is not None
            else None
        ),
        "overall_lstm": overall,
        "category_predictions": predicted_categories,
        "category_inference": {
            "available": category_predictions.get("available", False),
            "reason": category_predictions.get("reason"),
            "errors": category_predictions.get("errors", []),
        },
        "category_spending_six_months": pie,
        "transactions": transactions_recent,
        "alerts": alerts,
        "inference_errors": inference_errors,
    }


@app.get("/api/dashboard")
@require_onboarding
def dashboard_api():
    try:
        return jsonify(build_dashboard(session["user_id"]))
    except psycopg2.Error:
        logger.exception("Dashboard database query failed")
        return api_error("Unable to load your analytics right now.", 503)


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true",
    )
