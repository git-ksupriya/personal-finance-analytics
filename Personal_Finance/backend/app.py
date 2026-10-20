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
from psycopg2.extras import Json, RealDictCursor, execute_batch
from werkzeug.security import check_password_hash, generate_password_hash

from backend.inference import predict_category_monthly, predict_overall_monthly
from backend.spending_analytics import (
    detect_prediction_anomalies,
    detect_transaction_anomalies,
    monthly_financial_totals,
    select_user_history,
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


def parse_recurring_payments(form, category_ids):
    types = form.getlist("recurring_payment_type[]")
    categories = form.getlist("recurring_category[]")
    descriptions = form.getlist("recurring_description[]")
    amounts = form.getlist("recurring_amount[]")
    due_dates = form.getlist("recurring_next_due_date[]")
    lengths = {len(types), len(categories), len(descriptions), len(amounts), len(due_dates)}
    if len(lengths) > 1:
        raise ValueError("Recurring payment details are incomplete.")

    known_categories = {
        name.casefold(): (name, identifier)
        for name, identifier in category_ids.items()
    }
    result = []
    today = date.today()
    for index, values in enumerate(zip(types, categories, descriptions, amounts, due_dates), start=1):
        payment_type, category, description, amount_text, due_text = values
        if not any(value.strip() for value in values):
            continue
        payment_type = payment_type.strip().casefold()
        if payment_type not in ("income", "expenditure"):
            raise ValueError(f"Recurring entry {index}: select income or expenditure.")
        category_match = known_categories.get(category.strip().casefold())
        if category_match is None:
            raise ValueError(f"Recurring entry {index}: select a valid category.")
        try:
            amount = float(amount_text)
            due_date = date.fromisoformat(due_text)
            if not np_is_finite_positive(amount):
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(f"Recurring entry {index}: enter a positive amount and valid due date.")
        if len(description.strip()) > 500:
            raise ValueError(f"Recurring entry {index}: description must be 500 characters or fewer.")
        result.append(
            {
                "payment_type": payment_type,
                "category_id": category_match[1],
                "amount": round(amount, 2),
                "next_due_date": due_date,
                "description": description.strip() or category_match[0],
            }
        )
    return result


def _load_financial_context(cursor, user_id, today):
    month_start = today.replace(day=1)
    cursor.execute(
        """
        SELECT t.transaction_id, t.user_id, t.amount, t.transaction_type,
               t.transaction_date, t.description, t.source,
               c.category_id, c.category_name AS category
        FROM transactions t
        JOIN categories c ON c.category_id = t.category_id
        WHERE t.user_id = %s
        ORDER BY t.transaction_date, t.transaction_id
        """,
        (user_id,),
    )
    transactions = cursor.fetchall()
    cursor.execute("SELECT created_at FROM users WHERE user_id = %s", (user_id,))
    user_row = cursor.fetchone()
    account_created_at = user_row["created_at"] if user_row else None
    cursor.execute(
        "SELECT monthly_budget FROM user_preferences WHERE user_id = %s",
        (user_id,),
    )
    preference = cursor.fetchone()
    cursor.execute(
        """
        SELECT b.budget_amount, c.category_id, c.category_name AS category
        FROM budgets b
        JOIN categories c ON c.category_id = b.category_id
        WHERE b.user_id = %s AND b.month = %s
        """,
        (user_id, month_start),
    )
    budgets = cursor.fetchall()
    cursor.execute(
        """
        SELECT recurring_id, payment_type, category_id, amount, frequency,
               next_due_date, description, is_active
        FROM recurring_payments
        WHERE user_id = %s AND is_active = TRUE AND frequency = 'monthly'
        """,
        (user_id,),
    )
    recurring = cursor.fetchall()
    cursor.execute("SELECT category_id, category_name FROM categories")
    categories = {name: identifier for identifier, name in cursor.fetchall()}
    return (
        transactions,
        account_created_at,
        preference,
        budgets,
        recurring,
        categories,
    )


def _persist_anomalies(cursor, user_id, anomalies, category_ids):
    predicted_event_keys = [
        event["event_key"]
        for event in anomalies
        if event["detection_type"] == "predicted"
    ]
    cursor.execute(
        """
        UPDATE anomalies
        SET status = 'resolved', resolved_at = NOW()
        WHERE user_id = %s AND detection_type = 'predicted'
          AND status = 'active'
          AND (event_key IS NULL OR event_key <> ALL(%s))
        """,
        (user_id, predicted_event_keys),
    )
    for event in anomalies:
        cursor.execute(
            """
            INSERT INTO anomalies (
                transaction_id, model_id, user_id, prediction_id, category_id,
                reconstruction_error, is_anomaly, scope, detection_type,
                target_period, observed_amount, historical_mean, first_quartile,
                third_quartile, threshold_amount, history_count,
                detection_method, reason, event_key
            )
            VALUES (%s, NULL, %s, %s, %s, 0, TRUE, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, event_key)
                WHERE user_id IS NOT NULL AND event_key IS NOT NULL
            DO UPDATE SET
                prediction_id = EXCLUDED.prediction_id,
                observed_amount = EXCLUDED.observed_amount,
                historical_mean = EXCLUDED.historical_mean,
                first_quartile = EXCLUDED.first_quartile,
                third_quartile = EXCLUDED.third_quartile,
                threshold_amount = EXCLUDED.threshold_amount,
                history_count = EXCLUDED.history_count,
                detection_method = EXCLUDED.detection_method,
                reason = EXCLUDED.reason,
                status = 'active',
                resolved_at = NULL
            """,
            (
                event["transaction_id"],
                user_id,
                event["prediction_id"],
                category_ids.get(event["category"]),
                event["scope"],
                event["detection_type"],
                event["target_period"],
                event["observed_amount"],
                event["historical_mean"],
                event["q1"],
                event["q3"],
                event["threshold"],
                event["history_count"],
                event["method"],
                event["reason"],
                event["event_key"],
            ),
        )


def _persist_budget_alerts(
    cursor,
    user_id,
    today,
    totals,
    transactions,
    budgets,
    predicted_categories=None,
    predicted_overall=None,
    predictions_current=True,
    overall_predictions_current=None,
    has_overall_budget=True,
):
    month_start = today.replace(day=1)
    actual_month = totals["actual_monthly_spending"]
    overall_limit = totals["net_after_recurring_obligations"]
    category_budgets = {
        row["category"]: (float(row["budget_amount"]), row["category_id"])
        for row in budgets
    }
    category_actual = {}
    for row in transactions:
        if row["transaction_type"] != "Expense":
            continue
        if month_start <= row["transaction_date"] <= today:
            category_actual[row["category"]] = (
                category_actual.get(row["category"], 0.0) + float(row["amount"])
            )

    category_result = predicted_categories or {}
    target_period = month_start.isoformat()

    def forecast_for_month(result, category=None):
        if not result or target_period not in result.get("target_periods", []):
            return None
        index = result["target_periods"].index(target_period)
        values = (
            result.get("categories", {}).get(category, [])
            if category is not None
            else result.get("values", [])
        )
        if index >= len(values):
            return None
        return float(values[index])

    projected_categories = {}
    if predictions_current:
        for category in category_result.get("categories", {}):
            projection = forecast_for_month(category_result, category)
            if projection is not None:
                projected_categories[category] = projection
    projected_overall = (
        forecast_for_month(predicted_overall)
        if (
            predictions_current
            if overall_predictions_current is None
            else overall_predictions_current
        )
        else None
    )

    desired = []

    def add_alert(alert_type, scope, category_id, message):
        period = month_start.isoformat()
        dedupe_key = f"{alert_type}:{scope}:{period}"
        desired.append(dedupe_key)
        cursor.execute(
            """
            INSERT INTO alerts (
                user_id, alert_type, message, severity, scope, category_id,
                target_period, dedupe_key, status, resolved_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'active', NULL)
            ON CONFLICT (user_id, dedupe_key)
                WHERE dedupe_key IS NOT NULL
            DO UPDATE SET message = EXCLUDED.message,
                          severity = EXCLUDED.severity,
                          status = 'active',
                          resolved_at = NULL
            """,
            (
                user_id,
                alert_type,
                message,
                "high" if alert_type == "budget_actual" else "medium",
                scope,
                category_id,
                month_start,
                dedupe_key,
            ),
        )

    if has_overall_budget and actual_month > overall_limit:
        add_alert(
            "budget_actual",
            "overall",
            None,
            f"Actual spending INR {actual_month:,.2f} exceeded net spendable "
            f"amount INR {overall_limit:,.2f} after recurring obligations.",
        )
    elif (
        has_overall_budget
        and projected_overall is not None
        and projected_overall > overall_limit
    ):
        add_alert(
            "budget_predicted",
            "overall",
            None,
            f"Projected spending INR {projected_overall:,.2f} may exceed net "
            f"spendable amount INR {overall_limit:,.2f}.",
        )

    for category, (budget, category_id) in category_budgets.items():
        actual = category_actual.get(category, 0.0)
        projection = projected_categories.get(category) if predictions_current else None
        if actual > budget:
            add_alert(
                "budget_actual",
                f"category:{category_id}",
                category_id,
                f"Actual {category} spending INR {actual:,.2f} exceeded its "
                f"INR {budget:,.2f} monthly budget.",
            )
        elif projection is not None and projection > budget:
            add_alert(
                "budget_predicted",
                f"category:{category_id}",
                category_id,
                f"Projected {category} spending INR {projection:,.2f} may exceed "
                f"its INR {budget:,.2f} monthly budget.",
            )

    cursor.execute(
        """
        SELECT dedupe_key FROM alerts
        WHERE user_id = %s AND alert_type LIKE 'budget_%%'
          AND target_period = %s AND status = 'active'
        """,
        (user_id, month_start),
    )
    stale_keys = [
        row["dedupe_key"]
        for row in cursor.fetchall()
        if row["dedupe_key"] and row["dedupe_key"] not in desired
    ]
    if stale_keys:
        cursor.execute(
            """
            UPDATE alerts
            SET status = 'resolved', resolved_at = NOW()
            WHERE user_id = %s AND dedupe_key = ANY(%s) AND status = 'active'
            """,
            (user_id, stale_keys),
        )
    return projected_categories, projected_overall


def evaluate_actual_financial_events(cursor, user_id):
    today = date.today()
    (
        transactions,
        account_created_at,
        preference,
        budgets,
        recurring,
        category_ids,
    ) = _load_financial_context(cursor, user_id, today)
    totals = monthly_financial_totals(
        preference["monthly_budget"] if preference else 0,
        recurring,
        transactions,
        today,
    )
    _persist_budget_alerts(
        cursor,
        user_id,
        today,
        totals,
        transactions,
        budgets,
        has_overall_budget=preference is not None,
    )
    selected, _ = select_user_history(transactions, account_created_at)
    _persist_anomalies(
        cursor,
        user_id,
        detect_transaction_anomalies(selected),
        category_ids,
    )
    return transactions, account_created_at


def evaluate_predicted_financial_events(
    cursor,
    user_id,
    overall_result,
    category_result,
    prediction_ids,
    account_created_at,
):
    today = date.today()
    (
        transactions,
        _,
        preference,
        budgets,
        recurring,
        category_ids,
    ) = _load_financial_context(cursor, user_id, today)
    totals = monthly_financial_totals(
        preference["monthly_budget"] if preference else 0,
        recurring,
        transactions,
        today,
    )
    diagnostics = category_result.get("diagnostics", {})
    is_fresh = not diagnostics.get("is_stale", True)
    overall_diagnostics = overall_result.get("diagnostics", {})
    overall_is_fresh = not overall_diagnostics.get("is_stale", True)
    _persist_budget_alerts(
        cursor,
        user_id,
        today,
        totals,
        transactions,
        budgets,
        category_result,
        predicted_overall=overall_result,
        predictions_current=is_fresh,
        overall_predictions_current=overall_is_fresh,
        has_overall_budget=preference is not None,
    )
    selected, history_diagnostics = select_user_history(
        transactions, account_created_at
    )
    anomalies = detect_transaction_anomalies(selected)
    if is_fresh:
        anomalies += detect_prediction_anomalies(
            transactions,
            category_result,
            prediction_ids,
            history_diagnostics,
        )
    _persist_anomalies(
        cursor,
        user_id,
        anomalies,
        category_ids,
    )


def reevaluate_cached_predictions(cursor, user_id):
    cursor.execute(
        "SELECT created_at FROM users WHERE user_id = %s",
        (user_id,),
    )
    user = cursor.fetchone()
    cursor.execute(
        """
        SELECT overall_result, category_result, source_transaction_id
        FROM inference_runs WHERE user_id = %s
        """,
        (user_id,),
    )
    run = cursor.fetchone()
    cursor.execute(
        """
        SELECT COALESCE(MAX(transaction_id), 0) AS latest_transaction_id
        FROM transactions WHERE user_id = %s
        """,
        (user_id,),
    )
    latest_transaction_id = cursor.fetchone()["latest_transaction_id"]
    if (
        not run
        or run["source_transaction_id"] != latest_transaction_id
        or run["overall_result"].get("target_unit") != "INR per calendar month"
        or run["category_result"].get("target_unit") != "INR per calendar month"
    ):
        return
    cursor.execute(
        """
        SELECT p.prediction_id, p.target_period, c.category_name AS category
        FROM predictions p
        JOIN categories c ON c.category_id = p.category_id
        WHERE p.user_id = %s AND p.prediction_type = 'category_monthly'
        """,
        (user_id,),
    )
    prediction_ids = {
        (row["category"], row["target_period"]): row["prediction_id"]
        for row in cursor.fetchall()
    }
    evaluate_predicted_financial_events(
        cursor,
        user_id,
        run["overall_result"],
        run["category_result"],
        prediction_ids,
        user["created_at"] if user else None,
    )


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
                recurring_rows = parse_recurring_payments(
                    request.form,
                    category_ids,
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
                for payment in recurring_rows:
                    cursor.execute(
                        """
                        INSERT INTO recurring_payments (
                            user_id, category_id, amount, frequency, next_due_date,
                            description, is_active, payment_type
                        )
                        VALUES (%s, %s, %s, 'monthly', %s, %s, TRUE, %s)
                        """,
                        (
                            session["user_id"],
                            payment["category_id"],
                            payment["amount"],
                            payment["next_due_date"],
                            payment["description"],
                            payment["payment_type"],
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
                evaluate_actual_financial_events(cursor, session["user_id"])
                reevaluate_cached_predictions(cursor, session["user_id"])
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
    transaction_type = str(data.get("transaction_type", "Expense")).strip().title()
    if transaction_type not in ("Expense", "Income"):
        return api_error("Transaction type must be Expense or Income.", 400)
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
                    VALUES (%s, %s, %s, %s, %s, %s, 'manual')
                    RETURNING transaction_id
                    """,
                    (
                        session["user_id"],
                        selected_category["category_id"],
                        round(amount, 2),
                        transaction_type,
                        transaction_date,
                        description or None,
                    ),
                )
                transaction_id = cursor.fetchone()["transaction_id"]
                evaluate_actual_financial_events(cursor, session["user_id"])
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


def persist_predictions(
    user_id,
    transactions,
    overall,
    category_predictions,
    latest_transaction_id,
    account_created_at,
):
    with database_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            overall_model_id = None
            category_model_id = None
            if overall.get("available"):
                overall_model_id = model_id(
                    cursor,
                    "Overall monthly LSTM",
                    "LSTM_OVERALL",
                    PROJECT_DIR / "ml" / "lstm" / "monthly" / "lstm_model.pt",
                )
            if category_predictions.get("categories"):
                category_model_id = model_id(
                    cursor,
                    "Category monthly LSTM",
                    "LSTM_CATEGORY",
                    PROJECT_DIR / "ml" / "lstm_categories" / "models" / "monthly",
                )

            cursor.execute(
                """
                DELETE FROM predictions
                WHERE user_id = %s
                  AND (
                    prediction_type LIKE 'overall_transaction_step_%%'
                    OR prediction_type = 'category_daily'
                    OR prediction_type LIKE 'overall_monthly_step_%%'
                    OR prediction_type = 'category_monthly'
                  )
                """,
                (user_id,),
            )

            if overall_model_id is not None:
                target_periods = overall.get("target_periods", [])
                for step, value in enumerate(overall["values"], start=1):
                    if step > len(target_periods):
                        break
                    cursor.execute(
                        """
                        INSERT INTO predictions (
                            user_id, model_id, prediction_type, target_period,
                            predicted_value
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            user_id,
                            overall_model_id,
                            f"overall_monthly_step_{step}",
                            date.fromisoformat(target_periods[step - 1]),
                            value,
                        ),
                    )

            prediction_ids = {}
            if category_model_id is not None:
                cursor.execute("SELECT category_id, category_name FROM categories")
                ids = {name: identifier for identifier, name in cursor.fetchall()}
                target_periods = category_predictions.get("target_periods", [])
                for category, values in category_predictions["categories"].items():
                    if category not in ids:
                        continue
                    for offset, value in enumerate(values):
                        if offset >= len(target_periods):
                            break
                        target_period = date.fromisoformat(target_periods[offset])
                        cursor.execute(
                            """
                            INSERT INTO predictions (
                                user_id, model_id, prediction_type, category_id,
                                target_period, predicted_value
                            )
                            VALUES (%s, %s, 'category_monthly', %s, %s, %s)
                            RETURNING prediction_id
                            """,
                            (
                                user_id,
                                category_model_id,
                                ids[category],
                                target_period,
                                value,
                            ),
                        )
                        prediction_ids[(category, target_period)] = cursor.fetchone()[
                            "prediction_id"
                        ]

            cursor.execute(
                """
                INSERT INTO inference_runs (
                    user_id, source_transaction_id, generated_at,
                    overall_result, category_result
                )
                VALUES (%s, %s, NOW(), %s, %s)
                ON CONFLICT (user_id) DO UPDATE
                SET source_transaction_id = EXCLUDED.source_transaction_id,
                    generated_at = EXCLUDED.generated_at,
                    overall_result = EXCLUDED.overall_result,
                    category_result = EXCLUDED.category_result
                """,
                (
                    user_id,
                    latest_transaction_id,
                    Json(overall),
                    Json(category_predictions),
                ),
            )
            evaluate_predicted_financial_events(
                cursor,
                user_id,
                overall,
                category_predictions,
                prediction_ids,
                account_created_at,
            )


def build_dashboard(user_id):
    today = date.today()
    month_start = today.replace(day=1)
    six_months_ago = (pd.Timestamp(month_start) - pd.DateOffset(months=5)).date()
    with database_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            (
                transactions,
                account_created_at,
                preferences,
                budget_rows,
                recurring,
                category_ids,
            ) = _load_financial_context(cursor, user_id, today)
            latest_transaction_id = max(
                (row["transaction_id"] for row in transactions),
                default=0,
            )
            cursor.execute(
                "SELECT * FROM inference_runs WHERE user_id = %s",
                (user_id,),
            )
            inference_run = cursor.fetchone()
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
    inference_errors = []
    if (
        inference_run is not None
        and inference_run["source_transaction_id"] == latest_transaction_id
        and inference_run["overall_result"].get("target_unit")
        == "INR per calendar month"
        and inference_run["category_result"].get("target_unit")
        == "INR per calendar month"
        and inference_run["overall_result"].get("available")
        and inference_run["category_result"].get("available")
    ):
        overall = inference_run["overall_result"]
        category_predictions = inference_run["category_result"]
    else:
        try:
            overall = predict_overall_monthly(
                transactions,
                account_created_at=account_created_at,
            )
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
            logger.exception("Overall LSTM inference failed")
            overall = {
                "available": False,
                "reason": str(error),
                "values": [],
                "target_periods": [],
                "target_unit": "INR per calendar month",
                "diagnostics": {},
            }
            inference_errors.append(str(error))
        try:
            category_predictions = predict_category_monthly(
                transactions,
                account_created_at=account_created_at,
            )
        except (FileNotFoundError, ValueError, RuntimeError, OSError, KeyError) as error:
            logger.exception("Category LSTM inference failed")
            category_predictions = {
                "available": False,
                "reason": str(error),
                "categories": {},
                "errors": [str(error)],
                "target_periods": [],
                "target_unit": "INR per calendar month",
                "diagnostics": {},
            }
            inference_errors.append(str(error))
        persist_predictions(
            user_id,
            transactions,
            overall,
            category_predictions,
            latest_transaction_id,
            account_created_at,
        )

    with database_connection() as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            reevaluate_cached_predictions(cursor, user_id)
            cursor.execute(
                """
                SELECT a.alert_type, a.severity, a.scope, a.message, a.status,
                       a.target_period, a.created_at, c.category_name AS category
                FROM alerts a
                LEFT JOIN categories c ON c.category_id = a.category_id
                WHERE a.user_id = %s AND a.status = 'active'
                ORDER BY a.created_at DESC
                """,
                (user_id,),
            )
            stored_alerts = cursor.fetchall()
            cursor.execute(
                """
                SELECT a.scope, a.detection_type, a.target_period,
                       a.observed_amount, a.threshold_amount, a.reason,
                       a.detected_at, c.category_name AS category
                FROM anomalies a
                LEFT JOIN categories c ON c.category_id = a.category_id
                WHERE a.user_id = %s AND a.status = 'active'
                ORDER BY a.detected_at DESC
                LIMIT 50
                """,
                (user_id,),
            )
            stored_anomalies = cursor.fetchall()

    category_diagnostics = category_predictions.get("diagnostics", {})
    history_end = category_diagnostics.get("history_end_date")
    category_diagnostics["is_stale"] = (
        not history_end
        or (today - date.fromisoformat(history_end)).days > 7
    )
    totals = monthly_financial_totals(
        preferences["monthly_budget"] if preferences else 0,
        recurring,
        transactions,
        today,
    )
    actual_month = totals["actual_monthly_spending"]
    category_actual = {}
    for transaction in transactions:
        if (
            transaction["transaction_type"] == "Expense"
            and month_start <= transaction["transaction_date"] <= today
        ):
            category_actual[transaction["category"]] = (
                category_actual.get(transaction["category"], 0.0)
                + float(transaction["amount"])
            )
    category_budgets = {
        row["category"]: float(row["budget_amount"]) for row in budget_rows
    }
    predicted_categories = {}
    category_target_periods = category_predictions.get("target_periods", [])
    for name, monthly_values in category_predictions.get("categories", {}).items():
        actual_amount = category_actual.get(name, 0.0)
        projection = None
        if (
            monthly_values
            and not category_diagnostics["is_stale"]
            and month_start.isoformat() in category_target_periods
        ):
            projection = monthly_values[
                category_target_periods.index(month_start.isoformat())
            ]
        predicted_categories[name] = {
            "monthly_forecasts": [
                {
                    "target_period": target_period,
                    "predicted_amount": value,
                }
                for target_period, value in zip(
                    category_target_periods, monthly_values
                )
            ],
            "actual_month": round(actual_amount, 2),
            "budget": category_budgets.get(name),
            "projected_month": round(projection, 2) if projection is not None else None,
            "projected_month_period": (
                month_start.isoformat() if projection is not None else None
            ),
        }

    overall_diagnostics = overall.get("diagnostics", {})
    overall_is_stale = overall_diagnostics.get("is_stale", True)
    overall_target_periods = overall.get("target_periods", [])
    predicted_month = None
    if (
        overall.get("available")
        and not overall_is_stale
        and month_start.isoformat() in overall_target_periods
    ):
        predicted_month = round(
            float(overall["values"][overall_target_periods.index(month_start.isoformat())]),
            2,
        )
    budget_limit = totals["net_after_recurring_obligations"]
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
    alerts = [
        {
            "severity": "red" if row["alert_type"] == "budget_actual" else "yellow",
            "scope": row["category"] or row["scope"] or "Overall budget",
            "message": row["message"],
            "status": row["status"],
            "created_at": row["created_at"].isoformat(),
        }
        for row in stored_alerts
    ]
    recurring_summary = [
        {
            "payment_type": row["payment_type"],
            "category": next(
                (
                    name
                    for name, category_id in category_ids.items()
                    if category_id == row["category_id"]
                ),
                "Uncategorized",
            ),
            "description": row["description"],
            "amount": round(float(row["amount"]), 2),
            "frequency": row["frequency"],
            "next_due_date": row["next_due_date"].isoformat(),
            "is_active": row["is_active"],
        }
        for row in recurring
    ]
    anomalies = [
        {
            "scope": row["category"] or row["scope"] or "Overall spending",
            "type": row["detection_type"] or "actual",
            "target_period": (
                row["target_period"].isoformat() if row["target_period"] else None
            ),
            "observed_amount": (
                round(float(row["observed_amount"]), 2)
                if row["observed_amount"] is not None
                else None
            ),
            "threshold_amount": (
                round(float(row["threshold_amount"]), 2)
                if row["threshold_amount"] is not None
                else None
            ),
            "reason": row["reason"],
            "detected_at": row["detected_at"].isoformat(),
        }
        for row in stored_anomalies
    ]
    return {
        "month": month_start.isoformat(),
        "currency": "INR",
        **totals,
        "monthly_budget": round(float(preferences["monthly_budget"]), 2)
        if preferences
        else None,
        "budget_alert_limit": round(budget_limit, 2),
        "projected_monthly_spending": predicted_month,
        "projection_method": (
            f"Overall LSTM forecast for {month_start:%B %Y}, using the previous "
            "10 complete calendar-month expense totals."
            if predicted_month is not None
            else (
                overall.get("reason")
                or "No overall monthly prediction is available for the current calendar month."
            )
        ),
        "overall_lstm": overall,
        "category_predictions": predicted_categories,
        "category_inference": {
            "available": category_predictions.get("available", False),
            "reason": category_predictions.get("reason"),
            "errors": category_predictions.get("errors", []),
            "diagnostics": category_diagnostics,
        },
        "category_spending_six_months": pie,
        "transactions": transactions_recent,
        "recurring_payments": recurring_summary,
        "alerts": alerts,
        "anomalies": anomalies,
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
