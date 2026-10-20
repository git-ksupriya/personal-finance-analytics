import csv
import io
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from pydantic import BaseModel, Field

from backend.app.services.database import get_db_connection
from backend.app.routes.auth import get_current_user


router = APIRouter(prefix="/transactions", tags=["Transactions"])


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class TransactionRequest(BaseModel):
    amount: Decimal = Field(gt=0, max_digits=12, decimal_places=2)
    transaction_type: Literal["Expense", "Income"]
    category_id: int = Field(gt=0)
    transaction_date: date
    payment_mode_id: int | None = Field(default=None, gt=0)
    location_id: int | None = Field(default=None, gt=0)
    account_id: int | None = Field(default=None, gt=0)
    description: str | None = Field(default=None, max_length=2000)


# --------------------------------------------------
# LOOKUP APIs
# --------------------------------------------------

@router.get("/categories")
def get_categories(user=Depends(get_current_user)):
    """Return available transaction categories."""

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT category_id, category_name
                FROM categories
                ORDER BY category_name
                """
            )

            return {"categories": [dict(row) for row in cursor.fetchall()]}

    finally:
        connection.close()


@router.get("/payment-modes")
def get_payment_modes(user=Depends(get_current_user)):
    """Return available payment modes."""

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT payment_mode_id, mode_name
                FROM payment_modes
                ORDER BY mode_name
                """
            )

            return {
                "payment_modes": [
                    dict(row) for row in cursor.fetchall()
                ]
            }

    finally:
        connection.close()


@router.get("/locations")
def get_locations(user=Depends(get_current_user)):
    """Return available locations."""

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT location_id, location_name
                FROM locations
                ORDER BY location_name
                """
            )

            return {"locations": [dict(row) for row in cursor.fetchall()]}

    finally:
        connection.close()


# --------------------------------------------------
# ADD A TRANSACTION
# --------------------------------------------------

@router.post("/", status_code=201)
def add_transaction(
    data: TransactionRequest,
    user=Depends(get_current_user),
):
    """Add a transaction for the authenticated user."""

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:

            # Validate the category.
            cursor.execute(
                """
                SELECT 1
                FROM categories
                WHERE category_id = %s
                """,
                (data.category_id,),
            )

            if not cursor.fetchone():
                raise HTTPException(
                    status_code=400,
                    detail="Invalid category",
                )

            # Validate optional payment mode.
            if data.payment_mode_id is not None:
                cursor.execute(
                    """
                    SELECT 1
                    FROM payment_modes
                    WHERE payment_mode_id = %s
                    """,
                    (data.payment_mode_id,),
                )

                if not cursor.fetchone():
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid payment mode",
                    )

            # Validate optional location.
            if data.location_id is not None:
                cursor.execute(
                    """
                    SELECT 1
                    FROM locations
                    WHERE location_id = %s
                    """,
                    (data.location_id,),
                )

                if not cursor.fetchone():
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid location",
                    )

            # An account, if supplied, must belong to this user.
            if data.account_id is not None:
                cursor.execute(
                    """
                    SELECT 1
                    FROM accounts
                    WHERE account_id = %s
                      AND user_id = %s
                    """,
                    (data.account_id, user["user_id"]),
                )

                if not cursor.fetchone():
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid account",
                    )

            cursor.execute(
                """
                INSERT INTO transactions (
                    user_id,
                    account_id,
                    category_id,
                    payment_mode_id,
                    location_id,
                    amount,
                    transaction_type,
                    transaction_date,
                    description,
                    source
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'manual')
                RETURNING transaction_id, user_id, category_id,
                          amount, transaction_type, transaction_date,
                          description
                """,
                (
                    user["user_id"],
                    data.account_id,
                    data.category_id,
                    data.payment_mode_id,
                    data.location_id,
                    data.amount,
                    data.transaction_type,
                    data.transaction_date,
                    data.description,
                ),
            )

            transaction = dict(cursor.fetchone())

        connection.commit()

        return {
            "message": "Transaction added successfully",
            "transaction": transaction,
        }

    except HTTPException:
        connection.rollback()
        raise

    except Exception:
        connection.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not add transaction",
        )

    finally:
        connection.close()


# --------------------------------------------------
# TRANSACTION HISTORY
# --------------------------------------------------

@router.get("/")
def get_transactions(
    transaction_type: Literal["Expense", "Income"] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    user=Depends(get_current_user),
):
    """Retrieve only the authenticated user's transactions."""

    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date must not be after end_date",
        )

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:

            query = """
                SELECT
                    t.transaction_id,
                    t.amount,
                    t.transaction_type,
                    t.transaction_date,
                    t.description,
                    t.source,
                    c.category_name,
                    pm.mode_name AS payment_mode,
                    l.location_name
                FROM transactions t
                JOIN categories c
                    ON t.category_id = c.category_id
                LEFT JOIN payment_modes pm
                    ON t.payment_mode_id = pm.payment_mode_id
                LEFT JOIN locations l
                    ON t.location_id = l.location_id
                WHERE t.user_id = %s
            """

            params = [user["user_id"]]

            if transaction_type:
                query += " AND t.transaction_type = %s"
                params.append(transaction_type)

            if start_date:
                query += " AND t.transaction_date >= %s"
                params.append(start_date)

            if end_date:
                query += " AND t.transaction_date <= %s"
                params.append(end_date)

            query += """
                ORDER BY t.transaction_date DESC, t.transaction_id DESC
                LIMIT %s OFFSET %s
            """

            params.extend([limit, offset])

            cursor.execute(query, tuple(params))
            transactions = [
                dict(row) for row in cursor.fetchall()
            ]

            cursor.execute(
                """
                SELECT COUNT(*) AS total
                FROM transactions
                WHERE user_id = %s
                """,
                (user["user_id"],),
            )

            total = cursor.fetchone()["total"]

        return {
            "transactions": transactions,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    finally:
        connection.close()


# --------------------------------------------------
# SINGLE TRANSACTION
# --------------------------------------------------

@router.get("/{transaction_id}")
def get_transaction(
    transaction_id: int,
    user=Depends(get_current_user),
):
    """Retrieve one transaction belonging to the authenticated user."""

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    t.transaction_id,
                    t.amount,
                    t.transaction_type,
                    t.transaction_date,
                    t.description,
                    t.source,
                    c.category_name,
                    pm.mode_name AS payment_mode,
                    l.location_name
                FROM transactions t
                JOIN categories c
                    ON t.category_id = c.category_id
                LEFT JOIN payment_modes pm
                    ON t.payment_mode_id = pm.payment_mode_id
                LEFT JOIN locations l
                    ON t.location_id = l.location_id
                WHERE t.transaction_id = %s
                  AND t.user_id = %s
                """,
                (transaction_id, user["user_id"]),
            )

            transaction = cursor.fetchone()

            if not transaction:
                raise HTTPException(
                    status_code=404,
                    detail="Transaction not found",
                )

            return {"transaction": dict(transaction)}

    finally:
        connection.close()


# --------------------------------------------------
# CSV IMPORT
# --------------------------------------------------

@router.post("/import-csv", status_code=201)
def import_transactions_csv(
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    """
    Import historical transactions.

    Supported column names:
      amount, transaction_type, transaction_date, category

    Optional columns:
      payment_mode, location, description

    The logged-in user's ID is assigned to every imported row.
    A user_id column in the uploaded file is never trusted.
    """

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file",
        )

    try:
        raw_content = file.file.read(5 * 1024 * 1024 + 1)

        if len(raw_content) > 5 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="CSV file must not exceed 5 MB",
            )

        text_content = raw_content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text_content))

        if not reader.fieldnames:
            raise HTTPException(
                status_code=400,
                detail="CSV file has no header row",
            )

        # Ignore surrounding spaces and compare headers case-insensitively.
        reader.fieldnames = [
            (name or "").strip().lower()
            for name in reader.fieldnames
        ]

        required_columns = {
            "amount",
            "transaction_type",
            "transaction_date",
            "category",
        }

        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Missing required CSV columns: "
                    + ", ".join(sorted(missing_columns))
                ),
            )

        rows = list(reader)

        if not rows:
            raise HTTPException(
                status_code=400,
                detail="CSV contains no transactions",
            )

        if len(rows) > 10000:
            raise HTTPException(
                status_code=400,
                detail="CSV may contain at most 10,000 rows per upload",
            )

        connection = get_db_connection()

        try:
            with connection.cursor() as cursor:

                # Create an import record.
                cursor.execute(
                    """
                    INSERT INTO statement_uploads (
                        user_id,
                        blob_url,
                        file_name,
                        file_type,
                        status
                    )
                    VALUES (%s, %s, %s, 'csv', 'pending')
                    RETURNING upload_id
                    """,
                    (
                        user["user_id"],
                        "local-upload",
                        file.filename[:255],
                    ),
                )

                upload_id = cursor.fetchone()["upload_id"]

                # Load lookup values.
                cursor.execute(
                    "SELECT category_id, category_name FROM categories"
                )
                categories = {
                    row["category_name"].strip().casefold():
                    row["category_id"]
                    for row in cursor.fetchall()
                }

                cursor.execute(
                    "SELECT payment_mode_id, mode_name FROM payment_modes"
                )
                payment_modes = {
                    row["mode_name"].strip().casefold():
                    row["payment_mode_id"]
                    for row in cursor.fetchall()
                }

                cursor.execute(
                    "SELECT location_id, location_name FROM locations"
                )
                locations = {
                    row["location_name"].strip().casefold():
                    row["location_id"]
                    for row in cursor.fetchall()
                }

                # "Unknown" is present in the existing seed data.
                unknown_payment = payment_modes.get("unknown")
                unknown_location = locations.get("unknown")

                imported = 0
                errors = []

                for row_number, row in enumerate(rows, start=2):
                    try:
                        amount = Decimal(
                            (row.get("amount") or "").strip()
                            .replace(",", "")
                            .replace("₹", "")
                            .replace("$", "")
                            .replace("Rs.", "")
                            .strip()
                        )

                        if not amount.is_finite() or amount <= 0:
                            raise ValueError(
                                "amount must be a positive number"
                            )

                        if amount > Decimal("9999999999.99"):
                            raise ValueError("amount is too large")

                        transaction_type = (
                            row.get("transaction_type") or ""
                        ).strip().casefold()

                        if transaction_type not in {"expense", "income"}:
                            raise ValueError(
                                "transaction_type must be Expense or Income"
                            )

                        transaction_date = date.fromisoformat(
                            (row.get("transaction_date") or "").strip()
                        )

                        category_name = (
                            row.get("category") or ""
                        ).strip().casefold()

                        category_id = categories.get(category_name)

                        if category_id is None:
                            raise ValueError(
                                "category is not present in the categories table"
                            )

                        payment_name = (
                            row.get("payment_mode") or ""
                        ).strip().casefold()

                        if payment_name and payment_name not in payment_modes:
                            raise ValueError(
                                "payment_mode is not recognized"
                            )

                        payment_mode_id = (
                            payment_modes.get(payment_name)
                            if payment_name
                            else unknown_payment
                        )

                        location_name = (
                            row.get("location") or ""
                        ).strip().casefold()

                        if location_name and location_name not in locations:
                            raise ValueError(
                                "location is not recognized"
                            )

                        location_id = (
                            locations.get(location_name)
                            if location_name
                            else unknown_location
                        )

                        description = (
                            row.get("description") or ""
                        ).strip() or None

                        cursor.execute(
                            """
                            INSERT INTO transactions (
                                user_id,
                                category_id,
                                payment_mode_id,
                                location_id,
                                amount,
                                transaction_type,
                                transaction_date,
                                description,
                                source,
                                upload_id
                            )
                            VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s,
                                'csv', %s
                            )
                            """,
                            (
                                user["user_id"],
                                category_id,
                                payment_mode_id,
                                location_id,
                                amount,
                                transaction_type.title(),
                                transaction_date,
                                description,
                                upload_id,
                            ),
                        )

                        imported += 1

                    except (
                        ValueError,
                        InvalidOperation,
                        TypeError,
                    ) as error:
                        errors.append({
                            "row": row_number,
                            "reason": str(error),
                        })

                # Keep successful rows and report invalid rows.
                status = "processed" if not errors else (
                    "processed" if imported else "failed"
                )

                cursor.execute(
                    """
                    UPDATE statement_uploads
                    SET status = %s,
                        processed_at = CURRENT_TIMESTAMP
                    WHERE upload_id = %s
                      AND user_id = %s
                    """,
                    (status, upload_id, user["user_id"]),
                )

            connection.commit()

            return {
                "message": "CSV import completed",
                "upload_id": upload_id,
                "imported_count": imported,
                "failed_count": len(errors),
                "errors": errors[:100],
            }

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    except HTTPException:
        raise

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="CSV must use UTF-8 encoding",
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not process CSV file",
        )

    finally:
        file.file.close()

