from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.services.database import get_db_connection
from backend.app.routes.auth import get_current_user


router = APIRouter(prefix="/analytics", tags=["Analytics"])


# --------------------------------------------------
# DATE HELPERS
# --------------------------------------------------

def get_date_range(days: int):
    end_date = date.today()
    start_date = end_date - timedelta(days=days - 1)
    return start_date, end_date


# --------------------------------------------------
# DASHBOARD SUMMARY
# --------------------------------------------------

@router.get("/summary")
def dashboard_summary(
    days: int = Query(default=7, ge=1, le=365),
    user=Depends(get_current_user),
):
    """
    Return total income, expenses, and transaction count
    for the requested number of days.
    """

    start_date, end_date = get_date_range(days)

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    COALESCE(
                        SUM(amount) FILTER (
                            WHERE transaction_type = 'Income'
                        ),
                        0
                    ) AS total_income,

                    COALESCE(
                        SUM(amount) FILTER (
                            WHERE transaction_type = 'Expense'
                        ),
                        0
                    ) AS total_expenses,

                    COUNT(*) AS transaction_count

                FROM transactions

                WHERE user_id = %s
                  AND transaction_date BETWEEN %s AND %s
                """,
                (user["user_id"], start_date, end_date),
            )

            row = cursor.fetchone()

            income = float(row["total_income"])
            expenses = float(row["total_expenses"])

            return {
                "period": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "days": days,
                },
                "total_income": income,
                "total_expenses": expenses,
                "balance": income - expenses,
                "transaction_count": row["transaction_count"],
            }

    finally:
        connection.close()


# --------------------------------------------------
# DAILY SPENDING CHART
# --------------------------------------------------

@router.get("/daily-spending")
def daily_spending(
    days: int = Query(default=7, ge=1, le=365),
    user=Depends(get_current_user),
):
    """
    Return one record per day, including days with zero expenditure.
    """

    start_date, end_date = get_date_range(days)

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH calendar AS (
                    SELECT generate_series(
                        %s::date,
                        %s::date,
                        INTERVAL '1 day'
                    )::date AS spending_date
                ),

                daily_totals AS (
                    SELECT
                        transaction_date,
                        COALESCE(
                            SUM(amount) FILTER (
                                WHERE transaction_type = 'Expense'
                            ),
                            0
                        ) AS expenses,

                        COALESCE(
                            SUM(amount) FILTER (
                                WHERE transaction_type = 'Income'
                            ),
                            0
                        ) AS income

                    FROM transactions

                    WHERE user_id = %s
                      AND transaction_date BETWEEN %s AND %s

                    GROUP BY transaction_date
                )

                SELECT
                    calendar.spending_date,
                    COALESCE(daily_totals.expenses, 0) AS expenses,
                    COALESCE(daily_totals.income, 0) AS income

                FROM calendar

                LEFT JOIN daily_totals
                    ON calendar.spending_date =
                       daily_totals.transaction_date

                ORDER BY calendar.spending_date
                """,
                (
                    start_date,
                    end_date,
                    user["user_id"],
                    start_date,
                    end_date,
                ),
            )

            records = []

            for row in cursor.fetchall():
                records.append({
                    "date": row["spending_date"],
                    "expenses": float(row["expenses"]),
                    "income": float(row["income"]),
                })

            return {
                "daily_spending": records,
            }

    finally:
        connection.close()


# --------------------------------------------------
# CATEGORY BREAKDOWN
# --------------------------------------------------

@router.get("/category-breakdown")
def category_breakdown(
    days: int = Query(default=7, ge=1, le=365),
    user=Depends(get_current_user),
):
    """Return the authenticated user's expenditure by category."""

    start_date, end_date = get_date_range(days)

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    c.category_id,
                    c.category_name,
                    COALESCE(SUM(t.amount), 0) AS total_spending,
                    COUNT(t.transaction_id) AS transaction_count

                FROM categories c

                LEFT JOIN transactions t
                    ON t.category_id = c.category_id
                   AND t.user_id = %s
                   AND t.transaction_type = 'Expense'
                   AND t.transaction_date BETWEEN %s AND %s

                GROUP BY c.category_id, c.category_name

                HAVING COALESCE(SUM(t.amount), 0) > 0

                ORDER BY total_spending DESC
                """,
                (user["user_id"], start_date, end_date),
            )

            results = [
                {
                    "category_id": row["category_id"],
                    "category_name": row["category_name"],
                    "total_spending": float(row["total_spending"]),
                    "transaction_count": row["transaction_count"],
                }
                for row in cursor.fetchall()
            ]

            return {
                "start_date": start_date,
                "end_date": end_date,
                "categories": results,
            }

    finally:
        connection.close()


# --------------------------------------------------
# MONTHLY CATEGORY BUDGET COMPARISON
# --------------------------------------------------

@router.get("/budget-comparison")
def budget_comparison(
    month: date | None = None,
    user=Depends(get_current_user),
):
    """
    Compare monthly category budgets with actual expenditure.

    The month parameter may be any date within the desired month.
    If omitted, the current month is used.
    """

    selected_date = month or date.today()

    month_start = selected_date.replace(day=1)

    if month_start.month == 12:
        next_month = month_start.replace(
            year=month_start.year + 1,
            month=1,
        )
    else:
        next_month = month_start.replace(
            month=month_start.month + 1,
        )

    month_end = next_month - timedelta(days=1)

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    b.budget_id,
                    b.category_id,
                    c.category_name,
                    b.budget_amount,

                    COALESCE(
                        SUM(t.amount) FILTER (
                            WHERE t.transaction_type = 'Expense'
                        ),
                        0
                    ) AS actual_spending

                FROM budgets b

                JOIN categories c
                    ON c.category_id = b.category_id

                LEFT JOIN transactions t
                    ON t.user_id = b.user_id
                   AND t.category_id = b.category_id
                   AND t.transaction_date BETWEEN %s AND %s

                WHERE b.user_id = %s
                  AND b.month = %s

                GROUP BY
                    b.budget_id,
                    b.category_id,
                    c.category_name,
                    b.budget_amount

                ORDER BY c.category_name
                """,
                (
                    month_start,
                    month_end,
                    user["user_id"],
                    month_start,
                ),
            )

            comparisons = []

            for row in cursor.fetchall():
                budget = float(row["budget_amount"])
                actual = float(row["actual_spending"])

                comparisons.append({
                    "budget_id": row["budget_id"],
                    "category_id": row["category_id"],
                    "category_name": row["category_name"],
                    "budget_amount": budget,
                    "actual_spending": actual,
                    "remaining_budget": budget - actual,
                    "percentage_used": (
                        round(actual / budget * 100, 2)
                        if budget > 0
                        else None
                    ),
                    "is_over_budget": actual > budget,
                })

            return {
                "month": month_start,
                "categories": comparisons,
            }

    finally:
        connection.close()


# --------------------------------------------------
# RECENT TRANSACTIONS
# --------------------------------------------------

@router.get("/recent-transactions")
def recent_transactions(
    limit: int = Query(default=10, ge=1, le=100),
    user=Depends(get_current_user),
):
    """Return the most recent transactions for the logged-in user."""

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
                    pm.mode_name AS payment_mode

                FROM transactions t

                JOIN categories c
                    ON c.category_id = t.category_id

                LEFT JOIN payment_modes pm
                    ON pm.payment_mode_id = t.payment_mode_id

                WHERE t.user_id = %s

                ORDER BY
                    t.transaction_date DESC,
                    t.transaction_id DESC

                LIMIT %s
                """,
                (user["user_id"], limit),
            )

            transactions = [
                dict(row)
                for row in cursor.fetchall()
            ]

            return {
                "transactions": transactions,
            }

    finally:
        connection.close()


# --------------------------------------------------
# WEEKLY SPENDING SUMMARY
# --------------------------------------------------

@router.get("/weekly-summary")
def weekly_summary(user=Depends(get_current_user)):
    """
    Summarize expenditure over the latest seven calendar days
    and compare it with the applicable weekly overall budget.

    The comparison uses the budget's own date range.
    """

    today = date.today()
    seven_days_ago = today - timedelta(days=6)

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    COALESCE(SUM(amount), 0) AS total_spending

                FROM transactions

                WHERE user_id = %s
                  AND transaction_type = 'Expense'
                  AND transaction_date BETWEEN %s AND %s
                """,
                (
                    user["user_id"],
                    seven_days_ago,
                    today,
                ),
            )

            actual_spending = float(
                cursor.fetchone()["total_spending"]
            )

            cursor.execute(
                """
                SELECT
                    weekly_budget_id,
                    budget_amount,
                    start_date,
                    end_date

                FROM weekly_budgets

                WHERE user_id = %s
                  AND budget_type = 'overall'
                  AND start_date <= %s
                  AND end_date >= %s

                ORDER BY start_date DESC, weekly_budget_id DESC

                LIMIT 1
                """,
                (
                    user["user_id"],
                    today,
                    today,
                ),
            )

            budget_row = cursor.fetchone()

            if budget_row:
                budget_amount = float(budget_row["budget_amount"])

                return {
                    "start_date": seven_days_ago,
                    "end_date": today,
                    "actual_spending": actual_spending,
                    "budget_amount": budget_amount,
                    "remaining_budget": (
                        budget_amount - actual_spending
                    ),
                    "percentage_used": (
                        round(
                            actual_spending / budget_amount * 100,
                            2,
                        )
                        if budget_amount > 0
                        else None
                    ),
                    "is_over_budget": actual_spending > budget_amount,
                    "budget_period": {
                        "start_date": budget_row["start_date"],
                        "end_date": budget_row["end_date"],
                    },
                }

            return {
                "start_date": seven_days_ago,
                "end_date": today,
                "actual_spending": actual_spending,
                "budget_amount": None,
                "remaining_budget": None,
                "percentage_used": None,
                "is_over_budget": None,
                "message": "No applicable weekly overall budget has been set.",
            }

    finally:
        connection.close()
