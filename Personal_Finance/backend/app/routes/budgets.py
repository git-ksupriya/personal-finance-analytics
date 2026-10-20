from datetime import date, timedelta
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator

from backend.app.services.database import get_db_connection
from backend.app.routes.auth import get_current_user


router = APIRouter(prefix="/budgets", tags=["Weekly Budgets"])


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class WeeklyBudgetRequest(BaseModel):
    budget_type: str = Field(pattern="^(overall|category)$")
    budget_amount: Decimal = Field(
        gt=0,
        max_digits=12,
        decimal_places=2,
    )
    category_id: int | None = Field(default=None, gt=0)
    start_date: date
    end_date: date | None = None

    @model_validator(mode="after")
    def validate_budget(self):
        if self.budget_type == "overall" and self.category_id is not None:
            raise ValueError(
                "Overall budgets must not have a category_id"
            )

        if self.budget_type == "category" and self.category_id is None:
            raise ValueError(
                "Category budgets require a category_id"
            )

        if self.end_date is not None and self.end_date < self.start_date:
            raise ValueError(
                "end_date must not be before start_date"
            )

        return self


# --------------------------------------------------
# HELPER
# --------------------------------------------------

def calculate_end_date(start_date: date, end_date: date | None):
    """
    A weekly budget lasts seven calendar days, including
    its start date and end date.
    """

    calculated_end = end_date or (
        start_date + timedelta(days=6)
    )

    if (calculated_end - start_date).days != 6:
        raise HTTPException(
            status_code=400,
            detail="A weekly budget must cover exactly seven calendar days",
        )

    return calculated_end


# --------------------------------------------------
# CREATE OR UPDATE A WEEKLY BUDGET
# --------------------------------------------------

@router.post("/", status_code=201)
def save_weekly_budget(
    data: WeeklyBudgetRequest,
    user=Depends(get_current_user),
):
    """
    Save a weekly budget.

    If the same user, budget type, category, and date range
    already exist, update the existing budget instead.
    """

    end_date = calculate_end_date(
        data.start_date,
        data.end_date,
    )

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:

            if data.category_id is not None:
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

            # Find an existing budget for the same period.
            cursor.execute(
                """
                SELECT weekly_budget_id
                FROM weekly_budgets
                WHERE user_id = %s
                  AND budget_type = %s
                  AND category_id IS NOT DISTINCT FROM %s
                  AND start_date = %s
                  AND end_date = %s
                FOR UPDATE
                """,
                (
                    user["user_id"],
                    data.budget_type,
                    data.category_id,
                    data.start_date,
                    end_date,
                ),
            )

            existing = cursor.fetchone()

            if existing:
                cursor.execute(
                    """
                    UPDATE weekly_budgets
                    SET budget_amount = %s
                    WHERE weekly_budget_id = %s
                      AND user_id = %s
                    RETURNING *
                    """,
                    (
                        data.budget_amount,
                        existing["weekly_budget_id"],
                        user["user_id"],
                    ),
                )

                budget = dict(cursor.fetchone())
                message = "Weekly budget updated successfully"

            else:
                cursor.execute(
                    """
                    INSERT INTO weekly_budgets (
                        user_id,
                        budget_type,
                        category_id,
                        budget_amount,
                        start_date,
                        end_date
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (
                        user["user_id"],
                        data.budget_type,
                        data.category_id,
                        data.budget_amount,
                        data.start_date,
                        end_date,
                    ),
                )

                budget = dict(cursor.fetchone())
                message = "Weekly budget created successfully"

        connection.commit()

        budget["budget_amount"] = float(budget["budget_amount"])

        return {
            "message": message,
            "budget": budget,
        }

    except HTTPException:
        connection.rollback()
        raise

    except Exception:
        connection.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not save weekly budget",
        )

    finally:
        connection.close()


# --------------------------------------------------
# LIST WEEKLY BUDGETS
# --------------------------------------------------

@router.get("/")
def get_weekly_budgets(
    include_past: bool = False,
    user=Depends(get_current_user),
):
    """List the authenticated user's weekly budgets."""

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:

            query = """
                SELECT
                    wb.weekly_budget_id,
                    wb.budget_type,
                    wb.category_id,
                    c.category_name,
                    wb.budget_amount,
                    wb.start_date,
                    wb.end_date,
                    wb.created_at

                FROM weekly_budgets wb

                LEFT JOIN categories c
                    ON c.category_id = wb.category_id

                WHERE wb.user_id = %s
            """

            params = [user["user_id"]]

            if not include_past:
                query += " AND wb.end_date >= %s"
                params.append(date.today())

            query += """
                ORDER BY
                    wb.start_date DESC,
                    wb.budget_type,
                    c.category_name
            """

            cursor.execute(query, tuple(params))

            budgets = []

            for row in cursor.fetchall():
                item = dict(row)
                item["budget_amount"] = float(item["budget_amount"])
                budgets.append(item)

            return {"budgets": budgets}

    finally:
        connection.close()


# --------------------------------------------------
# GET ONE WEEKLY BUDGET
# --------------------------------------------------

@router.get("/{budget_id}")
def get_weekly_budget(
    budget_id: int,
    user=Depends(get_current_user),
):
    """Retrieve one budget belonging to the authenticated user."""

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    wb.weekly_budget_id,
                    wb.budget_type,
                    wb.category_id,
                    c.category_name,
                    wb.budget_amount,
                    wb.start_date,
                    wb.end_date,
                    wb.created_at

                FROM weekly_budgets wb

                LEFT JOIN categories c
                    ON c.category_id = wb.category_id

                WHERE wb.weekly_budget_id = %s
                  AND wb.user_id = %s
                """,
                (budget_id, user["user_id"]),
            )

            row = cursor.fetchone()

            if not row:
                raise HTTPException(
                    status_code=404,
                    detail="Weekly budget not found",
                )

            budget = dict(row)
            budget["budget_amount"] = float(
                budget["budget_amount"]
            )

            return {"budget": budget}

    finally:
        connection.close()


# --------------------------------------------------
# DELETE A WEEKLY BUDGET
# --------------------------------------------------

@router.delete("/{budget_id}")
def delete_weekly_budget(
    budget_id: int,
    user=Depends(get_current_user),
):
    """Delete a budget owned by the authenticated user."""

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM weekly_budgets
                WHERE weekly_budget_id = %s
                  AND user_id = %s
                RETURNING weekly_budget_id
                """,
                (budget_id, user["user_id"]),
            )

            deleted = cursor.fetchone()

            if not deleted:
                raise HTTPException(
                    status_code=404,
                    detail="Weekly budget not found",
                )

        connection.commit()

        return {
            "message": "Weekly budget deleted successfully",
            "budget_id": deleted["weekly_budget_id"],
        }

    except HTTPException:
        connection.rollback()
        raise

    except Exception:
        connection.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not delete weekly budget",
        )

    finally:
        connection.close()
