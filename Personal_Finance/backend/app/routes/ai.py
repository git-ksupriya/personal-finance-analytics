from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.routes.auth import get_current_user
from backend.app.services.database import get_db_connection
from backend.app.services.lstm_services import predict_user_spending


router = APIRouter(prefix="/ai", tags=["AI and Predictions"])


# ==================================================
# GET SAVED PREDICTIONS
# ==================================================

@router.get("/predictions")
def get_predictions(
    days: int = Query(default=7, ge=1, le=30),
    prediction_type: str = Query(
        default="overall",
        pattern="^(overall|category)$",
    ),
    user=Depends(get_current_user),
):
    """
    Retrieve saved predictions for the authenticated user.

    Expected table:
        spending_predictions

    This endpoint assumes that table contains:
        user_id
        forecast_date
        prediction_type
        predicted_amount
    """

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM spending_predictions
                WHERE user_id = %s
                  AND forecast_date >= %s
                  AND forecast_date < %s
                  AND prediction_type = %s
                ORDER BY forecast_date ASC
                """,
                (
                    user["user_id"],
                    date.today(),
                    date.today() + timedelta(days=days),
                    prediction_type,
                ),
            )

            rows = cursor.fetchall()

            predictions = []

            for row in rows:
                item = dict(row)

                if item.get("predicted_amount") is not None:
                    item["predicted_amount"] = float(
                        item["predicted_amount"]
                    )

                for key, value in item.items():
                    if hasattr(value, "isoformat"):
                        item[key] = value.isoformat()

                predictions.append(item)

            return {
                "user_id": user["user_id"],
                "prediction_type": prediction_type,
                "days": days,
                "predictions": predictions,
                "count": len(predictions),
            }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not retrieve saved predictions. "
                "Verify the spending_predictions table schema."
            ),
        ) from exc

    finally:
        connection.close()


# ==================================================
# GENERATE LSTM PREDICTIONS
# ==================================================

@router.post("/predict")
def predict_spending(
    user=Depends(get_current_user),
):
    """
    Generate a seven-day spending forecast
    for the authenticated user.

    The LSTM service currently prepares the input
    but inference must be enabled after verifying
    the final trained model and scaler.
    """

    try:
        result = predict_user_spending(
            user_id=user["user_id"]
        )

        return {
            "message": "Spending forecast generated successfully",
            "user_id": user["user_id"],
            "forecast_days": result.get("forecast_days", 7),
            "start_date": result.get("start_date"),
            "end_date": result.get("end_date"),
            "total_predicted_expenditure": result.get(
                "total_predicted_expenditure", 0.0
            ),
            "predictions": result.get("predictions", []),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate the spending forecast.",
        ) from exc


# ==================================================
# WEEKLY BUDGET RISK
# ==================================================

@router.get("/budget-risk")
def get_budget_risk(
    user=Depends(get_current_user),
):
    """
    Compare a user's weekly budget with actual
    expenditure and available future forecasts.

    Requires a weekly_budgets table and compatible
    saved predictions.
    """

    connection = get_db_connection()
    today = date.today()

    try:
        with connection.cursor() as cursor:

            # Find the current overall weekly budget.
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
                ORDER BY start_date DESC
                LIMIT 1
                """,
                (
                    user["user_id"],
                    today,
                    today,
                ),
            )

            budget_row = cursor.fetchone()

            if not budget_row:
                return {
                    "user_id": user["user_id"],
                    "status": "no_budget",
                    "message": "No active overall weekly budget found.",
                }

            budget = dict(budget_row)

            budget_start = budget["start_date"]
            budget_end = budget["end_date"]

            # Actual spending through today.
            cursor.execute(
                """
                SELECT COALESCE(SUM(amount), 0) AS actual_spending
                FROM transactions
                WHERE user_id = %s
                  AND transaction_type = 'Expense'
                  AND transaction_date >= %s
                  AND transaction_date <= %s
                """,
                (
                    user["user_id"],
                    budget_start,
                    today,
                ),
            )

            actual_row = cursor.fetchone()

            actual_spending = float(
                actual_row["actual_spending"]
            )

            # Only count forecasts for future days
            # within the budget's remaining period.
            future_start = today + timedelta(days=1)

            future_predictions = []

            if future_start <= budget_end:
                cursor.execute(
                    """
                    SELECT
                        forecast_date,
                        predicted_amount
                    FROM spending_predictions
                    WHERE user_id = %s
                      AND prediction_type = 'overall'
                      AND forecast_date >= %s
                      AND forecast_date <= %s
                    ORDER BY forecast_date ASC
                    """,
                    (
                        user["user_id"],
                        future_start,
                        budget_end,
                    ),
                )

                future_predictions = [
                    dict(row)
                    for row in cursor.fetchall()
                ]

            forecast_spending = sum(
                float(row["predicted_amount"])
                for row in future_predictions
                if row["predicted_amount"] is not None
            )

            budget_amount = float(
                budget["budget_amount"]
            )

            projected_total = (
                actual_spending + forecast_spending
            )

            remaining_budget = (
                budget_amount - projected_total
            )

            if not future_predictions:
                risk_status = "forecast_unavailable"
                message = (
                    "Actual spending is available, but future "
                    "predictions are missing for the remaining "
                    "budget period."
                )

            elif projected_total > budget_amount:
                risk_status = "over_budget"
                message = (
                    "Your projected spending exceeds your "
                    "weekly budget."
                )

            elif projected_total >= budget_amount * 0.8:
                risk_status = "high_risk"
                message = (
                    "Your projected spending is approaching "
                    "your weekly budget."
                )

            else:
                risk_status = "within_budget"
                message = (
                    "Your projected spending is within "
                    "your weekly budget."
                )

            return {
                "user_id": user["user_id"],
                "budget_id": budget["weekly_budget_id"],
                "budget_amount": budget_amount,
                "start_date": budget_start.isoformat(),
                "end_date": budget_end.isoformat(),
                "actual_spending": round(actual_spending, 2),
                "forecast_spending": round(forecast_spending, 2),
                "projected_total": round(projected_total, 2),
                "remaining_budget": round(remaining_budget, 2),
                "risk_status": risk_status,
                "message": message,
                "forecast_days_available": len(
                    future_predictions
                ),
            }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not calculate budget risk. "
                "Verify the weekly budget and prediction schemas."
            ),
        ) from exc

    finally:
        connection.close()


# ==================================================
# GET USER ALERTS
# ==================================================

@router.get("/alerts")
def get_alerts(
    unread_only: bool = False,
    user=Depends(get_current_user),
):
    """
    Retrieve alerts belonging to the authenticated user.

    Expected table:
        alerts

    Expected columns:
        alert_id, user_id, alert_type, message,
        is_read, created_at
    """

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:

            query = """
                SELECT *
                FROM alerts
                WHERE user_id = %s
            """

            params = [user["user_id"]]

            if unread_only:
                query += " AND is_read = FALSE"

            query += " ORDER BY created_at DESC"

            cursor.execute(query, tuple(params))

            rows = cursor.fetchall()

            alerts = []

            for row in rows:
                item = dict(row)

                for key, value in item.items():
                    if hasattr(value, "isoformat"):
                        item[key] = value.isoformat()

                alerts.append(item)

            return {
                "user_id": user["user_id"],
                "alerts": alerts,
                "count": len(alerts),
            }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve alerts.",
        ) from exc

    finally:
        connection.close()


# ==================================================
# MARK AN ALERT AS READ
# ==================================================

@router.patch("/alerts/{alert_id}/read")
def mark_alert_as_read(
    alert_id: int,
    user=Depends(get_current_user),
):
    """
    Mark an alert as read only if it belongs
    to the authenticated user.
    """

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE alerts
                SET is_read = TRUE
                WHERE alert_id = %s
                  AND user_id = %s
                RETURNING alert_id, is_read
                """,
                (
                    alert_id,
                    user["user_id"],
                ),
            )

            row = cursor.fetchone()

            if not row:
                connection.rollback()

                raise HTTPException(
                    status_code=404,
                    detail="Alert not found.",
                )

        connection.commit()

        return {
            "message": "Alert marked as read.",
            "alert_id": row["alert_id"],
            "is_read": row["is_read"],
        }

    except HTTPException:
        raise

    except Exception as exc:
        connection.rollback()

        raise HTTPException(
            status_code=500,
            detail="Could not update the alert.",
        ) from exc

    finally:
        connection.close()

