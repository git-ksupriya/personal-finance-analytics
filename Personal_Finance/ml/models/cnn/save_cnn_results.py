
from database import get_db_connection


def save_total_predictions(
    model_id,
    actual_values,
    predicted_values
):
    conn = get_db_connection()

    try:
        with conn.cursor() as cur:

            for day in range(len(predicted_values)):

                actual = float(actual_values[day])
                predicted = float(predicted_values[day])

                absolute_error = abs(actual - predicted)

                cur.execute(
                    """
                    INSERT INTO cnn_total_spending_predictions
                    (
                        model_id,
                        forecast_day,
                        actual_spending,
                        predicted_spending,
                        absolute_error
                    )
                    VALUES (%s, %s, %s, %s, %s)

                    ON CONFLICT (model_id, forecast_day)
                    DO UPDATE SET
                        actual_spending = EXCLUDED.actual_spending,
                        predicted_spending = EXCLUDED.predicted_spending,
                        absolute_error = EXCLUDED.absolute_error,
                        prediction_date = CURRENT_TIMESTAMP
                    """,
                    (
                        model_id,
                        day + 1,
                        actual,
                        predicted,
                        absolute_error
                    )
                )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


def save_category_predictions(
    model_id,
    forecast_dates,
    category_names,
    predicted_values
):
    conn = get_db_connection()

    try:
        with conn.cursor() as cur:

            for day_index, forecast_date in enumerate(forecast_dates):

                for category_index, category in enumerate(category_names):

                    predicted = float(
                        predicted_values[day_index][category_index]
                    )

                    cur.execute(
                        """
                        INSERT INTO cnn_category_spending_predictions
                        (
                            model_id,
                            forecast_date,
                            category_name,
                            predicted_spending
                        )
                        VALUES (%s, %s, %s, %s)

                        ON CONFLICT (
                            model_id,
                            forecast_date,
                            category_name
                        )
                        DO UPDATE SET
                            predicted_spending = EXCLUDED.predicted_spending
                        """,
                        (
                            model_id,
                            forecast_date,
                            str(category),
                            predicted
                        )
                    )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()