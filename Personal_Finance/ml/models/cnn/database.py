import os
import psycopg2


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"]
    )


def save_model_performance(
    model_name,
    mae,
    rmse,
    r2,
    forecast_days,
    window_size
):
    conn = get_db_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cnn_model_performance
                (
                    model_name,
                    mae,
                    rmse,
                    r2_score,
                    forecast_days,
                    window_size
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING model_id
                """,
                (
                    model_name,
                    float(mae),
                    float(rmse),
                    float(r2) if r2 is not None else None,
                    forecast_days,
                    window_size
                )
            )

            model_id = cur.fetchone()[0]

        conn.commit()

        return model_id

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()