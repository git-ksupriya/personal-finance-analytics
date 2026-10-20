import os

import psycopg2
from psycopg2.extras import RealDictCursor


# ==================================================
# DATABASE CONFIGURATION
# ==================================================

def get_db_connection():
    """
    Create and return a PostgreSQL database connection.

    Configuration is read from environment variables.

    Supported variables:
        DB_HOST
        DB_PORT
        DB_NAME
        DB_USER
        DB_PASSWORD
    """

    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv(
        "DB_NAME",
        "personal_finance_db",
    )
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")

    # Avoid attempting a connection with missing credentials.
    if not db_user:
        raise RuntimeError(
            "DB_USER is missing. Configure your database environment."
        )

    if not db_password:
        raise RuntimeError(
            "DB_PASSWORD is missing. Configure your database environment."
        )

    # Establish the PostgreSQL connection.
    connection = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        cursor_factory=RealDictCursor,
        connect_timeout=10,
        application_name="PersonalFinanceIntelligenceAPI",
    )

    return connection

