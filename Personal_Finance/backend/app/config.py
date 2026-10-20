import os
import secrets
from pathlib import Path

from dotenv import load_dotenv


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

# Locate backend/.env independently of the current
# terminal working directory.

BACKEND_DIRECTORY = Path(__file__).resolve().parents[1]

ENV_FILE = BACKEND_DIRECTORY / ".env"

load_dotenv(ENV_FILE, override=False)


# ==================================================
# DATABASE CONFIGURATION
# ==================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "personal_finance_db")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ==================================================
# AUTHENTICATION CONFIGURATION
# ==================================================

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "720")
)


def validate_auth_configuration():
    """
    Check that a secure JWT secret has been configured.

    A randomly generated fallback is deliberately not used:
    tokens must remain verifiable across application restarts.
    """

    if not JWT_SECRET_KEY:
        raise RuntimeError(
            "JWT_SECRET_KEY is missing. "
            "Configure it in backend/.env before starting the API."
        )

    if len(JWT_SECRET_KEY) < 32:
        raise RuntimeError(
            "JWT_SECRET_KEY must contain at least 32 characters."
        )

    if JWT_ALGORITHM != "HS256":
        raise RuntimeError(
            "This application currently supports HS256 only."
        )

    if ACCESS_TOKEN_EXPIRE_MINUTES <= 0:
        raise RuntimeError(
            "ACCESS_TOKEN_EXPIRE_MINUTES must be positive."
        )


# ==================================================
# FRONTEND CONFIGURATION
# ==================================================

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://localhost:3000",
)


# ==================================================
# APPLICATION CONFIGURATION
# ==================================================

APP_NAME = "Personal Finance Intelligence API"
APP_VERSION = "1.0.0"
