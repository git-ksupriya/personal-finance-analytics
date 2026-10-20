
import os
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field

from backend.app.services.database import get_db_connection


router = APIRouter(prefix="/auth", tags=["Authentication"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

SECRET_KEY = os.getenv("JWT_SECRET_KEY")

if not SECRET_KEY:
    raise RuntimeError(
        "Set the JWT_SECRET_KEY environment variable before starting the API."
    )

ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 12


# Request formats

class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)
    phone: str | None = Field(default=None, max_length=15)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# Password helpers

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return pwd_context.verify(password, hashed_password)


# Token helpers

def create_access_token(user_id: str) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(
        hours=TOKEN_EXPIRY_HOURS
    )

    payload = {
        "sub": user_id,
        "exp": expires_at
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        user_id = payload.get("sub")

        if not isinstance(user_id, str) or not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token"
            )

    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token"
        )

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT user_id, name, email
                FROM users
                WHERE user_id = %s
                """,
                (user_id,)
            )

            user = cursor.fetchone()

            if not user:
                raise HTTPException(
                    status_code=401,
                    detail="User no longer exists"
                )

            return dict(user)

    finally:
        connection.close()


# Registration

@router.post("/register", status_code=201)
def register_user(data: RegisterRequest):

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:

            cursor.execute(
                "SELECT 1 FROM users WHERE email = %s",
                (str(data.email),)
            )

            if cursor.fetchone():
                raise HTTPException(
                    status_code=409,
                    detail="Email is already registered"
                )

            # Generate a unique ID compatible with VARCHAR(10).
            user_id = None

            for _ in range(5):
                candidate = "U" + uuid.uuid4().hex[:9].upper()

                cursor.execute(
                    "SELECT 1 FROM users WHERE user_id = %s",
                    (candidate,)
                )

                if not cursor.fetchone():
                    user_id = candidate
                    break

            if user_id is None:
                raise HTTPException(
                    status_code=500,
                    detail="Could not generate a user ID"
                )

            cursor.execute(
                """
                INSERT INTO users
                    (user_id, name, email, password_hash, phone)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    data.name.strip(),
                    str(data.email).lower(),
                    hash_password(data.password),
                    data.phone
                )
            )

        connection.commit()

        return {
            "message": "Registration successful",
            "user_id": user_id
        }

    except HTTPException:
        connection.rollback()
        raise

    except Exception:
        connection.rollback()
        raise HTTPException(
            status_code=500,
            detail="Registration failed"
        )

    finally:
        connection.close()


# Login

@router.post("/login")
def login_user(data: LoginRequest):

    connection = get_db_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT user_id, name, email, password_hash
                FROM users
                WHERE email = %s
                """,
                (str(data.email).lower(),)
            )

            user = cursor.fetchone()

        if not user or not verify_password(
            data.password,
            user["password_hash"]
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )

        token = create_access_token(user["user_id"])

        # Record the login in the existing sessions table.
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO user_sessions
                    (user_id, token, device_info, expires_at)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    user["user_id"],
                    token,
                    "Web application",
                    datetime.now(timezone.utc)
                    + timedelta(hours=TOKEN_EXPIRY_HOURS)
                )
            )

        connection.commit()

        return {
            "message": "Login successful",
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "user_id": user["user_id"],
                "name": user["name"],
                "email": user["email"]
            }
        }

    except HTTPException:
        connection.rollback()
        raise

    except Exception:
        connection.rollback()
        raise HTTPException(
            status_code=500,
            detail="Login failed"
        )

    finally:
        connection.close()


# Current authenticated user

@router.get("/me")
def get_me(user=Depends(get_current_user)):
    return {"user": user}
