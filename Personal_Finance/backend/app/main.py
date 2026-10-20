from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routes import auth
from backend.app.routes import transactions
from backend.app.routes import analytics
from backend.app.routes import ai
from backend.app.routes import budgets


# ==========================================
# CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Personal Finance Intelligence API",
    description=(
        "Backend API for personal finance management, "
        "analytics, budgeting, and expenditure forecasting."
    ),
    version="1.0.0",
)


# ==========================================
# CORS CONFIGURATION
# ==========================================

# Allows the frontend to communicate with the backend.
# We can restrict origins to the deployed frontend later.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# REGISTER ROUTERS
# ==========================================

app.include_router(auth.router)
app.include_router(transactions.router)
app.include_router(analytics.router)
app.include_router(ai.router)
app.include_router(budgets.router)


# ==========================================
# ROOT ENDPOINT
# ==========================================

@app.get("/")
def root():
    return {
        "message": "Personal Finance Intelligence API",
        "status": "running",
        "version": "1.0.0",
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "personal-finance-backend",
    }


# ==========================================
# API INFORMATION
# ==========================================

@app.get("/api-info")
def api_info():
    return {
        "authentication": "/auth",
        "transactions": "/transactions",
        "analytics": "/analytics",
        "ai": "/ai",
        "budgets": "/budgets",
        "documentation": "/docs",
    }