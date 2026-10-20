import os
import sys
import io
import csv
import pandas as pd
from starlette.testclient import TestClient

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add backend directory to sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(repo_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import app, get_connection

def run_demo():
    print("=" * 70)
    print("PERSONAL FINANCE ANALYTICS: END-TO-END VERIFICATION")
    print("=" * 70)

    client = TestClient(app)

    # -------------------------------------------------------------
    # TEST 1: BRAND NEW USER WITH ZERO TRANSACTION HISTORY
    # -------------------------------------------------------------
    print("\n>>> TEST 1: Register Brand-New User (Zero History)")
    brand_new_username = f"newbie_{os.getpid()}"
    signup_payload = {
        "username": brand_new_username,
        "name": "Newbie Test User",
        "email": f"{brand_new_username}@example.com",
        "password": "Password123!"
    }
    res = client.post("/api/signup", json=signup_payload)
    print(f"POST /api/signup Status: {res.status_code}")
    assert res.status_code == 200, res.text
    new_user = res.json()["user"]
    user_id = new_user["user_id"]
    print(f"Created User: {new_user['name']} (ID: {user_id}, Username: {new_user['username']}, Onboarded: {new_user['onboarded']})")

    # Check /api/me
    res = client.get("/api/me")
    assert res.status_code == 200
    assert res.json()["onboarded"] is False
    print("Verified GET /api/me: user is authenticated and not yet onboarded.")

    # Check predictions on empty user: must report not enough data, NOT an error
    res = client.get("/api/alerts")
    assert res.status_code == 200, res.text
    alerts_data = res.json()
    print(f"Alerts count: {len(alerts_data.get('alerts', []))}")
    print(f"Data status notices (missing history handled gracefully): {alerts_data.get('data_status')}")
    assert len(alerts_data.get("data_status", [])) > 0
    print("PASS: Empty user gracefully receives 'not enough data yet' instead of error.")

    # Onboarding: Set budgets
    onboard_payload = {
        "weekly_overall": 12000.0,
        "category_budgets": [
            {"category_name": "Food", "weekly_amount": 2500.0},
            {"category_name": "Entertainment", "weekly_amount": 1000.0}
        ]
    }
    res = client.post("/api/onboarding/budgets", json=onboard_payload)
    assert res.status_code == 200
    print("POST /api/onboarding/budgets: Stored weekly budgets and marked user onboarded.")

    # Verify user is now onboarded
    res = client.get("/api/me")
    assert res.json()["onboarded"] is True
    print("PASS: User onboarded state is True.")

    # Log a transaction for this brand-new user
    tx_payload = {
        "amount": 750.0,
        "category": "Food",
        "payment_mode": "UPI",
        "location": "Bangalore",
        "description": "Lunch at cafe"
    }
    res = client.post("/api/transactions", json=tx_payload)
    assert res.status_code == 200, res.text
    tx_id = res.json()["transaction_id"]
    print(f"POST /api/transactions: Logged transaction #{tx_id} for {user_id}.")

    # Log out brand-new user
    client.post("/api/logout")
    res = client.get("/api/me")
    assert res.status_code == 401
    print("POST /api/logout: Verified session cleared and protected route returns 401.")


    # -------------------------------------------------------------
    # TEST 2: DEMO USER WITH HISTORICAL CSV IMPORT & REAL MODELS
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(">>> TEST 2: Demo User with Real Data CSV Upload & Model Inference")
    print("=" * 70)

    demo_username = f"demo_{os.getpid()}"
    demo_signup = {
        "username": demo_username,
        "name": "Demo Real Data User",
        "email": f"{demo_username}@test.com",
        "password": "SecureDemoPassword@123"
    }
    res = client.post("/api/signup", json=demo_signup)
    assert res.status_code == 200
    demo_user_id = res.json()["user"]["user_id"]
    print(f"Created Demo User: {demo_user_id} (@{demo_username})")

    # Read real transactions for user U001 from standardised_dataset.csv
    dataset_path = os.path.join(repo_root, "preprocessing", "processed", "standardised_dataset.csv")
    df_raw = pd.read_csv(dataset_path)
    user_u001_txns = df_raw[df_raw["user_id"] == "U001"].copy()
    print(f"Extracted {len(user_u001_txns)} transactions from dataset for user U001.")

    # Create CSV in-memory buffer
    csv_buffer = io.StringIO()
    user_u001_txns.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    # Upload CSV through /api/onboarding/upload-csv
    files = {"file": ("user_transactions.csv", csv_bytes, "text/csv")}
    res = client.post("/api/onboarding/upload-csv", files=files)
    assert res.status_code == 200, res.text
    upload_res = res.json()
    print(f"POST /api/onboarding/upload-csv Results: Inserted={upload_res['inserted']}, Skipped={upload_res['skipped']}, Bad={upload_res['bad_rows']}")
    assert upload_res["inserted"] > 0

    # Set Low Budgets to test alert generation
    # User U001 spends ~INR 85,000/week, so a budget of INR 10,000 MUST trigger an alert!
    budget_payload = {
        "weekly_overall": 10000.0,
        "category_budgets": [
            {"category_name": "Food", "weekly_amount": 100.0},
            {"category_name": "Education", "weekly_amount": 50.0}
        ]
    }
    res = client.post("/api/onboarding/budgets", json=budget_payload)
    assert res.status_code == 200
    print("Configured test budgets: Overall=INR 10,000, Food=INR 100, Education=INR 50.")

    # 1. Fetch Weekly Spend (Actual spend over last 7 days vs Budget)
    res = client.get("/api/analytics/weekly-spend")
    assert res.status_code == 200
    spend_data = res.json()
    print(f"\n--- GET /api/analytics/weekly-spend ---")
    print(f"Period: {spend_data['period']['start_date']} to {spend_data['period']['end_date']}")
    print(f"Overall Actual Spend: INR {spend_data['overall']['actual']:,.2f} | Stated Budget: INR {spend_data['overall']['budget']:,.2f}")
    for cat in spend_data["categories"][:5]:
        print(f"  {cat['category_name']}: Actual=INR {cat['actual']:,.2f}, Budget={cat['budget']}")

    # 2. Fetch AI Predictions
    res = client.get("/api/predictions")
    assert res.status_code == 200
    pred_data = res.json()
    print(f"\n--- GET /api/predictions ---")
    if pred_data.get("overall"):
        ov = pred_data["overall"]
        print(f"Overall 7-Day Predicted Spend Total: INR {ov['weekly_predicted']:,.2f}")
        print(f"Daily Step Forecasts: {ov['daily_predictions']}")
    print(f"Categories predicted: {list(pred_data.get('categories', {}).keys())}")

    # 3. Fetch Alerts
    res = client.get("/api/alerts")
    assert res.status_code == 200
    alerts_res = res.json()
    print(f"\n--- GET /api/alerts ---")
    print(f"Total Alerts Triggered: {len(alerts_res['alerts'])}")
    for a in alerts_res["alerts"]:
        print(f"  [{a['severity'].upper()}] {a['message']}")
    assert len(alerts_res["alerts"]) > 0, "Expected budget overage alert to trigger!"
    print("PASS: Alert triggered when stated budget is exceeded!")

    # 4. Log a daily transaction and verify real-time prediction refresh
    print("\n--- Testing Daily Transaction Logging ---")
    log_res = client.post("/api/transactions", json={
        "amount": 2500.0,
        "category": "Food",
        "payment_mode": "Credit Card",
        "description": "Dinner with colleagues"
    })
    assert log_res.status_code == 200, log_res.text
    print(f"Successfully logged new transaction ID #{log_res.json()['transaction_id']}.")
    print(f"Refreshed Predictions Total: INR {log_res.json()['refreshed_predictions']['overall']['weekly_predicted']:,.2f}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_demo()
