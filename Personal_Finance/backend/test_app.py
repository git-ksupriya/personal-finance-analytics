import io
import unittest
from contextlib import nullcontext
from datetime import date, timedelta
from unittest.mock import Mock, patch

import torch
from werkzeug.datastructures import FileStorage
from werkzeug.security import check_password_hash

from backend.app import app, parse_import
from backend.inference import predict_category_daily


class FakeCursor:
    def __init__(self):
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeConnection:
    def __init__(self):
        self.last_cursor = FakeCursor()

    def cursor(self, *args, **kwargs):
        return self.last_cursor


class ApplicationRouteTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True, SECRET_KEY="test-secret")
        self.client = app.test_client()

    def test_login_page_is_served_with_csrf_token(self):
        response = self.client.get("/login")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'name="csrf-token"', response.data)
        self.assertIn(b'id="login-form"', response.data)

    def test_dashboard_api_requires_login(self):
        response = self.client.get("/api/dashboard")

        self.assertEqual(response.status_code, 401)
        self.assertIn(b"Please log in", response.data)

    def test_post_requests_require_csrf_token(self):
        response = self.client.post(
            "/api/login",
            json={"username": "someone", "password": "password"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"token", response.data)

    def test_registration_persists_a_password_hash_and_starts_session(self):
        self.client.get("/register")
        with self.client.session_transaction() as flask_session:
            csrf = flask_session["_csrf_token"]

        connection = FakeConnection()
        with patch("backend.app.database_connection", return_value=nullcontext(connection)):
            response = self.client.post(
                "/api/register",
                json={"username": "new-user", "password": "correct horse battery"},
                headers={"X-CSRF-Token": csrf},
            )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json["redirect"], "/onboarding")
        query, params = connection.last_cursor.executed[0]
        self.assertIn("INSERT INTO users", query)
        self.assertTrue(check_password_hash(params[-1], "correct horse battery"))
        self.assertNotEqual(params[-1], "correct horse battery")
        with self.client.session_transaction() as flask_session:
            self.assertTrue(flask_session["user_id"].startswith("U"))
            self.assertEqual(flask_session["username"], "new-user")

    def test_account_deletion_removes_user_owned_records_and_clears_session(self):
        connection = FakeConnection()
        with self.client.session_transaction() as flask_session:
            flask_session["user_id"] = "U123456789"
            flask_session["_csrf_token"] = "delete-token"

        with patch("backend.app.database_connection", return_value=nullcontext(connection)):
            response = self.client.post(
                "/api/account/delete",
                headers={"X-CSRF-Token": "delete-token"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["redirect"], "/login")
        queries = connection.last_cursor.executed
        self.assertEqual(len(queries), 5)
        self.assertIn("DELETE FROM goal_progress", queries[0][0])
        self.assertIn("DELETE FROM monthly_summary", queries[1][0])
        self.assertIn("DELETE FROM category_monthly_summary", queries[2][0])
        self.assertIn("DELETE FROM detected_recurring_payments", queries[3][0])
        self.assertIn("DELETE FROM users WHERE user_id = %s", queries[4][0])
        self.assertTrue(all(params == ("U123456789",) for _, params in queries))
        with self.client.session_transaction() as flask_session:
            self.assertNotIn("user_id", flask_session)

    def test_account_deletion_requires_authentication(self):
        self.client.get("/login")
        with self.client.session_transaction() as flask_session:
            csrf = flask_session["_csrf_token"]

        response = self.client.post(
            "/api/account/delete",
            headers={"X-CSRF-Token": csrf},
        )

        self.assertEqual(response.status_code, 401)


class CsvImportValidationTests(unittest.TestCase):
    def setUp(self):
        self.categories = {"Food": 5, "Rent": 2}
        self.payment_modes = {"cash": 3}
        self.locations = {"mumbai": 1}

    def make_csv(self, content):
        return FileStorage(
            stream=io.BytesIO(content.encode("utf-8")),
            filename="transactions.csv",
            content_type="text/csv",
        )

    def test_import_maps_dataset_columns_and_does_not_trust_csv_user_id(self):
        file_storage = self.make_csv(
            "transaction_id,user_id,date,transaction_type,category,amount,payment_mode,location,notes\n"
            "T123,U-OTHER,2026-01-10,Expense,food,20.50,Cash,Mumbai,Lunch\n"
        )

        rows = parse_import(
            file_storage,
            self.categories,
            self.payment_modes,
            self.locations,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "T123")
        self.assertEqual(rows[0][1:6], (5, 3, 1, 20.5, "Expense"))
        self.assertEqual(rows[0][6].isoformat(), "2026-01-10")
        self.assertEqual(rows[0][7], "Lunch")
        self.assertNotIn("U-OTHER", rows[0])

    def test_rejects_missing_columns_and_invalid_rows(self):
        missing = self.make_csv("date,category\n2026-01-10,Food\n")
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            parse_import(missing, self.categories, self.payment_modes, self.locations)

        invalid = self.make_csv(
            "date,category,amount\nnot-a-date,Food,0\n"
        )
        with self.assertRaisesRegex(ValueError, "CSV validation failed"):
            parse_import(invalid, self.categories, self.payment_modes, self.locations)


class InferenceInputTests(unittest.TestCase):
    def test_category_forecast_handles_empty_transaction_history(self):
        result = predict_category_daily([], as_of=date(2026, 1, 31))

        self.assertFalse(result["available"])
        self.assertEqual(result["categories"], {})
        self.assertEqual(result["errors"], [])

    def test_category_forecast_produces_seven_daily_values(self):
        today = date(2026, 1, 31)
        transactions = [
            {
                "user_id": "TESTUSER",
                "category": "Food",
                "transaction_date": date(2025, 10, 24) + timedelta(days=offset),
                "transaction_type": "Expense",
                "amount": 100 + offset,
            }
            for offset in range(100)
        ]

        class IdentityScaler:
            def transform(self, values):
                return values

            def inverse_transform(self, values):
                return values

        model = Mock(return_value=torch.zeros((1, 7)))
        with patch(
            "backend.inference.load_category_model",
            return_value=(model, IdentityScaler()),
        ):
            result = predict_category_daily(transactions, as_of=today)

        self.assertTrue(result["available"])
        self.assertEqual(len(result["categories"]["Food"]), 7)


if __name__ == "__main__":
    unittest.main()
