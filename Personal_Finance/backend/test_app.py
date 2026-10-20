import io
import unittest
from contextlib import nullcontext
from datetime import date
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
from werkzeug.datastructures import FileStorage
from werkzeug.security import check_password_hash

from backend.app import app, parse_import
from backend.inference import (
    _monthly_overall_series,
    _complete_month_range,
    predict_category_monthly,
    predict_overall_monthly,
)
from ml.lstm_categories.preprocessing.feature_extraction import (
    create_complete_daily_series,
)


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
    def test_monthly_forecasts_handle_empty_transaction_history(self):
        result = predict_category_monthly(
            [],
            account_created_at=date(2026, 1, 31),
        )

        self.assertFalse(result["available"])
        self.assertEqual(result["categories"], {})
        self.assertEqual(result["errors"], [])

    def test_incomplete_boundary_months_are_not_used(self):
        segment = [
            {"transaction_date": date(2024, 1, 2)},
            {"transaction_date": date(2024, 12, 30)},
        ]

        periods = _complete_month_range(segment)

        self.assertEqual(len(periods), 10)
        self.assertEqual(str(periods[0]), "2024-02")
        self.assertEqual(str(periods[-1]), "2024-11")

    @staticmethod
    def make_monthly_transactions(first_month, months, include_gap=False):
        transactions = []
        transaction_id = 1
        for offset in range(months):
            month = pd.Period(first_month, freq="M") + offset
            transaction_dates = [month.start_time.date(), month.end_time.date()]
            for transaction_date in transaction_dates:
                transactions.append(
                    {
                        "transaction_id": transaction_id,
                        "user_id": "TESTUSER",
                        "category": "Food",
                        "transaction_date": transaction_date,
                        "transaction_type": "Expense",
                        "amount": 35.0 if transaction_date.day == 1 else 65.0,
                    }
                )
                transaction_id += 1
        if include_gap:
            transactions.append(
                {
                    "transaction_id": transaction_id,
                    "user_id": "TESTUSER",
                    "category": "Food",
                    "transaction_date": date(2024, 2, 15),
                    "transaction_type": "Income",
                    "amount": 500.0,
                }
            )
        return transactions

    class IdentityScaler:
        def transform(self, values):
            return values

        def inverse_transform(self, values):
            return values

    def test_overall_forecast_is_seven_calendar_month_totals(self):
        transactions = self.make_monthly_transactions("2024-01", 12, include_gap=True)
        model = Mock(return_value=torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]]))
        with patch(
            "backend.inference.load_overall_model",
            return_value=(model, self.IdentityScaler()),
        ):
            result = predict_overall_monthly(
                transactions,
                account_created_at=date(2024, 1, 1),
            )

        self.assertTrue(result["available"])
        self.assertEqual(result["target_unit"], "INR per calendar month")
        self.assertEqual(len(result["values"]), 7)
        self.assertEqual(result["target_periods"][0], "2025-01-01")
        np.testing.assert_array_equal(
            model.call_args.args[0].numpy()[0, :, 0],
            np.full(10, 100.0),
        )

    def test_category_forecast_aggregates_transactions_to_monthly_totals(self):
        transactions = self.make_monthly_transactions("2024-01", 14)
        transactions = [
            {key: value for key, value in row.items() if key != "transaction_id"}
            for row in transactions
        ]
        model = Mock(return_value=torch.zeros((1, 7)))
        with patch(
            "backend.inference.load_category_model",
            return_value=(model, self.IdentityScaler()),
        ):
            result = predict_category_monthly(
                transactions,
                account_created_at=date(2024, 1, 1),
            )

        self.assertTrue(result["available"])
        self.assertEqual(len(result["categories"]["Food"]), 7)
        self.assertEqual(result["target_unit"], "INR per calendar month")
        self.assertEqual(result["target_periods"][0], "2025-03-01")
        model_input = model.call_args.args[0].numpy()
        np.testing.assert_array_equal(model_input[0, :, 0], np.full(12, 100.0))

    def test_internal_missing_month_is_zero_filled(self):
        transactions = self.make_monthly_transactions("2024-01", 12)
        transactions = [
            row
            for row in transactions
            if row["transaction_date"].month != 3
        ]
        segment = [
            {**row, "transaction_date": row["transaction_date"]}
            for row in transactions
        ]
        periods = _complete_month_range(segment)
        totals = _monthly_overall_series(segment, periods)

        self.assertEqual(len(periods), 12)
        self.assertEqual(totals[1], 100.0)
        self.assertEqual(totals[2], 0.0)

    def test_missing_days_inside_history_coverage_are_zero_filled(self):
        daily = pd.DataFrame(
            [
                {
                    "user_id": "TESTUSER",
                    "category": "Food",
                    "date": pd.Timestamp("2024-01-01"),
                    "daily_spending": 10.0,
                },
                {
                    "user_id": "TESTUSER",
                    "category": "Food",
                    "date": pd.Timestamp("2024-01-03"),
                    "daily_spending": 20.0,
                },
            ]
        )

        completed = create_complete_daily_series(daily)

        self.assertEqual(
            completed["daily_spending"].tolist(),
            [10.0, 0.0, 20.0],
        )

    def test_unknown_upload_gap_is_not_bridged_for_monthly_history(self):
        uploaded = self.make_monthly_transactions("2023-01", 14)
        dashboard = self.make_monthly_transactions("2024-06", 6)
        for offset, row in enumerate(dashboard, start=len(uploaded) + 1):
            row["transaction_id"] = offset
            row["source"] = "manual"
        for row in uploaded:
            row["source"] = "csv"
        with patch(
            "backend.inference.load_overall_model",
            return_value=(
                Mock(return_value=torch.zeros((1, 7))),
                self.IdentityScaler(),
            ),
        ):
            result = predict_overall_monthly(
                uploaded + dashboard,
                account_created_at=date(2024, 6, 1),
            )

        self.assertTrue(result["available"])
        self.assertEqual(result["target_periods"][0], "2024-03-01")
        self.assertEqual(
            result["diagnostics"]["missing_gap"],
            {"start_exclusive": "2024-02-29", "end_exclusive": "2024-06-01"},
        )


if __name__ == "__main__":
    unittest.main()
