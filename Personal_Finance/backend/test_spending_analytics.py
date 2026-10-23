import unittest
from datetime import date, datetime

import pandas as pd

from backend.spending_analytics import (
    detect_prediction_anomalies,
    detect_transaction_anomalies,
    monthly_financial_totals,
    select_user_history,
    split_user_history,
)


class FinancialTotalsTests(unittest.TestCase):
    def test_expenses_and_typed_recurring_cash_flows_are_separated(self):
        transactions = [
            {
                "transaction_type": "Expense",
                "transaction_date": date(2026, 10, 4),
                "amount": 45,
            },
            {
                "transaction_type": "Income",
                "transaction_date": date(2026, 10, 5),
                "amount": 1000,
            },
            {
                "transaction_type": "Expense",
                "transaction_date": date(2026, 9, 30),
                "amount": 300,
            },
        ]
        recurring = [
            {"payment_type": "income", "amount": 2000, "is_active": True},
            {"payment_type": "expenditure", "amount": 500, "is_active": True},
            {"payment_type": None, "amount": 900, "is_active": True},
            {"payment_type": "income", "amount": 100, "is_active": False},
        ]

        totals = monthly_financial_totals(
            10000, recurring, transactions, date(2026, 10, 20)
        )

        self.assertEqual(totals["actual_monthly_spending"], 45)
        self.assertEqual(totals["recurring_monthly_income"], 2000)
        self.assertEqual(totals["recurring_monthly_expenditure"], 500)
        self.assertEqual(totals["total_spendable_amount"], 12000)
        self.assertEqual(totals["net_after_recurring_obligations"], 11500)


class HistorySegmentationTests(unittest.TestCase):
    def test_uploaded_to_manual_interval_is_excluded_and_not_zero_filled(self):
        transactions = [
            {
                "transaction_id": 1,
                "source": "csv",
                "transaction_date": date(2026, 1, 1),
                "amount": 10,
            },
            {
                "transaction_id": 2,
                "source": "csv",
                "transaction_date": date(2026, 1, 3),
                "amount": 20,
            },
            {
                "transaction_id": 3,
                "source": "manual",
                "transaction_date": date(2026, 1, 10),
                "amount": 30,
            },
            {
                "transaction_id": 4,
                "source": "manual",
                "transaction_date": date(2026, 1, 12),
                "amount": 40,
            },
        ]

        segments, diagnostics = split_user_history(transactions)
        selected, selected_diagnostics = select_user_history(transactions)

        self.assertEqual(
            [[row["transaction_date"] for row in segment] for segment in segments],
            [[date(2026, 1, 1), date(2026, 1, 3)], [date(2026, 1, 10), date(2026, 1, 12)]],
        )
        self.assertEqual(selected, segments[-1])
        self.assertEqual(
            diagnostics["missing_gap"],
            {"start_exclusive": "2026-01-03", "end_exclusive": "2026-01-10"},
        )
        self.assertEqual(selected_diagnostics["history_start_date"], "2026-01-10")
        self.assertEqual(selected_diagnostics["history_end_date"], "2026-01-12")

    def test_account_creation_date_is_used_when_no_manual_record_exists(self):
        transactions = [
            {
                "source": "csv",
                "transaction_date": date(2026, 1, 1),
                "amount": 10,
            },
            {
                "source": "csv",
                "transaction_date": date(2026, 1, 3),
                "amount": 20,
            },
        ]

        selected, diagnostics = select_user_history(
            transactions, datetime(2026, 1, 10, 12, 30)
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(
            diagnostics["missing_gap"],
            {"start_exclusive": "2026-01-03", "end_exclusive": "2026-01-10"},
        )


class AnomalyDetectionTests(unittest.TestCase):
    def test_income_is_not_an_anomaly_and_zero_dispersion_has_a_threshold(self):
        transactions = [
            {
                "transaction_id": index,
                "transaction_date": date(2026, 1, index),
                "transaction_type": "Expense",
                "category": "Food",
                "amount": 10,
            }
            for index in range(1, 13)
        ]
        transactions.extend(
            [
                {
                    "transaction_id": 13,
                    "transaction_date": date(2026, 1, 13),
                    "transaction_type": "Income",
                    "category": "Food",
                    "amount": 10000,
                },
                {
                    "transaction_id": 14,
                    "transaction_date": date(2026, 1, 14),
                    "transaction_type": "Expense",
                    "category": "Food",
                    "amount": 300,
                },
            ]
        )

        anomalies = detect_transaction_anomalies(transactions)

        self.assertEqual(len(anomalies), 2)
        self.assertEqual(
            {event["detection_type"] for event in anomalies}, {"actual"}
        )
        self.assertTrue(all(event["transaction_id"] == 14 for event in anomalies))
        self.assertTrue(
            all(
                event["method"] == "twice_median_zero_dispersion_fallback"
                for event in anomalies
            )
        )

    def test_anomaly_detection_uses_daily_category_totals(self):
        transactions = [
            {
                "transaction_id": index,
                "transaction_date": date(2026, 2, index),
                "transaction_type": "Expense",
                "category": "Food",
                "amount": 10,
            }
            for index in range(1, 13)
        ]
        transactions.extend(
            [
                {
                    "transaction_id": 13,
                    "transaction_date": date(2026, 2, 13),
                    "transaction_type": "Expense",
                    "category": "Food",
                    "amount": 15,
                },
                {
                    "transaction_id": 14,
                    "transaction_date": date(2026, 2, 13),
                    "transaction_type": "Expense",
                    "category": "Food",
                    "amount": 10,
                },
            ]
        )

        anomalies = detect_transaction_anomalies(transactions)

        self.assertEqual(len(anomalies), 2)
        self.assertTrue(all(event["observed_amount"] == 25 for event in anomalies))
        self.assertTrue(
            all(event["transaction_id"] == 14 for event in anomalies)
        )

    def test_prediction_anomalies_compare_monthly_amounts_to_monthly_history(self):
        transactions = []
        for offset in range(12):
            month = pd.Period("2025-01", freq="M") + offset
            for transaction_date in (month.start_time.date(), month.end_time.date()):
                transactions.append(
                    {
                        "transaction_id": len(transactions) + 1,
                        "transaction_date": transaction_date,
                        "transaction_type": "Expense",
                        "category": "Food",
                        "amount": 50,
                    }
                )

        anomalies = detect_prediction_anomalies(
            transactions,
            {
                "categories": {"Food": [250]},
                "target_periods": ["2026-01-01"],
            },
            {("Food", date(2026, 1, 1)): 22},
            {},
        )

        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["target_period"], date(2026, 1, 1))
        self.assertIn("monthly expense", anomalies[0]["reason"])


if __name__ == "__main__":
    unittest.main()
