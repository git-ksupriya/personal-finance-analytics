from datetime import date, datetime

import numpy as np
import pandas as pd


ANOMALY_LOOKBACK_MONTHS = 6
MIN_ANOMALY_HISTORY = 12
MIN_PREDICTION_MONTH_HISTORY = 6


def _as_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError, OverflowError):
        return None


def monthly_financial_totals(base_amount, recurring_payments, transactions, today):
    recurring_income = sum(
        float(payment["amount"])
        for payment in recurring_payments
        if payment["payment_type"] == "income" and payment["is_active"]
    )
    recurring_expenditure = sum(
        float(payment["amount"])
        for payment in recurring_payments
        if payment["payment_type"] == "expenditure" and payment["is_active"]
    )
    base_amount = float(base_amount or 0.0)
    month_start = today.replace(day=1)
    actual_spending = sum(
        float(transaction["amount"])
        for transaction in transactions
        if str(transaction["transaction_type"]).casefold() == "expense"
        and month_start <= transaction["transaction_date"] <= today
    )
    total_spendable = base_amount + recurring_income
    return {
        "base_monthly_amount": round(base_amount, 2),
        "recurring_monthly_income": round(recurring_income, 2),
        "recurring_monthly_expenditure": round(recurring_expenditure, 2),
        "total_spendable_amount": round(total_spendable, 2),
        "net_after_recurring_obligations": round(
            total_spendable - recurring_expenditure, 2
        ),
        "actual_monthly_spending": round(actual_spending, 2),
    }


def split_user_history(transactions, account_created_at=None):
    valid = []
    for transaction in transactions:
        transaction_date = _as_date(transaction.get("transaction_date"))
        try:
            amount = float(transaction["amount"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if transaction_date is None or not np.isfinite(amount) or amount <= 0:
            continue
        valid.append({**transaction, "transaction_date": transaction_date, "amount": amount})
    valid.sort(
        key=lambda item: (
            item["transaction_date"],
            item.get("transaction_id") or 0,
        )
    )
    if not valid:
        return [], {
            "latest_observed_date": None,
            "history_start_date": None,
            "history_end_date": None,
            "missing_gap": None,
            "fallback_reason": "No valid persisted transactions are available.",
        }

    uploaded_dates = [
        row["transaction_date"]
        for row in valid
        if str(row.get("source", "")).casefold() in ("csv", "pdf")
    ]
    manual_dates = [
        row["transaction_date"]
        for row in valid
        if str(row.get("source", "manual")).casefold() == "manual"
    ]
    latest_observed = max(row["transaction_date"] for row in valid)
    latest_upload = max(uploaded_dates) if uploaded_dates else None
    first_dashboard_entry = min(manual_dates) if manual_dates else None
    boundary = first_dashboard_entry or _as_date(account_created_at)
    missing_gap = None
    segments = [valid]
    fallback_reason = "Using the user's complete valid transaction history."

    if latest_upload and boundary and latest_upload < boundary:
        missing_gap = {
            "start_exclusive": latest_upload.isoformat(),
            "end_exclusive": boundary.isoformat(),
        }
        before_gap = [row for row in valid if row["transaction_date"] <= latest_upload]
        after_gap = [row for row in valid if row["transaction_date"] >= boundary]
        segments = [segment for segment in (before_gap, after_gap) if segment]
        fallback_reason = (
            "The interval between the last uploaded transaction and first "
            "dashboard entry or account creation is unknown and excluded. "
            "Inference uses one contiguous segment only."
        )

    return segments, {
        "latest_observed_date": latest_observed.isoformat(),
        "history_start_date": None,
        "history_end_date": None,
        "missing_gap": missing_gap,
        "fallback_reason": fallback_reason,
    }


def select_user_history(transactions, account_created_at=None):
    segments, diagnostics = split_user_history(transactions, account_created_at)
    if not segments:
        return [], diagnostics
    selected = segments[-1]
    diagnostics["history_start_date"] = selected[0]["transaction_date"].isoformat()
    diagnostics["history_end_date"] = selected[-1]["transaction_date"].isoformat()
    return selected, diagnostics


def _robust_threshold(history):
    values = np.asarray(history, dtype=np.float64)
    q1, q3 = np.percentile(values, [25, 75], method="linear")
    iqr = q3 - q1
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if iqr > 0:
        threshold = q3 + 1.5 * iqr
        method = "tukey_upper_fence"
    elif mad > 0:
        threshold = median + 3.0 * 1.4826 * mad
        method = "median_plus_3_scaled_mad"
    else:
        threshold = median * 2.0
        method = "twice_median_zero_dispersion_fallback"
    return {
        "q1": float(q1),
        "q3": float(q3),
        "mean": float(np.mean(values)),
        "threshold": float(threshold),
        "method": method,
    }


def _daily_expenses(transactions, category=None, start=None, end=None):
    grouped = {}
    for transaction in transactions:
        if str(transaction.get("transaction_type", "")).casefold() != "expense":
            continue
        if category is not None and transaction.get("category") != category:
            continue
        day = _as_date(transaction.get("transaction_date"))
        try:
            amount = float(transaction["amount"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if day is None or not np.isfinite(amount) or amount <= 0:
            continue
        if start and day < start or end and day > end:
            continue
        grouped[day] = grouped.get(day, 0.0) + amount
    return grouped


def detect_transaction_anomalies(transactions):
    expenses = [
        row
        for row in transactions
        if str(row.get("transaction_type", "")).casefold() == "expense"
    ]
    daily = {}
    for row in expenses:
        event_date = _as_date(row.get("transaction_date"))
        try:
            amount = float(row["amount"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if event_date is None or not np.isfinite(amount) or amount <= 0:
            continue
        event = daily.setdefault(
            event_date,
            {"overall": 0.0, "categories": {}, "transaction_id": None},
        )
        event["overall"] += amount
        category = row.get("category")
        if category is not None:
            event["categories"][category] = (
                event["categories"].get(category, 0.0) + amount
            )
        event["transaction_id"] = row.get("transaction_id") or event["transaction_id"]

    detected = []
    for event_date in sorted(daily):
        event = daily[event_date]
        start = (pd.Timestamp(event_date) - pd.DateOffset(months=ANOMALY_LOOKBACK_MONTHS)).date()
        prior_days = [
            (day, values)
            for day, values in daily.items()
            if start <= day < event_date
        ]
        scopes = [("overall", None, "overall")]
        scopes.extend(
            (f"category:{category}", category, category)
            for category in event["categories"]
        )
        for scope, category, value_key in scopes:
            history = [
                values["overall"]
                if value_key == "overall"
                else values["categories"].get(value_key, 0.0)
                for _, values in prior_days
            ]
            history = [amount for amount in history if amount > 0]
            if len(history) < MIN_ANOMALY_HISTORY:
                continue
            evidence = _robust_threshold(history)
            observed_amount = (
                event["overall"]
                if value_key == "overall"
                else event["categories"][value_key]
            )
            if observed_amount <= evidence["threshold"]:
                continue
            detected.append(
                {
                    "event_key": f"actual:{event_date.isoformat()}:{scope}",
                    "transaction_id": event["transaction_id"],
                    "prediction_id": None,
                    "category": category,
                    "scope": scope,
                    "detection_type": "actual",
                    "target_period": event_date,
                    "observed_amount": observed_amount,
                    "history_count": len(history),
                    "historical_mean": evidence["mean"],
                    "q1": evidence["q1"],
                    "q3": evidence["q3"],
                    "threshold": evidence["threshold"],
                    "method": evidence["method"],
                    "reason": (
                        f"Daily expense {observed_amount:.2f} exceeded the "
                        f"{evidence['method']} threshold {evidence['threshold']:.2f} "
                        f"for {scope}."
                    ),
                }
            )
    return detected


def detect_prediction_anomalies(
    transactions, forecast_categories, prediction_ids, history_diagnostics
):
    segments, _ = split_user_history(transactions)
    detected = []
    periods = forecast_categories.get("target_periods", [])
    for category, category_values in forecast_categories.get("categories", {}).items():
        if not isinstance(category_values, (list, tuple)):
            continue
        eligible = []
        for segment in segments:
            dates = [row["transaction_date"] for row in segment]
            if not dates:
                continue
            first = pd.Timestamp(min(dates))
            last = pd.Timestamp(max(dates))
            start_period = first.to_period("M")
            end_period = last.to_period("M")
            if first.day != 1:
                start_period += 1
            if last.day != last.days_in_month:
                end_period -= 1
            if start_period <= end_period:
                eligible.append((segment, pd.period_range(start_period, end_period, freq="M")))
        if not eligible:
            continue
        segment, history_periods = eligible[-1]
        monthly_amounts = {}
        for row in segment:
            if str(row.get("transaction_type", "")).casefold() != "expense":
                continue
            if row.get("category") != category:
                continue
            period = pd.Timestamp(row["transaction_date"]).to_period("M")
            monthly_amounts[period] = monthly_amounts.get(period, 0.0) + float(
                row["amount"]
            )
        history = [
            monthly_amounts.get(period, 0.0)
            for period in history_periods[-ANOMALY_LOOKBACK_MONTHS:]
        ]
        if len(history) < MIN_PREDICTION_MONTH_HISTORY:
            continue
        evidence = _robust_threshold(history)
        for index, raw_amount in enumerate(category_values):
            if index >= len(periods):
                break
            target_period = _as_date(periods[index])
            if target_period is None:
                continue
            amount = float(raw_amount)
            if amount <= evidence["threshold"]:
                continue
            prediction_id = prediction_ids.get((category, target_period))
            detected.append(
                {
                    "event_key": f"predicted:{category}:{target_period.isoformat()}",
                    "transaction_id": None,
                    "prediction_id": prediction_id,
                    "category": category,
                    "scope": f"category:{category}",
                    "detection_type": "predicted",
                    "target_period": target_period,
                    "observed_amount": amount,
                    "history_count": len(history),
                    "historical_mean": evidence["mean"],
                    "q1": evidence["q1"],
                    "q3": evidence["q3"],
                    "threshold": evidence["threshold"],
                    "method": evidence["method"],
                    "reason": (
                        f"Predicted monthly expense {amount:.2f} exceeded the "
                        f"{evidence['method']} threshold {evidence['threshold']:.2f} "
                        f"for category:{category}."
                    ),
                }
            )
    return detected
