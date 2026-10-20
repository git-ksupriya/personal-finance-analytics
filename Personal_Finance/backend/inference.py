import os
import sys
import datetime
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from psycopg2.extras import RealDictCursor

# =========================================================
# Model Architectures (matching ml/predict.py & ml/lstm_categories/train_lstm.py)
# =========================================================

class TotalLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=7, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out


class CategoryLSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=7, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out


# =========================================================
# Inference Engine Singleton
# =========================================================

class InferenceEngine:
    _instance = None

    def __init__(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.ml_dir = os.path.join(self.base_dir, "ml")
        self.device = torch.device("cpu")
        
        # 1. Total Spend Model
        self.total_model_path = os.path.join(self.ml_dir, "lstm", "lstm_model.pt")
        self.total_scaler_path = os.path.join(self.ml_dir, "utils", "scaler.pkl")
        if not os.path.exists(self.total_scaler_path):
            self.total_scaler_path = os.path.join(self.base_dir, "preprocessing", "processed", "scaler.pkl")

        self.total_model = TotalLSTMModel()
        if os.path.exists(self.total_model_path):
            self.total_model.load_state_dict(torch.load(self.total_model_path, map_location=self.device))
            self.total_model.eval()
        self.total_scaler = joblib.load(self.total_scaler_path) if os.path.exists(self.total_scaler_path) else None

        # 2. Category Spend Models
        self.cat_dir = os.path.join(self.ml_dir, "lstm_categories", "models")
        self.category_models = {}
        self.category_scalers = {}

        # Mapping between lookup category_name (lower) and folder names
        self.cat_folder_map = {
            "education": "education",
            "rent": "rent",
            "entertainment": "entertainment",
            "food": "food",
            "utilities": "utilities",
            "others": "others",
            "travel": "travel",
            "health": "health",
            "savings": "savings"
        }

        for cat_key, folder in self.cat_folder_map.items():
            model_file = os.path.join(self.cat_dir, folder, "model.pt")
            scaler_file = os.path.join(self.cat_dir, folder, "scaler.pkl")
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                model = CategoryLSTMModel()
                model.load_state_dict(torch.load(model_file, map_location=self.device))
                model.eval()
                self.category_models[cat_key] = model
                self.category_scalers[cat_key] = joblib.load(scaler_file)

        self.model_registry_cache = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def ensure_model_registered(self, cursor, model_name: str, model_type: str, path: str) -> int:
        """Register or retrieve model_id from model_registry."""
        if model_name in self.model_registry_cache:
            return self.model_registry_cache[model_name]

        cursor.execute("SELECT model_id FROM model_registry WHERE model_name = %s", (model_name,))
        row = cursor.fetchone()
        if row:
            m_id = row["model_id"] if isinstance(row, dict) else row[0]
            self.model_registry_cache[model_name] = m_id
            return m_id

        cursor.execute(
            """
            INSERT INTO model_registry (model_name, model_type, blob_storage_path)
            VALUES (%s, %s, %s)
            ON CONFLICT (model_name) DO UPDATE SET model_type = EXCLUDED.model_type
            RETURNING model_id;
            """,
            (model_name, model_type, path)
        )
        res = cursor.fetchone()
        m_id = res["model_id"] if isinstance(res, dict) else res[0]
        self.model_registry_cache[model_name] = m_id
        return m_id

    def predict_overall(self, cursor, user_id: str):
        """
        Runs 7-day multi-step forecast using last 10 expense transactions.
        Returns: (weekly_total, list_of_7_predictions, status_dict)
        """
        cursor.execute(
            """
            SELECT amount, transaction_date
            FROM transactions
            WHERE user_id = %s AND transaction_type = 'Expense'
            ORDER BY transaction_date ASC, transaction_id ASC;
            """,
            (user_id,)
        )
        rows = cursor.fetchall()
        amounts = [float(r["amount"] if isinstance(r, dict) else r[0]) for r in rows]

        if len(amounts) < 10:
            return None, None, {
                "status": "not_enough_data",
                "target": "Overall Weekly Spend",
                "needed": 10,
                "current": len(amounts),
                "missing": 10 - len(amounts),
                "message": f"Not enough data yet for Overall Forecast (have {len(amounts)}, need {10 - len(amounts)} more transactions)"
            }

        input_seq = amounts[-10:]
        seq_array = np.array(input_seq, dtype=np.float32).reshape(-1, 1)
        scaled_seq = self.total_scaler.transform(seq_array).reshape(1, 10, 1)
        input_tensor = torch.tensor(scaled_seq, dtype=torch.float32)

        with torch.no_grad():
            pred_scaled = self.total_model(input_tensor).numpy().reshape(-1, 1)

        pred_raw = self.total_scaler.inverse_transform(pred_scaled).reshape(-1)
        pred_7_days = [round(max(0.0, float(val)), 2) for val in pred_raw]
        weekly_total = round(sum(pred_7_days), 2)

        # Upsert into predictions table
        model_id = self.ensure_model_registered(
            cursor, "Multi-Step 7-Day LSTM Spending Predictor", "LSTM", self.total_model_path
        )
        today = datetime.date.today()
        for d_idx, val in enumerate(pred_7_days):
            target_date = today + datetime.timedelta(days=d_idx + 1)
            cursor.execute(
                """
                INSERT INTO predictions (user_id, model_id, prediction_type, category_id, target_period, predicted_value)
                VALUES (%s, %s, 'overall', NULL, %s, %s)
                ON CONFLICT (user_id, prediction_type, target_period) WHERE category_id IS NULL
                DO UPDATE SET predicted_value = EXCLUDED.predicted_value, model_id = EXCLUDED.model_id;
                """,
                (user_id, model_id, target_date, val)
            )

        return weekly_total, pred_7_days, {"status": "ok", "target": "Overall Weekly Spend"}

    def predict_category(self, cursor, user_id: str, category_id: int, category_name: str):
        """
        Runs 7-day multi-step forecast using last 90 continuous calendar days for this category.
        Returns: (weekly_total, list_of_7_predictions, status_dict)
        """
        cat_key = category_name.strip().lower()
        if cat_key not in self.category_models:
            return None, None, {"status": "unsupported", "target": category_name, "message": f"No trained model for {category_name}"}

        cursor.execute(
            """
            SELECT transaction_date, SUM(amount) AS daily_amount
            FROM transactions
            WHERE user_id = %s AND category_id = %s AND transaction_type = 'Expense'
            GROUP BY transaction_date
            ORDER BY transaction_date ASC;
            """,
            (user_id, category_id)
        )
        rows = cursor.fetchall()
        if not rows:
            return None, None, {
                "status": "not_enough_data",
                "target": category_name,
                "needed": 90,
                "current": 0,
                "missing": 90,
                "message": f"Not enough data yet for {category_name} (need 90 calendar days of history)"
            }

        dates = [pd.to_datetime(r["transaction_date"] if isinstance(r, dict) else r[0]) for r in rows]
        daily_amounts = [float(r["daily_amount"] if isinstance(r, dict) else r[1]) for r in rows]

        df_user = pd.DataFrame({"date": dates, "daily_spending": daily_amounts})
        df_user = df_user.drop_duplicates(subset=["date"]).sort_values("date")

        min_date = df_user["date"].min()
        max_date = df_user["date"].max()
        # Build continuous calendar days
        full_dates = pd.date_range(start=min_date, end=max_date, freq="D")
        total_days = len(full_dates)

        if total_days < 90:
            return None, None, {
                "status": "not_enough_data",
                "target": category_name,
                "needed": 90,
                "current": total_days,
                "missing": 90 - total_days,
                "message": f"Not enough data yet for {category_name} (have {total_days} days, need {90 - total_days} more)"
            }

        complete_df = pd.DataFrame({"date": full_dates}).merge(df_user, on="date", how="left")
        complete_df["daily_spending"] = complete_df["daily_spending"].fillna(0.0)

        # Take last 90 calendar days
        recent_90 = complete_df.iloc[-90:].copy().reset_index(drop=True)

        # 6 Features:
        # 0: daily_spending (scaled with category scaler)
        # 1: weekday_sin
        # 2: weekday_cos
        # 3: month_sin
        # 4: month_cos
        # 5: is_weekend
        scaler = self.category_scalers[cat_key]
        spending_raw = recent_90["daily_spending"].values.reshape(-1, 1)
        spending_scaled = scaler.transform(spending_raw).reshape(-1)

        day_of_week = recent_90["date"].dt.dayofweek.values
        month = recent_90["date"].dt.month.values

        weekday_sin = np.sin(2 * np.pi * day_of_week / 7)
        weekday_cos = np.cos(2 * np.pi * day_of_week / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        is_weekend = (day_of_week >= 5).astype(float)

        feat_matrix = np.column_stack([
            spending_scaled,
            weekday_sin,
            weekday_cos,
            month_sin,
            month_cos,
            is_weekend
        ]).astype(np.float32).reshape(1, 90, 6)

        model = self.category_models[cat_key]
        input_tensor = torch.tensor(feat_matrix, dtype=torch.float32)

        with torch.no_grad():
            pred_scaled = model(input_tensor).numpy().reshape(-1, 1)

        pred_raw = scaler.inverse_transform(pred_scaled).reshape(-1)
        pred_7_days = [round(max(0.0, float(val)), 2) for val in pred_raw]
        weekly_total = round(sum(pred_7_days), 2)

        model_path = os.path.join(self.cat_dir, self.cat_folder_map[cat_key], "model.pt")
        model_id = self.ensure_model_registered(
            cursor, f"Category LSTM: {category_name}", "CategoryLSTM", model_path
        )
        today = datetime.date.today()
        for d_idx, val in enumerate(pred_7_days):
            target_date = today + datetime.timedelta(days=d_idx + 1)
            cursor.execute(
                """
                INSERT INTO predictions (user_id, model_id, prediction_type, category_id, target_period, predicted_value)
                VALUES (%s, %s, 'category', %s, %s, %s)
                ON CONFLICT (user_id, prediction_type, category_id, target_period) WHERE category_id IS NOT NULL
                DO UPDATE SET predicted_value = EXCLUDED.predicted_value, model_id = EXCLUDED.model_id;
                """,
                (user_id, model_id, category_id, target_date, val)
            )

        return weekly_total, pred_7_days, {"status": "ok", "target": category_name}

    def refresh_predictions(self, conn, user_id: str):
        """
        Runs predictions for overall spend and for each category that has data.
        Upserts predictions and returns summary of forecasts, alerts, and data status.
        """
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # 1. Overall forecast
            overall_total, overall_7d, overall_status = self.predict_overall(cursor, user_id)

            # 2. Category forecasts for categories the user has transactions in
            cursor.execute(
                """
                SELECT DISTINCT c.category_id, c.category_name
                FROM transactions t
                JOIN categories c ON t.category_id = c.category_id
                WHERE t.user_id = %s AND t.transaction_type = 'Expense';
                """,
                (user_id,)
            )
            cat_rows = cursor.fetchall()
            cat_predictions = {}
            data_statuses = []

            if overall_status["status"] != "ok":
                data_statuses.append(overall_status)

            for crow in cat_rows:
                cid = crow["category_id"]
                cname = crow["category_name"]
                c_total, c_7d, c_status = self.predict_category(cursor, user_id, cid, cname)
                if c_status["status"] == "ok":
                    cat_predictions[cname] = {
                        "category_id": cid,
                        "weekly_predicted": c_total,
                        "daily_predictions": c_7d
                    }
                else:
                    data_statuses.append(c_status)

            # 3. Check Budgets vs Predictions to generate Alerts
            cursor.execute(
                """
                SELECT b.category_id, c.category_name, b.weekly_amount
                FROM weekly_budgets b
                LEFT JOIN categories c ON b.category_id = c.category_id
                WHERE b.user_id = %s;
                """,
                (user_id,)
            )
            budgets = cursor.fetchall()

            alerts = []
            for b in budgets:
                cid = b["category_id"]
                b_amount = float(b["weekly_amount"])
                if cid is None:
                    # Overall budget
                    if overall_total is not None and overall_total > b_amount:
                        overage = round(overall_total - b_amount, 2)
                        alerts.append({
                            "type": "overall",
                            "name": "Overall Budget",
                            "budget": b_amount,
                            "predicted": overall_total,
                            "overage": overage,
                            "severity": "high" if overage > (b_amount * 0.2) else "medium",
                            "message": f"Predicted weekly overall spend (₹{overall_total:,.2f}) exceeds budget (₹{b_amount:,.2f}) by ₹{overage:,.2f}"
                        })
                else:
                    # Category budget
                    cname = b["category_name"]
                    if cname in cat_predictions:
                        c_pred = cat_predictions[cname]["weekly_predicted"]
                        if c_pred > b_amount:
                            overage = round(c_pred - b_amount, 2)
                            alerts.append({
                                "type": "category",
                                "name": cname,
                                "category_id": cid,
                                "budget": b_amount,
                                "predicted": c_pred,
                                "overage": overage,
                                "severity": "high" if overage > (b_amount * 0.2) else "medium",
                                "message": f"Predicted weekly {cname} spend (₹{c_pred:,.2f}) exceeds budget (₹{b_amount:,.2f}) by ₹{overage:,.2f}"
                            })

            conn.commit()

            return {
                "overall": {
                    "weekly_predicted": overall_total,
                    "daily_predictions": overall_7d
                } if overall_total is not None else None,
                "categories": cat_predictions,
                "alerts": alerts,
                "data_status": data_statuses
            }
