import os
import json
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

FORECAST_MONTHS = 7

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=FORECAST_MONTHS, dropout=0.2):
        super(LSTMModel, self).__init__()
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
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.lstm(x)
        # Take last time step output
        last_out = out[:, -1, :]
        out = self.fc1(last_out)
        out = self.relu(out)
        out = self.fc2(out) # shape: (batch_size, output_size)
        return out

def train_lstm(epochs=40, batch_size=64, learning_rate=0.001):
    lstm_dir = os.path.dirname(os.path.abspath(__file__))
    ml_dir = os.path.abspath(os.path.join(lstm_dir, ".."))
    processed_dir = os.path.abspath(
        os.path.join(ml_dir, "..", "preprocessing", "processed", "monthly")
    )
    monthly_model_dir = os.path.join(lstm_dir, "monthly")
    os.makedirs(monthly_model_dir, exist_ok=True)

    print("--- Loading Monthly Expense Sequences for 7-Month LSTM ---")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    y_test_raw = np.load(os.path.join(processed_dir, "y_test_raw.npy"))

    scaler_path = os.path.join(processed_dir, "scaler.pkl")
    scaler = joblib.load(scaler_path)

    # Convert to Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = LSTMModel(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=FORECAST_MONTHS,
        dropout=0.2,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"--- Training Multi-Step LSTM Model ({epochs} Epochs) ---")
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_X.size(0)
        
        epoch_loss = total_loss / len(train_dataset)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss (MSE): {epoch_loss:.6f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_preds_scaled = model(X_test_tensor).numpy()

    # Inverse transform predictions and targets to raw INR currency
    test_preds_raw = scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).reshape(test_preds_scaled.shape)

    mse_scaled = mean_squared_error(y_test, test_preds_scaled)
    mae_scaled = mean_absolute_error(y_test, test_preds_scaled)

    mse_raw_overall = mean_squared_error(y_test_raw, test_preds_raw)
    rmse_raw_overall = np.sqrt(mse_raw_overall)
    mae_raw_overall = mean_absolute_error(y_test_raw, test_preds_raw)
    r2_raw_overall = r2_score(y_test_raw.flatten(), test_preds_raw.flatten())

    print("\n=== Multi-Step LSTM Model Evaluation Results ===")
    print(f"Scaled Overall MSE : {mse_scaled:.6f}")
    print(f"Raw Overall RMSE   : INR {rmse_raw_overall:.2f}")
    print(f"Raw Overall MAE    : INR {mae_raw_overall:.2f}")
    print(f"Raw Overall R²     : {r2_raw_overall:.4f}")

    per_month_metrics = {}
    print("\n--- Per-Month Forecasting Performance ---")
    for month in range(FORECAST_MONTHS):
        month_actual = y_test_raw[:, month]
        month_pred = test_preds_raw[:, month]
        month_rmse = np.sqrt(mean_squared_error(month_actual, month_pred))
        month_mae = mean_absolute_error(month_actual, month_pred)
        month_r2 = r2_score(month_actual, month_pred)
        per_month_metrics[f"Month_{month+1}"] = {
            "rmse_inr": float(month_rmse),
            "mae_inr": float(month_mae),
            "r2_score": float(month_r2)
        }
        print(
            f"  Month {month+1}: RMSE = INR {month_rmse:8.2f} | "
            f"MAE = INR {month_mae:8.2f} | R² = {month_r2:.4f}"
        )

    model_path = os.path.join(monthly_model_dir, "lstm_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nLSTM model weights saved to {model_path}")
    joblib.dump(scaler, os.path.join(monthly_model_dir, "scaler.pkl"))

    metrics = {
        "model_name": "Multi-Step 7-Month LSTM Spending Predictor",
        "model_type": "LSTM",
        "input_timesteps_months": 10,
        "forecast_horizon_months": FORECAST_MONTHS,
        "target_unit": "INR per calendar month",
        "epochs": epochs,
        "batch_size": batch_size,
        "scaled_overall_mse": float(mse_scaled),
        "scaled_overall_mae": float(mae_scaled),
        "raw_overall_rmse": float(rmse_raw_overall),
        "raw_overall_mae": float(mae_raw_overall),
        "raw_overall_r2": float(r2_raw_overall),
        "per_month_metrics": per_month_metrics
    }

    metrics_path = os.path.join(monthly_model_dir, "lstm_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"LSTM metrics saved to {metrics_path}")

    res_df = pd.DataFrame()
    for month in range(FORECAST_MONTHS):
        res_df[f"Actual_Month_{month+1}"] = y_test_raw[:, month].round(2)
        res_df[f"Pred_Month_{month+1}"] = test_preds_raw[:, month].round(2)
    
    results_csv_path = os.path.join(monthly_model_dir, "lstm_results.csv")
    res_df.to_csv(results_csv_path, index=False)
    print(f"LSTM test predictions saved to {results_csv_path}")

    # 4. Generate Report MD
    report_md = f"""# LSTM 7-Month Multi-Step Spending Forecast Report

## 📌 Model Overview
- **Architecture**: 2-Layer PyTorch LSTM (`hidden_size=64, dropout=0.2`) -> Dense(32) -> Output(7)
- **Input Sequence Length**: 10 past spending timesteps
- **Forecast Horizon**: Next {FORECAST_MONTHS} calendar-month expense totals
- **Epochs Trained**: {epochs}

---

## 📊 Overall Performance Metrics
- **Scaled MSE**: {mse_scaled:.6f}
- **Scaled MAE**: {mae_scaled:.6f}
- **Raw Currency RMSE**: INR {rmse_raw_overall:,.2f}
- **Raw Currency MAE**: INR {mae_raw_overall:,.2f}
- **Overall $R^2$ Score**: {r2_raw_overall:.4f}

---

## 📅 Month-by-Month Forecast Performance

| Month Horizon | RMSE (INR) | MAE (INR) | $R^2$ Score |
|---|---|---|---|
"""
    for month in range(FORECAST_MONTHS):
        m = per_month_metrics[f"Month_{month+1}"]
        report_md += f"| Month {month+1} | INR {m['rmse_inr']:,.2f} | INR {m['mae_inr']:,.2f} | {m['r2_score']:.4f} |\n"

    report_md += f"""
---

## 🔮 Sample Monthly Test Predictions

Below are sample seven-month spending forecasts generated by the trained LSTM model:

| Sample | Forecast Horizon (Month 1 → Month 7) Predicted Amounts (INR) |
|---|---|
"""
    for i in range(min(5, len(res_df))):
        sample_preds = [
            f"INR {res_df.loc[i, f'Pred_Month_{month+1}']:,.2f}"
            for month in range(FORECAST_MONTHS)
        ]
        report_md += f"| Sample {i+1} | {', '.join(sample_preds)} |\n"

    report_path = os.path.join(monthly_model_dir, "lstm_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"LSTM human-readable report saved to {report_path}")

    return model, metrics

if __name__ == "__main__":
    train_lstm()
