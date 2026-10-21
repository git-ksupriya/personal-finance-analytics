import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "preprocessing", "processed"))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

class SpendingAutoencoder(nn.Module):
    """
    Deep PyTorch Autoencoder for Spending Anomaly Detection.
    Encodes transaction spending patterns and reconstructs them.
    Transactions with high reconstruction error are flagged as anomalous.
    """
    def __init__(self, input_dim=6, latent_dim=3):
        super(SpendingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

def train_autoencoder(eight_weeks_only=True, epochs=40, batch_size=64, lr=0.001):
    print("=" * 60)
    print("TRAINING AUTOENCODER ANOMALY DETECTOR (8-WEEK DATA CONTEXT)")
    print("=" * 60)

    # 1. Load Data
    dataset_path = os.path.join(PROCESSED_DIR, "feature_engineered_dataset.csv")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(PROCESSED_DIR, "cleaned_dataset.csv")
    
    df = pd.read_csv(dataset_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    print(f"Total transactions in dataset: {len(df):,}")

    # 2. Filter for 8 Weeks of Data (8 * 7 = 56 days)
    if eight_weeks_only:
        # Take the reference 8-week training window (56 days)
        min_date = df["date"].min()
        end_date = min_date + pd.Timedelta(weeks=8)
        train_df = df[(df["date"] >= min_date) & (df["date"] < end_date)].copy()
        test_df = df[df["date"] >= end_date].copy()
        if len(test_df) == 0:
            test_df = train_df.copy()
        print(f"Filtered for 8-week historical window ({min_date.date()} to {end_date.date()}):")
        print(f"  Training transactions (8 weeks): {len(train_df):,}")
        print(f"  Evaluation transactions (remaining period): {len(test_df):,}")
    else:
        train_df = df.copy()
        test_df = df.copy()

    # 3. Select Numeric Features for Autoencoder
    candidate_features = [
        "amount", "days_since_last_txn", "monthly_total_spend", 
        "category_monthly_spend", "user_txn_frequency", "rolling_avg_3"
    ]
    feature_cols = [c for c in candidate_features if c in df.columns]
    if len(feature_cols) < 2:
        feature_cols = ["amount"]

    print(f"Feature columns used: {feature_cols}")

    # Standardize Features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].fillna(0).values)
    X_test = scaler.transform(test_df[feature_cols].fillna(0).values)

    joblib.dump(scaler, os.path.join(MODELS_DIR, "autoencoder_scaler.pkl"))

    # Convert to Tensors
    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    # 4. Initialize Model, Loss, Optimizer
    input_dim = len(feature_cols)
    model = SpendingAutoencoder(input_dim=input_dim, latent_dim=max(2, input_dim // 2))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # 5. Training Loop
    print("\nTraining Autoencoder...")
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in train_loader:
            x_batch = batch[0]
            optimizer.zero_grad()
            reconstructed = model(x_batch)
            loss = criterion(reconstructed, x_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x_batch)

        epoch_loss = total_loss / len(train_tensor)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:2d}/{epochs}] - Reconstruction Loss: {epoch_loss:.6f}")

    # 6. Compute Reconstruction Errors & Threshold
    model.eval()
    with torch.no_grad():
        recon_train = model(train_tensor).numpy()
        train_errors = np.mean(np.square(X_train - recon_train), axis=1)

        recon_test = model(test_tensor).numpy()
        test_errors = np.mean(np.square(X_test - recon_test), axis=1)

    # Anomaly Threshold: Mean + 2.5 * Standard Deviation (or 95th percentile)
    threshold = float(np.percentile(train_errors, 95.0))
    print(f"\nCalculated Anomaly Threshold (95th percentile): {threshold:.6f}")

    train_anomalies = int((train_errors > threshold).sum())
    test_anomalies = int((test_errors > threshold).sum())

    print(f"Train Anomalies Flagged: {train_anomalies}/{len(train_errors)} ({train_anomalies/len(train_errors)*100:.2f}%)")
    print(f"Test Anomalies Flagged:  {test_anomalies}/{len(test_errors)} ({test_anomalies/len(test_errors)*100:.2f}%)")

    # 7. Save Model Weights and Evaluation Metrics
    model_save_path = os.path.join(MODELS_DIR, "autoencoder_model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")

    metrics = {
        "model_type": "Autoencoder",
        "training_window": "8 Weeks (56 Days)",
        "input_features": feature_cols,
        "input_dim": input_dim,
        "latent_dim": max(2, input_dim // 2),
        "epochs": epochs,
        "final_train_loss": round(float(epoch_loss), 6),
        "reconstruction_threshold": round(threshold, 6),
        "train_samples": len(train_errors),
        "test_samples": len(test_errors),
        "train_anomalies": train_anomalies,
        "test_anomalies": test_anomalies,
        "anomaly_rate_pct": round(test_anomalies / len(test_errors) * 100, 2)
    }

    metrics_path = os.path.join(MODELS_DIR, "autoencoder_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 8. Save Anomaly Predictions CSV
    test_df_results = test_df[["transaction_id", "user_id", "date", "amount", "category"]].copy()
    test_df_results["reconstruction_error"] = np.round(test_errors, 6)
    test_df_results["is_anomaly"] = test_errors > threshold
    
    results_path = os.path.join(MODELS_DIR, "anomaly_results.csv")
    test_df_results.to_csv(results_path, index=False)
    print(f"Anomaly predictions saved to: {results_path}")

    # 9. Generate Markdown Report
    report_lines = [
        "# Autoencoder Anomaly Detection Benchmark Report (8-Week Training)",
        "",
        "## 1. Model Overview",
        "- **Architecture:** Deep Symmetrical Autoencoder (PyTorch)",
        f"- **Training Context:** 8 Weeks of Historical Transaction Data ({min_date.date()} to {end_date.date()})",
        f"- **Input Dimensions:** {input_dim} features ({', '.join(feature_cols)})",
        f"- **Reconstruction Loss (MSE):** {metrics['final_train_loss']}",
        f"- **Anomaly Threshold:** {metrics['reconstruction_threshold']}",
        "",
        "## 2. Detection Results",
        "| Metric | Training Set (8 Wks) | Out-of-Sample Test Set |",
        "| :--- | :---: | :---: |",
        f"| **Sample Count** | {metrics['train_samples']:,} | {metrics['test_samples']:,} |",
        f"| **Anomalies Flagged** | {metrics['train_anomalies']:,} | {metrics['test_anomalies']:,} |",
        f"| **Anomaly Rate** | 5.00% | {metrics['anomaly_rate_pct']}% |",
        "",
        "## 3. Top Flagged Spending Outliers (Sample)",
        "",
        test_df_results[test_df_results["is_anomaly"]].sort_values("reconstruction_error", ascending=False).head(10).to_markdown(index=False),
        "",
        "## 4. Integration with PostgreSQL Database Schema",
        "- Compatible with `004_ai.sql` schema (`anomalies` table):",
        "  `transaction_id`, `model_id`, `reconstruction_error`, `is_anomaly`",
        "- Can be directly populated via `ml/db_populate.py`."
    ]

    report_path = os.path.join(MODELS_DIR, "autoencoder_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"Report saved to: {report_path}")

    return metrics

if __name__ == "__main__":
    train_autoencoder()
