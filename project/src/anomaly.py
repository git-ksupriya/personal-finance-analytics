import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Exact Anomaly Detection Methodology from feature/full-app-v2
# ---------------------------------------------------------

def robust_threshold(history):
    """
    Robust spending thresholding methodology from feature/full-app-v2:
    - Primary: Tukey's Upper Fence (Q3 + 1.5 * IQR)
    - Fallback A: Median + 3 * Scaled MAD (if IQR == 0 but MAD > 0)
    - Fallback B: Twice Median (if zero dispersion / discrete spending)
    """
    values = np.asarray(history, dtype=np.float64)
    if len(values) == 0:
        return {"threshold": 0.0, "q1": 0.0, "q3": 0.0, "mean": 0.0, "method": "no_history"}

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

def run_anomaly_detection(data_path="c:/Users/megha/Downloads/Spakov_DL/fraudTrain.csv", sample_rows=150000):
    print("=" * 65)
    print("RUNNING ANOMALY DETECTION (Tukey IQR & Robust Thresholding)")
    print("Implementation directly aligned with feature/full-app-v2")
    print("=" * 65)

    print(f"Loading transactions from {data_path}...")
    df = pd.read_csv(
        data_path,
        usecols=["trans_date_trans_time", "cc_num", "category", "amt", "is_fraud", "trans_num"],
        nrows=sample_rows
    )
    df["date"] = pd.to_datetime(df["trans_date_trans_time"]).dt.date
    df = df.sort_values(["cc_num", "trans_date_trans_time"])

    print(f"Loaded {len(df):,} transactions across {df['cc_num'].nunique():,} users and {df['category'].nunique():,} categories.")

    # 1. Split Chronologically: 8-Week Training Period (56 days) vs Evaluation Period
    min_date = df["date"].min()
    split_date = min_date + pd.Timedelta(weeks=8)
    
    train_mask = df["date"] < split_date
    test_mask = df["date"] >= split_date

    train_df = df[train_mask]
    test_df = df[test_mask]

    print(f"Train period (8 weeks): {min_date} to {split_date} ({len(train_df):,} transactions)")
    print(f"Test period: post-{split_date} ({len(test_df):,} transactions)")

    # 2. Compute Robust Baselines per User & Category on Train Period
    user_cat_baselines = {}
    user_overall_baselines = {}

    print("Computing user-category historical baselines from 8-week training data...")
    for (user, cat), group in train_df.groupby(["cc_num", "category"]):
        history = group["amt"].values
        user_cat_baselines[(user, cat)] = robust_threshold(history)

    for user, group in train_df.groupby("cc_num"):
        history = group["amt"].values
        user_overall_baselines[user] = robust_threshold(history)

    # 3. Detect Anomalies on Evaluation Period
    print("Evaluating test transactions for spending anomalies...")
    anomalies = []
    
    for idx, row in test_df.iterrows():
        user = row["cc_num"]
        cat = row["category"]
        amt = float(row["amt"])
        
        # Look up baseline (user-category specific, fallback to user overall)
        baseline = user_cat_baselines.get((user, cat)) or user_overall_baselines.get(user)
        if baseline is None:
            continue
            
        threshold = baseline["threshold"]
        is_anom = amt > threshold
        
        if is_anom:
            anomalies.append({
                "trans_num": row["trans_num"],
                "user_id": user,
                "category": cat,
                "date": row["date"],
                "amount": amt,
                "threshold": round(threshold, 2),
                "excess_amt": round(amt - threshold, 2),
                "method": baseline["method"],
                "is_ground_truth_fraud": bool(row["is_fraud"])
            })

    anom_df = pd.DataFrame(anomalies)
    print(f"\nTotal Anomalies Flagged in Test Period: {len(anom_df):,} / {len(test_df):,} ({len(anom_df)/len(test_df)*100:.2f}%)")

    # Evaluate against ground truth fraud if available
    if "is_ground_truth_fraud" in anom_df.columns and len(anom_df) > 0:
        true_positives = anom_df["is_ground_truth_fraud"].sum()
        total_ground_truth_fraud = test_df["is_fraud"].sum()
        precision = true_positives / len(anom_df) if len(anom_df) > 0 else 0.0
        recall = true_positives / total_ground_truth_fraud if total_ground_truth_fraud > 0 else 0.0
        print(f"Ground-Truth Fraud in Test: {total_ground_truth_fraud:,}")
        print(f"Correctly Flagged (True Positives): {true_positives:,}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f}")

    # 4. Save Flagged Anomalies CSV
    anom_output_path = MODELS_DIR / "anomalies_detected_v2.csv"
    anom_df.to_csv(anom_output_path, index=False)
    print(f"Saved anomalies dataset to {anom_output_path}")

    # 5. Generate Markdown Report
    top_samples = anom_df.sort_values("excess_amt", ascending=False).head(10)
    report_lines = [
        "# Anomaly Detection Benchmark Report (Feature/Full-App-V2 Methodology)",
        "",
        "## 1. Overview & Setup",
        "- **Methodology Source:** `feature/full-app-v2` (`spending_analytics.py`)",
        "- **Technique:** Multi-Tier Robust Thresholding (Tukey's Upper Fence Q3 + 1.5 * IQR & Scaled MAD)",
        f"- **Dataset:** `fraudTrain.csv` (150,000 transaction sample)",
        f"- **Historical Training Window:** 8 Weeks ({min_date} to {split_date})",
        f"- **Evaluation Period Transactions:** {len(test_df):,}",
        "",
        "## 2. Detection Performance Summary",
        "| Metric | Value |",
        "| :--- | :---: |",
        f"| **Test Transactions Evaluated** | {len(test_df):,} |",
        f"| **Anomalies Flagged** | {len(anom_df):,} ({len(anom_df)/len(test_df)*100:.2f}%) |",
        f"| **Ground-Truth Fraud Cases in Test** | {test_df['is_fraud'].sum():,} |",
        f"| **Fraud Recall (True Positive Rate)** | {recall:.4f} ({recall*100:.1f}%) |",
        f"| **Precision** | {precision:.4f} |",
        "",
        "## 3. Sample Top Flagged Transactions",
        "",
        top_samples[["user_id", "category", "date", "amount", "threshold", "excess_amt", "method"]].to_markdown(index=False),
        "",
        "## 4. Key Strengths of Full-App-V2 Methodology",
        "- **Non-parametric & Robust:** Does not assume normal distribution, preventing single outlier distortion.",
        "- **Zero-dispersion Handling:** Seamlessly falls back to Scaled MAD or median doubling when series variance is zero.",
        "- **Per-User Category Resolution:** An amount that is standard for `travel` or `grocery_pos` is not falsely flagged, while an abnormal jump in `entertainment` is detected."
    ]

    report_path = REPORTS_DIR / "anomaly_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"Saved anomaly report to {report_path}")

if __name__ == "__main__":
    run_anomaly_detection()
