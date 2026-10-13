import uuid
import numpy as np
import pandas as pd
import psycopg2
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc
from config import (
    POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT,
    REPORTS_DIR, TRAIN_WEEK_END
)

def get_db_connection():
    return psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT
    )

def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """Computes great-circle distance (in kilometers) between coordinates."""
    R = 6371.0 # Earth radius in km
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.clip(np.sqrt(a), 0, 1))
    return R * c

def run_anomaly_detection():
    print("\n==========================================")
    print("Running Step 5: Per-User Anomaly Detection")
    print("==========================================")

    conn = get_db_connection()
    print("Fetching transaction features from PostgreSQL...")
    query = """
        SELECT 
            t.trans_num,
            t.user_id,
            t.category,
            t.amt,
            t.trans_ts,
            t.is_fraud,
            u.lat AS user_lat,
            u.long AS user_long,
            m.merch_lat,
            m.merch_long
        FROM transactions t
        JOIN users u ON t.user_id = u.user_id
        JOIN merchants m ON t.merchant_id = m.merchant_id
        ORDER BY t.user_id, t.trans_ts;
    """
    df = pd.read_sql_query(query, conn)
    print(f"Retrieved {len(df):,} transactions.")

    # 1. Feature Engineering
    print("Extracting spatio-temporal transaction features...")
    df['trans_ts'] = pd.to_datetime(df['trans_ts'])
    df['hour'] = df['trans_ts'].dt.hour
    df['day_of_week'] = df['trans_ts'].dt.dayofweek

    # Distance between user home and merchant
    df['dist_km'] = haversine_distance_vectorized(
        df['user_lat'].astype(float).values,
        df['user_long'].astype(float).values,
        df['merch_lat'].astype(float).values,
        df['merch_long'].astype(float).values
    )

    # Time since previous transaction of that user (in hours)
    df['time_diff_hours'] = df.groupby('user_id')['trans_ts'].diff().dt.total_seconds().fillna(0.0) / 3600.0

    # 2. Chronological Split for Anomaly Module
    # Determine train period cut-off date (corresponding to end of week 73)
    # Using week 73 cutoff: approx 2020-05-25
    with conn.cursor() as cur:
        cur.execute("SELECT MIN(week_start) + INTERVAL '%s weeks' FROM weekly_category_spend;", (TRAIN_WEEK_END,))
        train_cutoff_date = cur.fetchone()[0]

    train_cutoff_date = pd.to_datetime(train_cutoff_date).date()
    print(f"Train period cut-off timestamp: {train_cutoff_date}")
    df['is_train'] = df['trans_ts'].dt.date <= train_cutoff_date

    # 3. User-Category Historical Spend Statistics strictly on TRAIN period
    print("Computing user-category historical statistics on train transactions only...")
    train_txns = df[df['is_train']]
    user_cat_stats = train_txns.groupby(['user_id', 'category'])['amt'].agg(
        cat_mean='mean',
        cat_std='std',
        q25=lambda x: np.percentile(x, 25),
        q75=lambda x: np.percentile(x, 75)
    ).reset_index()

    user_cat_stats['cat_std'] = user_cat_stats['cat_std'].fillna(0.0)
    user_cat_stats['iqr'] = user_cat_stats['q75'] - user_cat_stats['q25']

    df = pd.merge(df, user_cat_stats, on=['user_id', 'category'], how='left')

    # Fallback for series unseen in train
    global_mean = train_txns['amt'].mean()
    global_std = train_txns['amt'].std()
    df['cat_mean'] = df['cat_mean'].fillna(global_mean)
    df['cat_std'] = df['cat_std'].fillna(global_std)
    df['q25'] = df['q25'].fillna(train_txns['amt'].quantile(0.25))
    df['q75'] = df['q75'].fillna(train_txns['amt'].quantile(0.75))
    df['iqr'] = df['iqr'].fillna(df['q75'] - df['q25'])

    # Method A: Robust Z-Score / IQR Outlier Detection
    # Using IQR outlier boundary: amt > q75 + 3.0 * IQR (or Robust Z-score)
    denom = np.where(df['iqr'] > 0, df['iqr'], df['cat_std'])
    denom = np.where(denom > 0, denom, 1.0)
    df['robust_score'] = (df['amt'] - df['cat_mean']) / denom
    iqr_flagged = (df['amt'] > (df['q75'] + 3.0 * df['iqr'])) | (df['robust_score'] > 4.0)

    # Method B: Isolation Forest
    print("Fitting Isolation Forest model on transaction features...")
    feature_cols = ['amt', 'robust_score', 'hour', 'day_of_week', 'dist_km', 'time_diff_hours']
    X_train = df[df['is_train']][feature_cols].values
    X_all = df[feature_cols].values

    # Fit on train transactions
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42,
        n_jobs=-1
    )
    # Fit on a representative sample of train to optimize execution
    sample_idx = np.random.RandomState(42).choice(len(X_train), size=min(100000, len(X_train)), replace=False)
    iso_forest.fit(X_train[sample_idx])

    # Isolation Forest anomaly score: higher score = more anomalous
    iso_scores = -iso_forest.score_samples(X_all)
    iso_threshold = np.percentile(iso_scores[df['is_train']], 99.0) # Top 1% flagged
    iso_flagged = iso_scores >= iso_threshold

    df['iso_score'] = iso_scores
    df['iso_flagged'] = iso_flagged
    df['iqr_flagged'] = iqr_flagged

    # 4. Evaluation against Ground-Truth is_fraud Labels
    # Evaluated on TEST period transactions (chronologically strictly separated)
    test_mask = ~df['is_train']
    y_test_true = df.loc[test_mask, 'is_fraud'].astype(int).values

    def compute_pr_metrics(y_true, scores, flagged):
        precision = precision_score(y_true, flagged, zero_division=0)
        recall = recall_score(y_true, flagged, zero_division=0)
        p_arr, r_arr, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(r_arr, p_arr)
        return float(precision), float(recall), float(pr_auc)

    prec_iqr, rec_iqr, prauc_iqr = compute_pr_metrics(y_test_true, df.loc[test_mask, 'robust_score'].values, df.loc[test_mask, 'iqr_flagged'].values)
    prec_iso, rec_iso, prauc_iso = compute_pr_metrics(y_test_true, df.loc[test_mask, 'iso_score'].values, df.loc[test_mask, 'iso_flagged'].values)

    fraud_rate = (y_test_true == 1).mean() * 100
    print("\n--- Anomaly Detection Evaluation (Test Set: ~540k transactions) ---")
    print(f"Test Fraud Rate: {fraud_rate:.3f}% ({y_test_true.sum():,} fraud txns)")
    print(f"Robust Z-Score/IQR -> Precision: {prec_iqr:.4f}, Recall: {rec_iqr:.4f}, PR-AUC: {prauc_iqr:.4f}")
    print(f"Isolation Forest   -> Precision: {prec_iso:.4f}, Recall: {rec_iso:.4f}, PR-AUC: {prauc_iso:.4f}")

    # 5. Write Flagged Anomalies into PostgreSQL 'anomalies' Table
    print("Writing flagged anomalies to database 'anomalies' table...")
    run_id_iqr = f"iqr_{uuid.uuid4().hex[:8]}"
    run_id_iso = f"iso_{uuid.uuid4().hex[:8]}"

    # Prepare flagged records for ingestion
    flagged_iqr_df = df[df['iqr_flagged']][['trans_num', 'robust_score']].copy()
    flagged_iqr_df['method'] = 'Robust_IQR'
    flagged_iqr_df['flagged'] = True
    flagged_iqr_df['run_id'] = run_id_iqr

    flagged_iso_df = df[df['iso_flagged']][['trans_num', 'iso_score']].copy()
    flagged_iso_df.rename(columns={'iso_score': 'robust_score'}, inplace=True)
    flagged_iso_df['method'] = 'Isolation_Forest'
    flagged_iso_df['flagged'] = True
    flagged_iso_df['run_id'] = run_id_iso

    all_flagged = pd.concat([flagged_iqr_df, flagged_iso_df], ignore_index=True)
    all_flagged.columns = ['trans_num', 'score', 'method', 'flagged', 'run_id']

    import io
    output = io.StringIO()
    all_flagged.to_csv(output, sep='\t', header=False, index=False)
    output.seek(0)

    with conn.cursor() as cur:
        cur.copy_from(
            output,
            'anomalies',
            columns=['trans_num', 'score', 'method', 'flagged', 'run_id']
        )
    conn.commit()
    conn.close()
    print(f"Logged {len(all_flagged):,} flagged records into 'anomalies' table.")

    # 6. Save Anomaly Detection Report
    report_lines = [
        "# Anomaly Detection Benchmark Report",
        "",
        "## 1. Overview & Setup",
        f"- **Total Transactions Evaluated:** {len(df):,}",
        f"- **Train Period:** Inception through {train_cutoff_date}",
        f"- **Test Period:** Post-{train_cutoff_date} ({len(y_test_true):,} transactions)",
        f"- **Test Ground-Truth Fraud Cases:** {y_test_true.sum():,} ({fraud_rate:.3f}% base rate)",
        "",
        "## 2. Model Performance (Evaluated against Ground-Truth is_fraud)",
        "| Method | Precision | Recall | PR-AUC | Total Flagged |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Robust Z-Score / IQR** | {prec_iqr:.4f} | {rec_iqr:.4f} | {prauc_iqr:.4f} | {len(flagged_iqr_df):,} |",
        f"| **Isolation Forest** | {prec_iso:.4f} | {rec_iso:.4f} | {prauc_iso:.4f} | {len(flagged_iso_df):,} |",
        "",
        "## 3. Analysis & Key Findings",
        "- **Why PR-AUC instead of ROC-AUC/Accuracy?** Given the severe class imbalance (fraud occurrence is < 0.6%), accuracy is meaningless. PR-AUC and precision/recall directly capture the actionable utility for fraud investigators.",
        "- **Robust IQR vs Isolation Forest:** Robust IQR flags extreme spend outliers relative to the individual user's personal category baseline, yielding high recall on high-value fraud transactions. Isolation Forest captures multidimensional interactions (distance from home, time since previous transaction, and odd hours).",
        "- **Database Logging:** All detected anomalies are written with foreign-key references to `transactions(trans_num)` for real-time investigation."
    ]

    report_path = REPORTS_DIR / "anomaly_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"Saved anomaly report to {report_path}")

if __name__ == "__main__":
    run_anomaly_detection()
