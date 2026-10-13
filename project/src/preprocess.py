import time
import numpy as np
import pandas as pd
import psycopg2
from config import (
    POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT,
    MODELS_DIR, REPORTS_DIR, MIN_NONZERO_WEEKS,
    TRAIN_WEEK_END, VAL_WEEK_END, TEST_WEEK_END, SEQ_LENS
)

def get_connection():
    return psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT
    )

def fetch_weekly_data():
    conn = get_connection()
    query = """
        SELECT user_id, category, week_start, total_amt, txn_count
        FROM weekly_category_spend
        ORDER BY user_id, category, week_start;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def generate_preprocessing_report(df, kept_series_mask, series_stats, categories):
    total_series = len(series_stats)
    kept_count = kept_series_mask.sum()
    dropped_count = total_series - kept_count

    total_rows = len(df)
    zero_rows = (df['total_amt'] == 0).sum()
    overall_zero_rate = (zero_rows / total_rows) * 100

    report_lines = [
        "# Preprocessing & Dataset Summary Report",
        "",
        "## 1. Overall Statistics",
        f"- **Total Raw Weekly Rows:** {total_rows:,}",
        f"- **Zero-Spend Weeks Count:** {zero_rows:,} ({overall_zero_rate:.2f}%)",
        f"- **Unique Users:** {df['user_id'].nunique():,}",
        f"- **Unique Categories:** {len(categories)}",
        f"- **Total (User, Category) Series:** {total_series:,}",
        f"- **Kept Series (>= {MIN_NONZERO_WEEKS} non-zero weeks in Train):** {kept_count:,} ({kept_count/total_series*100:.2f}%)",
        f"- **Dropped Series:** {dropped_count:,} ({dropped_count/total_series*100:.2f}%)",
        "",
        "## 2. Chronological Split Setup",
        f"- **Train Weeks:** 1 to {TRAIN_WEEK_END} (Total 73 weeks)",
        f"- **Validation Weeks:** {TRAIN_WEEK_END + 1} to {VAL_WEEK_END} (Total 16 weeks)",
        f"- **Test Weeks:** {VAL_WEEK_END + 1} to {TEST_WEEK_END} (Total 16 weeks)",
        "- **Leakage Prevention:** All user-level statistics (mean, std of weekly spend) and scalers were fit strictly on Train weeks (1-73).",
        "",
        "## 3. Per-Category Sparsity & Breakdown",
        "| Category | Total Series | Kept Series | Dropped Series | Zero-Spend Rate (%) | Train Mean Spend ($) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    for cat in sorted(categories):
        cat_df = df[df['category'] == cat]
        cat_zero_rate = (cat_df['total_amt'] == 0).mean() * 100
        cat_stats = series_stats[series_stats['category'] == cat]
        c_total = len(cat_stats)
        c_kept = (cat_stats['train_nonzero'] >= MIN_NONZERO_WEEKS).sum()
        c_drop = c_total - c_kept
        c_mean = cat_df[cat_df['week_idx'] <= TRAIN_WEEK_END]['total_amt'].mean()
        report_lines.append(f"| {cat} | {c_total:,} | {c_kept:,} | {c_drop:,} | {cat_zero_rate:.2f}% | ${c_mean:.2f} |")

    report_path = REPORTS_DIR / "preprocessing_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"Saved preprocessing report to {report_path}")

def run_preprocessing():
    print("Fetching weekly data from PostgreSQL...")
    df = fetch_weekly_data()
    df['week_start'] = pd.to_datetime(df['week_start'])

    # Map sorted unique weeks to 1..105 index
    unique_weeks = sorted(df['week_start'].unique())
    week2idx = {w: i + 1 for i, w in enumerate(unique_weeks)}
    df['week_idx'] = df['week_start'].map(week2idx)

    # Encode categories
    categories = sorted(df['category'].unique())
    cat2idx = {cat: i for i, cat in enumerate(categories)}
    df['cat_idx'] = df['category'].map(cat2idx)

    # Encode users
    users = sorted(df['user_id'].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    df['u_idx'] = df['user_id'].map(user2idx)

    print(f"Loaded {len(df):,} weekly records across {len(unique_weeks)} weeks.")

    # 1. Filter series: Keep only (user, category) series with >= MIN_NONZERO_WEEKS in train
    train_mask = df['week_idx'] <= TRAIN_WEEK_END
    train_df = df[train_mask]
    nonzero_train = train_df[train_df['total_amt'] > 0].groupby(['user_id', 'category']).size().reset_index(name='train_nonzero')

    all_series = df[['user_id', 'category']].drop_duplicates()
    series_stats = pd.merge(all_series, nonzero_train, on=['user_id', 'category'], how='left')
    series_stats['train_nonzero'] = series_stats['train_nonzero'].fillna(0).astype(int)

    kept_series_mask = series_stats['train_nonzero'] >= MIN_NONZERO_WEEKS
    valid_series = set(zip(
        series_stats[kept_series_mask]['user_id'],
        series_stats[kept_series_mask]['category']
    ))
    print(f"Filtering series: Kept {len(valid_series):,} of {len(series_stats):,} series (Dropped {(~kept_series_mask).sum():,})")

    # Generate and save preprocessing report
    generate_preprocessing_report(df, kept_series_mask, series_stats, categories)

    # 2. Compute User-Level Static Stats on TRAIN weeks ONLY
    user_train_stats = train_df.groupby('user_id')['total_amt'].agg(['mean', 'std']).reset_index()
    user_train_stats['std'] = user_train_stats['std'].fillna(0.0)
    user_train_stats.rename(columns={'mean': 'user_spend_mean', 'std': 'user_spend_std'}, inplace=True)

    # Global normalization for user stats
    u_mean_m = user_train_stats['user_spend_mean'].mean()
    u_mean_s = user_train_stats['user_spend_mean'].std() + 1e-6
    u_std_m = user_train_stats['user_spend_std'].mean()
    u_std_s = user_train_stats['user_spend_std'].std() + 1e-6

    user_train_stats['user_spend_mean_norm'] = (user_train_stats['user_spend_mean'] - u_mean_m) / u_mean_s
    user_train_stats['user_spend_std_norm'] = (user_train_stats['user_spend_std'] - u_std_m) / u_std_s

    user_stats_map = {
        row['user_id']: (row['user_spend_mean_norm'], row['user_spend_std_norm'])
        for _, row in user_train_stats.iterrows()
    }

    # 3. Time-step Input Features:
    # log1p(weekly_amt), log1p(txn_count), has_spend flag, sin(week_of_year), cos(week_of_year)
    df['feat_log_amt'] = np.log1p(df['total_amt'].astype(float).values)
    df['feat_log_count'] = np.log1p(df['txn_count'].astype(float).values)
    df['feat_has_spend'] = (df['total_amt'].values > 0).astype(float)

    # Sin/Cos of calendar week-of-year (1..52)
    woy = df['week_start'].dt.isocalendar().week.astype(float)
    df['feat_sin_woy'] = np.sin(2 * np.pi * woy / 52.0)
    df['feat_cos_woy'] = np.cos(2 * np.pi * woy / 52.0)

    # Target features: next-week has_spend (binary) and next-week log1p(amt)
    # We will build sliding windows per (user, category) series

    filtered_df = df[df.set_index(['user_id', 'category']).index.isin(valid_series)].copy()
    filtered_df.sort_values(by=['user_id', 'category', 'week_idx'], inplace=True)

    dynamic_cols = ['feat_log_amt', 'feat_log_count', 'feat_has_spend', 'feat_sin_woy', 'feat_cos_woy']

    # Build windows for multiple sequence lengths [8, 10, 12]
    for seq_len in SEQ_LENS:
        print(f"\nConstructing sliding windows for seq_len = {seq_len}...")
        X_dynamic_train, X_static_train, y_clf_train, y_reg_train = [], [], [], []
        X_dynamic_val, X_static_val, y_clf_val, y_reg_val = [], [], [], []
        X_dynamic_test, X_static_test, y_clf_test, y_reg_test = [], [], [], []

        # Keep metadata for evaluation and database forecasting writes
        meta_test = [] # (user_id, category, target_week_idx, target_week_date, actual_amt)

        grouped = filtered_df.groupby(['user_id', 'category'])
        for (u_id, cat), group in grouped:
            c_idx = cat2idx[cat]
            u_mean_val, u_std_val = user_stats_map.get(u_id, (0.0, 0.0))
            static_feat = np.array([c_idx, u_mean_val, u_std_val], dtype=np.float32)

            dyn_mat = group[dynamic_cols].values.astype(np.float32) # shape: (105, 5)
            amt_arr = group['total_amt'].values.astype(np.float32)
            has_spend_arr = group['feat_has_spend'].values.astype(np.float32)
            log_amt_arr = group['feat_log_amt'].values.astype(np.float32)
            week_starts = group['week_start'].values

            n_weeks = len(group)
            # Sliding windows: history [t-seq_len : t], target is week t
            for t in range(seq_len, n_weeks):
                target_week_idx = t + 1 # 1-indexed (week 1 to 105)
                x_dyn = dyn_mat[t - seq_len : t]
                y_c = has_spend_arr[t]
                y_r = log_amt_arr[t]

                # Chronological split without crossing boundaries:
                # Train: target_week_idx in [seq_len + 1, TRAIN_WEEK_END]
                # Val: target_week_idx in [TRAIN_WEEK_END + 1, VAL_WEEK_END]
                # Test: target_week_idx in [VAL_WEEK_END + 1, TEST_WEEK_END]
                if target_week_idx <= TRAIN_WEEK_END:
                    X_dynamic_train.append(x_dyn)
                    X_static_train.append(static_feat)
                    y_clf_train.append(y_c)
                    y_reg_train.append(y_r)
                elif target_week_idx <= VAL_WEEK_END:
                    X_dynamic_val.append(x_dyn)
                    X_static_val.append(static_feat)
                    y_clf_val.append(y_c)
                    y_reg_val.append(y_r)
                elif target_week_idx <= TEST_WEEK_END:
                    X_dynamic_test.append(x_dyn)
                    X_static_test.append(static_feat)
                    y_clf_test.append(y_c)
                    y_reg_test.append(y_r)
                    meta_test.append({
                        'user_id': u_id,
                        'category': cat,
                        'target_week': str(pd.to_datetime(week_starts[t]).date()),
                        'actual_amt': float(amt_arr[t]),
                        'last_week_amt': float(amt_arr[t - 1]),
                        'ma4_amt': float(amt_arr[max(0, t - 4) : t].mean())
                    })

        # Convert to numpy arrays
        tensors = {
            'X_dyn_train': np.array(X_dynamic_train, dtype=np.float32),
            'X_stat_train': np.array(X_static_train, dtype=np.float32),
            'y_clf_train': np.array(y_clf_train, dtype=np.float32),
            'y_reg_train': np.array(y_reg_train, dtype=np.float32),

            'X_dyn_val': np.array(X_dynamic_val, dtype=np.float32),
            'X_stat_val': np.array(X_static_val, dtype=np.float32),
            'y_clf_val': np.array(y_clf_val, dtype=np.float32),
            'y_reg_val': np.array(y_reg_val, dtype=np.float32),

            'X_dyn_test': np.array(X_dynamic_test, dtype=np.float32),
            'X_stat_test': np.array(X_static_test, dtype=np.float32),
            'y_clf_test': np.array(y_clf_test, dtype=np.float32),
            'y_reg_test': np.array(y_reg_test, dtype=np.float32),
        }

        save_prefix = MODELS_DIR / f"processed_w{seq_len}"
        np.savez_compressed(f"{save_prefix}.npz", **tensors)
        pd.DataFrame(meta_test).to_csv(f"{save_prefix}_test_meta.csv", index=False)
        print(f"Saved window {seq_len} dataset: Train={len(y_clf_train):,}, Val={len(y_clf_val):,}, Test={len(y_clf_test):,}")

    # Save mapping metadata
    import json
    with open(MODELS_DIR / "mappings.json", "w") as f:
        json.dump({
            'categories': categories,
            'cat2idx': cat2idx,
            'num_categories': len(categories)
        }, f, indent=2)

    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    run_preprocessing()
