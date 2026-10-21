# Autoencoder Anomaly Detection Benchmark Report (8-Week Training)

## 1. Model Overview
- **Architecture:** Deep Symmetrical Autoencoder (PyTorch)
- **Training Context:** 8 Weeks of Historical Transaction Data (2021-01-01 to 2021-02-26)
- **Input Dimensions:** 6 features (amount, days_since_last_txn, monthly_total_spend, category_monthly_spend, user_txn_frequency, rolling_avg_3)
- **Reconstruction Loss (MSE):** 0.146968
- **Anomaly Threshold:** 0.174037

## 2. Detection Results
| Metric | Training Set (8 Wks) | Out-of-Sample Test Set |
| :--- | :---: | :---: |
| **Sample Count** | 547 | 13,433 |
| **Anomalies Flagged** | 28 | 1,094 |
| **Anomaly Rate** | 5.00% | 8.14% |

## 3. Top Flagged Spending Outliers (Sample)

| transaction_id   | user_id   | date                |   amount | category      |   reconstruction_error | is_anomaly   |
|:-----------------|:----------|:--------------------|---------:|:--------------|-----------------------:|:-------------|
| T5577            | U026      | 2024-02-01 00:00:00 |    17210 | Rent          |                19.6237 | True         |
| T1930            | U049      | 2024-11-17 00:00:00 |     6401 | Education     |                16.9835 | True         |
| T4950            | U117      | 2023-01-16 00:00:00 |      943 | Entertainment |                13.0381 | True         |
| T5083            | U117      | 2022-12-14 00:00:00 |     8886 | Food          |                12.9956 | True         |
| T1154            | U005      | 2021-08-03 00:00:00 |      917 | Food          |                12.7236 | True         |
| T12614           | U131      | 2022-11-24 00:00:00 |     3148 | Utilities     |                12.6187 | True         |
| T3052            | U026      | 2024-02-25 00:00:00 |     5162 | Rent          |                11.3891 | True         |
| T8851            | U055      | 2021-08-04 00:00:00 |     7208 | Food          |                11.3641 | True         |
| T6706            | U072      | 2022-03-15 00:00:00 |     3987 | Food          |                11.111  | True         |
| T5140            | U091      | 2023-06-12 00:00:00 |     1313 | Entertainment |                11.0214 | True         |

## 4. Integration with PostgreSQL Database Schema
- Compatible with `004_ai.sql` schema (`anomalies` table):
  `transaction_id`, `model_id`, `reconstruction_error`, `is_anomaly`
- Can be directly populated via `ml/db_populate.py`.
