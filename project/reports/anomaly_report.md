# Anomaly Detection Benchmark Report

## 1. Overview & Setup
- **Total Transactions Evaluated:** 1,852,394
- **Train Period:** Inception through 2020-05-25
- **Test Period:** Post-2020-05-25 (629,306 transactions)
- **Test Ground-Truth Fraud Cases:** 2,556 (0.406% base rate)

## 2. Model Performance (Evaluated against Ground-Truth is_fraud)
| Method | Precision | Recall | PR-AUC | Total Flagged |
| :--- | :---: | :---: | :---: | :---: |
| **Robust Z-Score / IQR** | 0.0790 | 0.7394 | 0.1078 | 69,206 |
| **Isolation Forest** | 0.1961 | 0.4292 | 0.1271 | 17,826 |

## 3. Analysis & Key Findings
- **Why PR-AUC instead of ROC-AUC/Accuracy?** Given the severe class imbalance (fraud occurrence is < 0.6%), accuracy is meaningless. PR-AUC and precision/recall directly capture the actionable utility for fraud investigators.
- **Robust IQR vs Isolation Forest:** Robust IQR flags extreme spend outliers relative to the individual user's personal category baseline, yielding high recall on high-value fraud transactions. Isolation Forest captures multidimensional interactions (distance from home, time since previous transaction, and odd hours).
- **Database Logging:** All detected anomalies are written with foreign-key references to `transactions(trans_num)` for real-time investigation.
