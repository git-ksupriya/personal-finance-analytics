# Anomaly Detection Benchmark Report (Feature/Full-App-V2 Methodology)

## 1. Overview & Setup
- **Methodology Source:** `feature/full-app-v2` (`spending_analytics.py`)
- **Technique:** Multi-Tier Robust Thresholding (Tukey's Upper Fence Q3 + 1.5 * IQR & Scaled MAD)
- **Dataset:** `fraudTrain.csv` (150,000 transaction sample)
- **Historical Training Window:** 8 Weeks (2019-01-01 to 2019-02-26)
- **Evaluation Period Transactions:** 54,174

## 2. Detection Performance Summary
| Metric | Value |
| :--- | :---: |
| **Test Transactions Evaluated** | 54,174 |
| **Anomalies Flagged** | 6,143 (11.34%) |
| **Ground-Truth Fraud Cases in Test** | 434 |
| **Fraud Recall (True Positive Rate)** | 0.5945 (59.4%) |
| **Precision** | 0.0420 |

## 3. Sample Top Flagged Transactions

|          user_id | category     | date       |   amount |   threshold |   excess_amt | method            |
|-----------------:|:-------------|:-----------|---------:|------------:|-------------:|:------------------|
| 4633065159406313 | travel       | 2019-03-02 | 11629.3  |        9.45 |     11619.9  | tukey_upper_fence |
|    4745996322265 | travel       | 2019-03-05 |  9931.82 |       14.73 |      9917.09 | tukey_upper_fence |
| 2286236465059468 | travel       | 2019-03-15 |  9455.15 |        7.53 |      9447.62 | tukey_upper_fence |
| 2248348993063230 | travel       | 2019-03-20 |  8762.37 |       10.91 |      8751.45 | tukey_upper_fence |
|  213148039875802 | travel       | 2019-03-11 |  8428.57 |        7.75 |      8420.82 | tukey_upper_fence |
| 6011367958204270 | travel       | 2019-03-20 |  8524.52 |      269.92 |      8254.6  | tukey_upper_fence |
| 6011893664860915 | travel       | 2019-03-16 |  8217.23 |        8.52 |      8208.71 | tukey_upper_fence |
| 4302480582202074 | shopping_net | 2019-03-10 |  6818.74 |       37.26 |      6781.48 | tukey_upper_fence |
| 3556613125071656 | travel       | 2019-03-16 |  6756.43 |       11.32 |      6745.11 | tukey_upper_fence |
| 3591919803438423 | travel       | 2019-03-03 |  5579.69 |      414.8  |      5164.89 | tukey_upper_fence |

## 4. Key Strengths of Full-App-V2 Methodology
- **Non-parametric & Robust:** Does not assume normal distribution, preventing single outlier distortion.
- **Zero-dispersion Handling:** Seamlessly falls back to Scaled MAD or median doubling when series variance is zero.
- **Per-User Category Resolution:** An amount that is standard for `travel` or `grocery_pos` is not falsely flagged, while an abnormal jump in `entertainment` is detected.
