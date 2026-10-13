# Preprocessing & Dataset Summary Report

## 1. Overall Statistics
- **Total Raw Weekly Rows:** 1,382,955
- **Zero-Spend Weeks Count:** 526,489 (38.07%)
- **Unique Users:** 999
- **Unique Categories:** 14
- **Total (User, Category) Series:** 13,171
- **Kept Series (>= 10 non-zero weeks in Train):** 12,504 (94.94%)
- **Dropped Series:** 667 (5.06%)

## 2. Chronological Split Setup
- **Train Weeks:** 1 to 73 (Total 73 weeks)
- **Validation Weeks:** 74 to 89 (Total 16 weeks)
- **Test Weeks:** 90 to 105 (Total 16 weeks)
- **Leakage Prevention:** All user-level statistics (mean, std of weekly spend) and scalers were fit strictly on Train weeks (1-73).

## 3. Per-Category Sparsity & Breakdown
| Category | Total Series | Kept Series | Dropped Series | Zero-Spend Rate (%) | Train Mean Spend ($) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| entertainment | 922 | 908 | 14 | 33.14% | $82.59 |
| food_dining | 918 | 908 | 10 | 34.19% | $65.38 |
| gas_transport | 967 | 902 | 65 | 31.61% | $111.02 |
| grocery_net | 920 | 734 | 186 | 60.80% | $34.08 |
| grocery_pos | 992 | 907 | 85 | 31.38% | $180.55 |
| health_fitness | 910 | 908 | 2 | 35.59% | $65.89 |
| home | 911 | 908 | 3 | 25.03% | $100.75 |
| kids_pets | 916 | 908 | 8 | 27.15% | $91.48 |
| misc_net | 982 | 901 | 81 | 50.40% | $57.58 |
| misc_pos | 926 | 907 | 19 | 39.60% | $68.78 |
| personal_care | 919 | 908 | 11 | 34.70% | $60.97 |
| shopping_net | 993 | 908 | 85 | 39.15% | $89.78 |
| shopping_pos | 973 | 908 | 65 | 32.09% | $113.03 |
| travel | 922 | 889 | 33 | 58.17% | $62.83 |
