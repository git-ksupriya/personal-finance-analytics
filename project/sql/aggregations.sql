-- Aggregations script: Generates zero-filled weekly series for all (user_id, category)
-- Total weeks = 105 (Monday 2018-12-31 to Monday 2020-12-28)

TRUNCATE TABLE weekly_category_spend;

WITH calendar_weeks AS (
    SELECT generate_series(
        '2018-12-31'::date,
        '2020-12-28'::date,
        '1 week'::interval
    )::date AS week_start
),
active_series AS (
    SELECT DISTINCT user_id, category
    FROM transactions
),
full_grid AS (
    SELECT s.user_id, s.category, w.week_start
    FROM active_series s
    CROSS JOIN calendar_weeks w
),
actual_weekly_spend AS (
    SELECT 
        user_id,
        category,
        date_trunc('week', trans_ts)::date AS week_start,
        SUM(amt) AS total_amt,
        COUNT(*) AS txn_count
    FROM transactions
    WHERE is_fraud = FALSE
    GROUP BY user_id, category, date_trunc('week', trans_ts)::date
)
INSERT INTO weekly_category_spend (user_id, category, week_start, total_amt, txn_count)
SELECT 
    g.user_id,
    g.category,
    g.week_start,
    COALESCE(a.total_amt, 0.00) AS total_amt,
    COALESCE(a.txn_count, 0) AS txn_count
FROM full_grid g
LEFT JOIN actual_weekly_spend a 
    ON g.user_id = a.user_id 
   AND g.category = a.category 
   AND g.week_start = a.week_start
ORDER BY g.user_id, g.category, g.week_start;
