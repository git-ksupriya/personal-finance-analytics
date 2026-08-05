-- 003_analytics.sql — Analytics domain (computed rollups only)
-- Depends on: users, transactions/categories/goals (so is subject to change depending on udated code for users and transactions

-- 1. Monthly income/expense/savings summary per user
CREATE TABLE monthly_summary (
    user_id         VARCHAR(10) REFERENCES users(user_id),
    month           DATE NOT NULL,              -- store as first-of-month, e.g. '2024-01-01'
    total_income    NUMERIC(12,2) DEFAULT 0,
    total_expense   NUMERIC(12,2) DEFAULT 0,
    savings_rate    NUMERIC(5,2),                -- percentage
    updated_at      TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_id, month)
);

-- 2. Category-wise monthly spend per user
CREATE TABLE category_monthly_summary (
    user_id         VARCHAR(10) REFERENCES users(user_id),
    category_id     INT REFERENCES categories(category_id),
    month           DATE NOT NULL,
    total_spent     NUMERIC(12,2) DEFAULT 0,
    txn_count       INT DEFAULT 0,
    updated_at      TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_id, category_id, month)
);

-- 3. Recurring payments detected from transaction patterns
CREATE TABLE detected_recurring_payments (
    recurring_id      SERIAL PRIMARY KEY,
    user_id           VARCHAR(10) REFERENCES users(user_id),
    category_id       INT REFERENCES categories(category_id),
    amount_estimate   NUMERIC(12,2),
    frequency         VARCHAR(20),                -- 'monthly', 'weekly'
    months_seen       INT,
    last_detected_at  TIMESTAMP DEFAULT NOW()
);

-- 4. Progress tracking against user-declared goals
CREATE TABLE goal_progress (
    goal_id         INT REFERENCES goals(goal_id),
    as_of_month     DATE NOT NULL,
    current_amount  NUMERIC(12,2) DEFAULT 0,
    progress_pct    NUMERIC(5,2),
    updated_at      TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (goal_id, as_of_month)
);

-- Indexes to support fast dashboard reads
CREATE INDEX idx_monthly_summary_user       ON monthly_summary (user_id);
CREATE INDEX idx_cat_monthly_summary_user   ON category_monthly_summary (user_id);
CREATE INDEX idx_recurring_user             ON detected_recurring_payments (user_id);
CREATE INDEX idx_goal_progress_goal         ON goal_progress (goal_id);