-- ===========================================
-- USER BUDGETS
-- ===========================================

CREATE TABLE user_budgets (
    budget_id SERIAL PRIMARY KEY,

    user_id VARCHAR(10) NOT NULL,

    budget_type VARCHAR(20) NOT NULL
        CHECK (budget_type IN ('overall', 'category')),

    category VARCHAR(100),

    budget_amount DECIMAL(12,2) NOT NULL
        CHECK (budget_amount > 0),

    period_type VARCHAR(20) NOT NULL DEFAULT 'weekly'
        CHECK (period_type = 'weekly'),

    start_date DATE NOT NULL,

    end_date DATE NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE,

    CHECK (end_date >= start_date),

    CHECK (
        (budget_type = 'overall' AND category IS NULL)
        OR
        (budget_type = 'category' AND category IS NOT NULL)
    )
);

CREATE INDEX idx_user_budgets_user_dates
ON user_budgets(user_id, start_date, end_date);
