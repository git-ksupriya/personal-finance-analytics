-- ===========================================
-- BUDGET ALERTS
-- ===========================================

CREATE TABLE budget_alerts (
    alert_id SERIAL PRIMARY KEY,

    user_id VARCHAR(10) NOT NULL,

    budget_id INTEGER REFERENCES user_budgets(budget_id)
        ON DELETE CASCADE,

    alert_type VARCHAR(30) NOT NULL
        CHECK (
            alert_type IN (
                'predicted_excess',
                'budget_exceeded'
            )
        ),

    message TEXT NOT NULL,

    actual_amount DECIMAL(14,2) DEFAULT 0,

    predicted_amount DECIMAL(14,2),

    budget_amount DECIMAL(14,2) NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    is_read BOOLEAN DEFAULT FALSE,

    FOREIGN KEY (user_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE
);

CREATE INDEX idx_budget_alerts_user_created
ON budget_alerts(user_id, created_at DESC);

