-- ===========================================
-- SPENDING PREDICTIONS
-- ===========================================

CREATE TABLE spending_predictions (
    prediction_id SERIAL PRIMARY KEY,

    user_id VARCHAR(10) NOT NULL,

    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    forecast_date DATE NOT NULL,

    prediction_type VARCHAR(20) NOT NULL
        CHECK (prediction_type IN ('overall', 'category')),

    category VARCHAR(100),

    predicted_amount DECIMAL(14,2) NOT NULL,

    model_version VARCHAR(100),

    FOREIGN KEY (user_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE,

    CHECK (
        (prediction_type = 'overall' AND category IS NULL)
        OR
        (prediction_type = 'category' AND category IS NOT NULL)
    )
);

CREATE INDEX idx_spending_predictions_user_date
ON spending_predictions(user_id, forecast_date);

CREATE INDEX idx_spending_predictions_generated
ON spending_predictions(user_id, prediction_date);

