-- ===========================================
-- MIGRATION 005: Auth enhancements, weekly budgets, prediction upserts
-- ===========================================

-- 1. Enhance users table for authentication & onboarding flow
ALTER TABLE users ADD COLUMN IF NOT EXISTS username VARCHAR(50) UNIQUE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS onboarded BOOLEAN NOT NULL DEFAULT FALSE;

-- If users table has rows without a username, set username default to user_id
UPDATE users SET username = user_id WHERE username IS NULL;

-- 2. Create weekly_budgets table
CREATE TABLE IF NOT EXISTS weekly_budgets (
    budget_id SERIAL PRIMARY KEY,
    user_id VARCHAR(10) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES categories(category_id) ON DELETE CASCADE,
    weekly_amount NUMERIC(12, 2) NOT NULL CHECK (weekly_amount >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Unique constraint ensuring one weekly budget per user per category, and one overall budget (category_id IS NULL)
CREATE UNIQUE INDEX IF NOT EXISTS uq_weekly_budget_overall
    ON weekly_budgets (user_id)
    WHERE category_id IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_weekly_budget_category
    ON weekly_budgets (user_id, category_id)
    WHERE category_id IS NOT NULL;

-- 3. Enhance predictions table for idempotent upserts
-- Ensure model_registry model_name uniqueness
CREATE UNIQUE INDEX IF NOT EXISTS uq_model_registry_name
    ON model_registry (model_name);

-- Index for upserting predictions
CREATE UNIQUE INDEX IF NOT EXISTS uq_predictions_overall
    ON predictions (user_id, prediction_type, target_period)
    WHERE category_id IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_predictions_category
    ON predictions (user_id, prediction_type, category_id, target_period)
    WHERE category_id IS NOT NULL;
