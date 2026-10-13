-- PostgreSQL 15+ Normalized Schema for Personalised Spending Analytics

-- Drop existing tables in reverse dependency order
DROP TABLE IF EXISTS anomalies CASCADE;
DROP TABLE IF EXISTS forecasts CASCADE;
DROP TABLE IF EXISTS model_runs CASCADE;
DROP TABLE IF EXISTS weekly_category_spend CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS merchants CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- 1. Users table (deduplicated by cc_num)
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    cc_num BIGINT UNIQUE NOT NULL,
    first VARCHAR(64),
    last VARCHAR(64),
    gender VARCHAR(10),
    dob DATE,
    job VARCHAR(128),
    street VARCHAR(128),
    city VARCHAR(64),
    state VARCHAR(32),
    zip VARCHAR(16),
    lat NUMERIC(9, 6),
    long NUMERIC(9, 6),
    city_pop INT
);

-- 2. Merchants table (deduplicated by merchant_name)
CREATE TABLE merchants (
    merchant_id SERIAL PRIMARY KEY,
    merchant_name VARCHAR(128) UNIQUE NOT NULL,
    category VARCHAR(64),
    merch_lat NUMERIC(9, 6),
    merch_long NUMERIC(9, 6)
);

-- 3. Transactions table
CREATE TABLE transactions (
    trans_num VARCHAR(64) PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    merchant_id INT NOT NULL REFERENCES merchants(merchant_id) ON DELETE CASCADE,
    category VARCHAR(64) NOT NULL,
    amt NUMERIC(12, 2) NOT NULL,
    trans_ts TIMESTAMP NOT NULL,
    is_fraud BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX idx_transactions_user_ts ON transactions (user_id, trans_ts);
CREATE INDEX idx_transactions_user_cat ON transactions (user_id, category);
CREATE INDEX idx_transactions_ts ON transactions (trans_ts);

-- 4. Weekly Category Spend Table (Complete weekly series with zero-fill)
CREATE TABLE weekly_category_spend (
    user_id INT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    category VARCHAR(64) NOT NULL,
    week_start DATE NOT NULL,
    total_amt NUMERIC(12, 2) NOT NULL DEFAULT 0.00,
    txn_count INT NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, category, week_start)
);

CREATE INDEX idx_wcs_user_cat ON weekly_category_spend (user_id, category);
CREATE INDEX idx_wcs_week_start ON weekly_category_spend (week_start);

-- 5. Forecasts table
CREATE TABLE forecasts (
    forecast_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    category VARCHAR(64) NOT NULL,
    target_week DATE NOT NULL,
    model_name VARCHAR(64) NOT NULL,
    p_spend NUMERIC(6, 4) NOT NULL,
    predicted_amt NUMERIC(12, 2) NOT NULL,
    run_id VARCHAR(64) NOT NULL
);

CREATE INDEX idx_forecasts_run ON forecasts (run_id);
CREATE INDEX idx_forecasts_user_cat ON forecasts (user_id, category);

-- 6. Model runs metadata table
CREATE TABLE model_runs (
    run_id VARCHAR(64) PRIMARY KEY,
    model_name VARCHAR(64) NOT NULL,
    hyperparams JSONB NOT NULL,
    metrics JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 7. Anomalies table
CREATE TABLE anomalies (
    anomaly_id SERIAL PRIMARY KEY,
    trans_num VARCHAR(64) NOT NULL REFERENCES transactions(trans_num) ON DELETE CASCADE,
    score NUMERIC(16, 6) NOT NULL,
    method VARCHAR(64) NOT NULL,
    flagged BOOLEAN NOT NULL,
    run_id VARCHAR(64) NOT NULL
);

CREATE INDEX idx_anomalies_run ON anomalies (run_id);
CREATE INDEX idx_anomalies_trans ON anomalies (trans_num);
