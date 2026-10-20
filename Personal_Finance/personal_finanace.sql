-- 1. Store overall CNN model performance

CREATE TABLE IF NOT EXISTS cnn_model_performance (
    model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    mae NUMERIC(15, 4),
    rmse NUMERIC(15, 4),
    r2_score NUMERIC(10, 6),
    forecast_days INTEGER NOT NULL,
    window_size INTEGER NOT NULL,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- 2. Store total spending predictions

CREATE TABLE IF NOT EXISTS cnn_total_spending_predictions (
    prediction_id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES cnn_model_performance(model_id),
    forecast_day INTEGER NOT NULL,
    actual_spending NUMERIC(15, 2),
    predicted_spending NUMERIC(15, 2) NOT NULL,
    absolute_error NUMERIC(15, 2),
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (model_id, forecast_day)
);


-- 3. Store category-wise spending predictions

CREATE TABLE IF NOT EXISTS cnn_category_spending_predictions (
    prediction_id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES cnn_model_performance(model_id),
    forecast_day INTEGER NOT NULL,
    category VARCHAR(100) NOT NULL,
    predicted_spending NUMERIC(15, 2) NOT NULL,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (model_id, forecast_day, category)
);


-- 4. Store category-wise model performance

CREATE TABLE IF NOT EXISTS cnn_category_model_performance (
    performance_id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES cnn_model_performance(model_id),
    category VARCHAR(100) NOT NULL,
    mae NUMERIC(15, 4) NOT NULL,
    UNIQUE (model_id, category)
);
