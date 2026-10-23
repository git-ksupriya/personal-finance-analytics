-- Preserve legacy recurring rows with an unknown type instead of guessing.
ALTER TABLE recurring_payments
    ADD COLUMN IF NOT EXISTS payment_type VARCHAR(12);

ALTER TABLE recurring_payments
    DROP CONSTRAINT IF EXISTS chk_recurring_payment_type;

ALTER TABLE recurring_payments
    ADD CONSTRAINT chk_recurring_payment_type
    CHECK (payment_type IS NULL OR payment_type IN ('income', 'expenditure'));

-- Persist alert scope and lifecycle while retaining existing alert history.
ALTER TABLE alerts
    ADD COLUMN IF NOT EXISTS scope VARCHAR(100),
    ADD COLUMN IF NOT EXISTS category_id INTEGER,
    ADD COLUMN IF NOT EXISTS target_period DATE,
    ADD COLUMN IF NOT EXISTS dedupe_key TEXT,
    ADD COLUMN IF NOT EXISTS status VARCHAR(12) NOT NULL DEFAULT 'active',
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMPTZ;

ALTER TABLE alerts
    DROP CONSTRAINT IF EXISTS fk_alerts_category,
    DROP CONSTRAINT IF EXISTS chk_alert_status;

ALTER TABLE alerts
    ADD CONSTRAINT fk_alerts_category
        FOREIGN KEY (category_id) REFERENCES categories(category_id),
    ADD CONSTRAINT chk_alert_status
        CHECK (status IN ('active', 'resolved', 'read'));

CREATE UNIQUE INDEX IF NOT EXISTS uq_alerts_user_dedupe
    ON alerts (user_id, dedupe_key)
    WHERE dedupe_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_alerts_user_status_created
    ON alerts (user_id, status, created_at DESC);

-- Preserve existing anomaly rows and infer ownership where possible.
ALTER TABLE anomalies
    ALTER COLUMN transaction_id DROP NOT NULL,
    ALTER COLUMN model_id DROP NOT NULL;

ALTER TABLE anomalies
    ADD COLUMN IF NOT EXISTS user_id VARCHAR(10),
    ADD COLUMN IF NOT EXISTS prediction_id INTEGER,
    ADD COLUMN IF NOT EXISTS category_id INTEGER,
    ADD COLUMN IF NOT EXISTS scope VARCHAR(100),
    ADD COLUMN IF NOT EXISTS detection_type VARCHAR(12),
    ADD COLUMN IF NOT EXISTS target_period DATE,
    ADD COLUMN IF NOT EXISTS observed_amount NUMERIC(12,2),
    ADD COLUMN IF NOT EXISTS historical_mean NUMERIC(12,2),
    ADD COLUMN IF NOT EXISTS first_quartile NUMERIC(12,2),
    ADD COLUMN IF NOT EXISTS third_quartile NUMERIC(12,2),
    ADD COLUMN IF NOT EXISTS threshold_amount NUMERIC(12,2),
    ADD COLUMN IF NOT EXISTS history_count INTEGER,
    ADD COLUMN IF NOT EXISTS detection_method VARCHAR(80),
    ADD COLUMN IF NOT EXISTS reason TEXT,
    ADD COLUMN IF NOT EXISTS event_key TEXT,
    ADD COLUMN IF NOT EXISTS status VARCHAR(12) NOT NULL DEFAULT 'active',
    ADD COLUMN IF NOT EXISTS detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMPTZ;

UPDATE anomalies a
SET user_id = t.user_id
FROM transactions t
WHERE a.transaction_id = t.transaction_id
  AND a.user_id IS NULL;

ALTER TABLE anomalies
    DROP CONSTRAINT IF EXISTS fk_anomalies_user,
    DROP CONSTRAINT IF EXISTS fk_anomalies_prediction,
    DROP CONSTRAINT IF EXISTS fk_anomalies_category,
    DROP CONSTRAINT IF EXISTS chk_anomaly_status,
    DROP CONSTRAINT IF EXISTS chk_anomaly_detection_type;

ALTER TABLE anomalies
    ADD CONSTRAINT fk_anomalies_user
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    ADD CONSTRAINT fk_anomalies_prediction
        FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id) ON DELETE SET NULL,
    ADD CONSTRAINT fk_anomalies_category
        FOREIGN KEY (category_id) REFERENCES categories(category_id),
    ADD CONSTRAINT chk_anomaly_status
        CHECK (status IN ('active', 'resolved')),
    ADD CONSTRAINT chk_anomaly_detection_type
        CHECK (detection_type IS NULL OR detection_type IN ('actual', 'predicted'));

CREATE UNIQUE INDEX IF NOT EXISTS uq_anomalies_user_event
    ON anomalies (user_id, event_key)
    WHERE user_id IS NOT NULL AND event_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_anomalies_user_status_detected
    ON anomalies (user_id, status, detected_at DESC);

-- Cache inference for the latest persisted transaction set, including unavailable results.
CREATE TABLE IF NOT EXISTS inference_runs (
    user_id VARCHAR(10) PRIMARY KEY
        REFERENCES users(user_id) ON DELETE CASCADE,
    source_transaction_id INTEGER NOT NULL DEFAULT 0,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    overall_result JSONB NOT NULL,
    category_result JSONB NOT NULL
);
