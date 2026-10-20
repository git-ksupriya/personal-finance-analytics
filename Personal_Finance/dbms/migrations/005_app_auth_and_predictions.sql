-- Application login identity and prediction generation metadata.
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS username VARCHAR(64);

UPDATE users
SET username = 'legacy_' || user_id
WHERE username IS NULL;

ALTER TABLE users
    ALTER COLUMN username SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_ci
    ON users (LOWER(username));

ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

CREATE INDEX IF NOT EXISTS idx_predictions_user_generated
    ON predictions (user_id, generated_at DESC);
