-- ===========================================
-- SAMPLE DATA LOADER
-- Loads preprocessing/processed/standardised_dataset.csv
-- into users + transactions.
--
-- categories / payment_modes / locations are already
-- seeded by 002_transactions.sql, so this file only
-- needs to map text -> their IDs.
--
-- HOW TO RUN:
--   1. Make sure migrations 001-004 have already been applied.
--   2. Run this file with psql FROM THE Personal_Finance/ DIRECTORY
--      (the \copy path below is relative to where psql is launched,
--      not to this .sql file):
--
--        cd Personal_Finance
--        psql "$DATABASE_URL" -f dbms/seed/sample_data.sql
--
--   \copy is a psql client-side command, so this must be run with
--   the actual psql CLI (works fine against a local DB or a remote
--   one you have a connection string for).
-- ===========================================

BEGIN;

-- 1. Staging table: mirrors the CSV columns as plain text so
--    nothing fails on load; we cast/validate types afterwards.
DROP TABLE IF EXISTS staging_transactions;
CREATE TEMP TABLE staging_transactions (
    transaction_id      TEXT,
    user_id              TEXT,
    txn_date              TEXT,
    transaction_type      TEXT,
    category               TEXT,
    amount                  TEXT,
    payment_mode             TEXT,
    location                  TEXT,
    notes                      TEXT
);

\copy staging_transactions FROM 'preprocessing/processed/standardised_dataset.csv' WITH (FORMAT csv, HEADER true)

-- 2. Populate users from the distinct user_ids in the data.
--    The dataset has no name/email/password, so we synthesize
--    placeholder values -- fine for a seeded demo/dev DB, NOT for prod.
INSERT INTO users (user_id, name, email, password_hash)
SELECT DISTINCT
    user_id,
    'User ' || substring(user_id FROM 2)                 AS name,        -- 'U018' -> 'User 018'
    lower(user_id) || '@example.com'                     AS email,
    '$2b$12$SEEDDATAPLACEHOLDERHASHDONOTUSEINPROD0000000'  AS password_hash
FROM staging_transactions
ON CONFLICT (user_id) DO NOTHING;

-- 3. Populate transactions, resolving category/payment_mode/location
--    text values to their FK ids via the lookup tables.
--    LEFT JOIN on purpose: if any CSV value doesn't match a lookup row,
--    the corresponding *_id comes back NULL and the NOT NULL constraint
--    on transactions.category_id will make the insert fail loudly
--    instead of silently dropping rows (which an INNER JOIN would do).
INSERT INTO transactions (
    source_transaction_id, user_id, category_id, payment_mode_id,
    location_id, amount, transaction_type, transaction_date,
    description, source
)
SELECT
    s.transaction_id,
    s.user_id,
    c.category_id,
    p.payment_mode_id,
    l.location_id,
    s.amount::NUMERIC(12,2),
    s.transaction_type,
    s.txn_date::DATE,
    s.notes,
    'csv'
FROM staging_transactions s
LEFT JOIN categories    c ON c.category_name = s.category
LEFT JOIN payment_modes p ON p.mode_name    = s.payment_mode
LEFT JOIN locations     l ON l.location_name = s.location;

COMMIT;

-- 4. Sanity check -- run manually after loading if you like:
-- SELECT (SELECT COUNT(*) FROM users)        AS users_loaded,
--        (SELECT COUNT(*) FROM transactions) AS transactions_loaded;