-- transactions module

CREATE TABLE categories (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE payment_modes (
    payment_mode_id SERIAL PRIMARY KEY,
    mode_name VARCHAR(30) NOT NULL UNIQUE
);

CREATE TABLE locations (
    location_id SERIAL PRIMARY KEY,
    location_name VARCHAR(100) NOT NULL UNIQUE
);



CREATE TABLE accounts (
    account_id SERIAL PRIMARY KEY,
    user_id VARCHAR(10) NOT NULL,
    bank_name VARCHAR(100) NOT NULL,
    account_type VARCHAR(30) NOT NULL,
    CONSTRAINT fk_accounts_user
        FOREIGN KEY (user_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE,
    CONSTRAINT chk_account_type
        CHECK (account_type IN (
            'Savings',
            'Current',
            'Credit',
            'Other'
        )),
    -- Required so (account_id, user_id)
    -- can be referenced by transactions.
    CONSTRAINT uq_account_user
        UNIQUE (account_id, user_id)
);

CREATE TABLE statement_uploads (
    upload_id SERIAL PRIMARY KEY,
    user_id VARCHAR(10) NOT NULL,
    blob_url TEXT NOT NULL,
    file_name VARCHAR(255),
    file_type VARCHAR(10),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    uploaded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    CONSTRAINT fk_uploads_user
        FOREIGN KEY (user_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE,
    CONSTRAINT chk_upload_status
        CHECK (status IN (
            'pending',
            'processed',
            'failed'
        )),
    CONSTRAINT chk_file_type
        CHECK (
            file_type IS NULL
            OR file_type IN ('csv', 'pdf')
        ),
    -- Required for the composite FK from transactions.
    CONSTRAINT uq_upload_user
        UNIQUE (upload_id, user_id)
);