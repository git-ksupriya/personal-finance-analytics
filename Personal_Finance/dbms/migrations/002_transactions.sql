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


CREATE TABLE transactions (
    -- Internal DB identifier.
    -- Can't use the raw dataset transaction ID as PK
    -- because the dataset contains duplicates.
    transaction_id SERIAL PRIMARY KEY,
    -- Original transaction ID from the dataset.
    source_transaction_id VARCHAR(20),
    user_id VARCHAR(10) NOT NULL,
    account_id INTEGER,
    category_id INTEGER NOT NULL,
    payment_mode_id INTEGER,
    location_id INTEGER,
    amount NUMERIC(12,2) NOT NULL,
    transaction_type VARCHAR(10) NOT NULL,
    transaction_date DATE NOT NULL,
    description TEXT,
    source VARCHAR(10) NOT NULL DEFAULT 'manual',
    upload_id INTEGER,
    -- User must exist.
    CONSTRAINT fk_transactions_user
        FOREIGN KEY (user_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE,
    -- Account must belong to the same user.
    CONSTRAINT fk_transactions_account_user
        FOREIGN KEY (account_id, user_id)
        REFERENCES accounts(account_id, user_id)
        ON DELETE SET NULL (account_id),
    -- Category must exist.
    CONSTRAINT fk_transactions_category
        FOREIGN KEY (category_id)
        REFERENCES categories(category_id),
    -- Payment mode must exist.
    CONSTRAINT fk_transactions_payment_mode
        FOREIGN KEY (payment_mode_id)
        REFERENCES payment_modes(payment_mode_id),
    -- Location must exist.
    CONSTRAINT fk_transactions_location
        FOREIGN KEY (location_id)
        REFERENCES locations(location_id),
    -- Upload must belong to the same user.
    CONSTRAINT fk_transactions_upload_user
        FOREIGN KEY (upload_id, user_id)
        REFERENCES statement_uploads(upload_id, user_id)
        ON DELETE SET NULL (upload_id),
    -- Amount is stored as a positive value.
    -- transaction_type determines Income vs Expense.
    CONSTRAINT chk_transaction_amount
        CHECK (amount > 0),
    CONSTRAINT chk_transaction_type
        CHECK (
            transaction_type IN ('Expense', 'Income')
        ),
    CONSTRAINT chk_transaction_source
        CHECK (
            source IN ('manual', 'csv', 'pdf')
        )
);


CREATE TABLE budgets (
    budget_id SERIAL PRIMARY KEY,
    user_id VARCHAR(10) NOT NULL,
    category_id INTEGER NOT NULL,
    -- Always store first day of month.
    -- Example: 2026-08-01
    month DATE NOT NULL,
    budget_amount NUMERIC(12,2) NOT NULL,
    CONSTRAINT fk_budgets_user
        FOREIGN KEY (user_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_budgets_category
        FOREIGN KEY (category_id)
        REFERENCES categories(category_id),
    CONSTRAINT chk_budget_amount
        CHECK (budget_amount >= 0),
    CONSTRAINT chk_budget_month
        CHECK (EXTRACT(DAY FROM month) = 1),
    -- A user can have only one budget
    -- for a category in a particular month.
    CONSTRAINT uq_user_category_month
        UNIQUE (user_id, category_id, month)
);


CREATE TABLE recurring_payments (
    recurring_id SERIAL PRIMARY KEY,
    user_id VARCHAR(10) NOT NULL,
    category_id INTEGER NOT NULL,
    amount NUMERIC(12,2) NOT NULL,
    frequency VARCHAR(20) NOT NULL,
    next_due_date DATE NOT NULL,
    description TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    CONSTRAINT fk_recurring_user
        FOREIGN KEY (user_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_recurring_category
        FOREIGN KEY (category_id)
        REFERENCES categories(category_id),
    CONSTRAINT chk_recurring_amount
        CHECK (amount > 0),
    CONSTRAINT chk_recurring_frequency
        CHECK (
            frequency IN (
                'daily',
                'weekly',
                'monthly',
                'quarterly',
                'yearly'
            )
        )
);

CREATE TABLE goals (
    goal_id SERIAL PRIMARY KEY,
    user_id VARCHAR(10) NOT NULL,
    goal_name VARCHAR(100) NOT NULL,
    target_amount NUMERIC(12,2) NOT NULL,
    current_amount NUMERIC(12,2) NOT NULL DEFAULT 0,
    target_date DATE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_goals_user
        FOREIGN KEY (user_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE,
    CONSTRAINT chk_goal_target
        CHECK (target_amount > 0),
    CONSTRAINT chk_goal_current
        CHECK (current_amount >= 0),
    CONSTRAINT chk_goal_progress
        CHECK (current_amount <= target_amount)
);