-- ===========================================
-- USERS TABLE
-- ===========================================

CREATE TABLE users (
    user_id VARCHAR(10) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    phone VARCHAR(15),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- ===========================================
-- USER SESSIONS TABLE
-- ===========================================

CREATE TABLE user_sessions (
    session_id SERIAL PRIMARY KEY,

    user_id VARCHAR(10) NOT NULL,

    token TEXT NOT NULL,

    device_info VARCHAR(100),

    expires_at TIMESTAMP NOT NULL,

    FOREIGN KEY(user_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE
);
