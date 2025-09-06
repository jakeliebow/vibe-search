-- Database Tables Setup
CREATE TABLE IF NOT EXISTS rooms (
    id BIGSERIAL PRIMARY KEY,
    room_id UUID DEFAULT gen_random_uuid(),
    user_key CHAR(88) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding vector(512) NOT NULL
);
