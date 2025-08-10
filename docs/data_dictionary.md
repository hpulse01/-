# Data Dictionary

## users
- id: int, PK
- email: string(255), unique
- password_hash: string(255)
- role: string(50), enum[user, admin]
- created_at: timestamptz, default now()

## audit_logs
- id: int, PK
- user_id: int, FK users.id nullable
- action: string(100)
- details: text
- created_at: timestamptz, default now()

## predictions
- id: int, PK
- user_id: int, FK users.id
- subject_name: string(255)
- seed: string(255) nullable
- result_summary: text
- embedding: vector(3) nullable (requires pgvector extension)
- created_at: timestamptz, default now()