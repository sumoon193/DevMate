-- 长期记忆治理字段：内容哈希、重要性、访问统计与 TTL。
CREATE EXTENSION IF NOT EXISTS pgcrypto;

ALTER TABLE episodic_memories
    ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64);
ALTER TABLE episodic_memories
    ADD COLUMN IF NOT EXISTS importance_score DOUBLE PRECISION NOT NULL DEFAULT 0.5;
ALTER TABLE episodic_memories
    ADD COLUMN IF NOT EXISTS access_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE episodic_memories
    ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMPTZ NULL;

UPDATE episodic_memories
SET content_hash = encode(digest(lower(trim(content)), 'sha256'), 'hex')
WHERE content_hash IS NULL;

ALTER TABLE episodic_memories
    ALTER COLUMN content_hash SET NOT NULL;

CREATE INDEX IF NOT EXISTS ix_episodic_memory_tenant_hash
    ON episodic_memories (tenant_id, memory_key, content_hash);
