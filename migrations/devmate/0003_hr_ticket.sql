-- Reference Application 的真实内部 HR 工单表。
CREATE TABLE IF NOT EXISTS hr_tickets (
    id VARCHAR(64) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    priority VARCHAR(32) NOT NULL DEFAULT 'medium',
    category VARCHAR(64) NOT NULL DEFAULT '其他',
    status VARCHAR(32) NOT NULL DEFAULT 'created',
    created_by VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_hr_ticket_tenant ON hr_tickets (tenant_id);
CREATE INDEX IF NOT EXISTS ix_hr_ticket_status ON hr_tickets (status);
CREATE INDEX IF NOT EXISTS ix_hr_ticket_created_by ON hr_tickets (created_by);
