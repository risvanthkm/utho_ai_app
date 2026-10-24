CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS resource_inventory (
    resource_id text PRIMARY KEY,
    resource_type text NOT NULL,
    plan text NOT NULL,
    region text NOT NULL,
    environment text NOT NULL,
    team text,
    vcpu double precision NOT NULL CHECK (vcpu >= 0),
    ram_gb double precision NOT NULL CHECK (ram_gb >= 0),
    disk_size_gb double precision NOT NULL CHECK (disk_size_gb >= 0),
    attributes jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS resource_inventory_dimensions_idx
    ON resource_inventory (region, environment, team);

CREATE TABLE IF NOT EXISTS pricing_catalog (
    plan text PRIMARY KEY,
    resource_type text NOT NULL,
    vcpu double precision NOT NULL CHECK (vcpu >= 0),
    ram_gb double precision NOT NULL CHECK (ram_gb >= 0),
    rate double precision NOT NULL CHECK (rate >= 0),
    storage_rate double precision NOT NULL CHECK (storage_rate >= 0),
    attributes jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS telemetry (
    resource_id text NOT NULL REFERENCES resource_inventory(resource_id) ON DELETE CASCADE,
    timestamp timestamptz NOT NULL,
    cpu_utilization_pct double precision CHECK (
        cpu_utilization_pct IS NULL OR cpu_utilization_pct BETWEEN 0 AND 100
    ),
    memory_utilization_pct double precision CHECK (
        memory_utilization_pct IS NULL OR memory_utilization_pct BETWEEN 0 AND 100
    ),
    disk_utilization_pct double precision CHECK (
        disk_utilization_pct IS NULL OR disk_utilization_pct BETWEEN 0 AND 100
    ),
    disk_iops double precision CHECK (disk_iops IS NULL OR disk_iops >= 0),
    network_in_mbps double precision CHECK (network_in_mbps IS NULL OR network_in_mbps >= 0),
    network_out_mbps double precision CHECK (network_out_mbps IS NULL OR network_out_mbps >= 0),
    attributes jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (resource_id, timestamp)
);

SELECT create_hypertable(
    'telemetry', 'timestamp', chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS telemetry_timestamp_resource_idx
    ON telemetry (timestamp DESC, resource_id);

CREATE TABLE IF NOT EXISTS chatbot_conversations (
    conversation_id uuid PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chatbot_messages (
    message_id bigserial PRIMARY KEY,
    conversation_id uuid NOT NULL REFERENCES chatbot_conversations(conversation_id)
        ON DELETE CASCADE,
    role text NOT NULL CHECK (role IN ('user', 'assistant')),
    content text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chatbot_messages_conversation_idx
    ON chatbot_messages (conversation_id, message_id);

CREATE TABLE IF NOT EXISTS dataset_imports (
    dataset_name text PRIMARY KEY,
    source_path text NOT NULL,
    imported_rows bigint NOT NULL,
    imported_at timestamptz NOT NULL DEFAULT now()
);
