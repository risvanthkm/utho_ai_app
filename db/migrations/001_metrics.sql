-- 001_metrics.sql: Idempotent migration for FinSort metrics hypertable and continuous aggregates
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

CREATE TABLE IF NOT EXISTS metrics (
    resource_id VARCHAR(128) NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    cpu_utilization_pct DOUBLE PRECISION NOT NULL CHECK (cpu_utilization_pct >= 0.0 AND cpu_utilization_pct <= 100.0),
    memory_utilization_pct DOUBLE PRECISION NOT NULL CHECK (memory_utilization_pct >= 0.0 AND memory_utilization_pct <= 100.0),
    disk_utilization_pct DOUBLE PRECISION NOT NULL DEFAULT 25.0 CHECK (disk_utilization_pct >= 0.0 AND disk_utilization_pct <= 100.0),
    disk_iops DOUBLE PRECISION NOT NULL DEFAULT 0.0 CHECK (disk_iops >= 0.0),
    network_in_mbps DOUBLE PRECISION NOT NULL DEFAULT 0.0 CHECK (network_in_mbps >= 0.0),
    network_out_mbps DOUBLE PRECISION NOT NULL DEFAULT 0.0 CHECK (network_out_mbps >= 0.0)
);

-- Ensure Hypertable conversion (chunk interval: 1 day)
SELECT create_hypertable(
    'metrics',
    'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Unique index on (resource_id, ts)
CREATE UNIQUE INDEX IF NOT EXISTS uq_metrics_resource_ts
    ON metrics (resource_id, ts);

-- Fast reverse lookup index on (resource_id, ts DESC)
CREATE INDEX IF NOT EXISTS idx_metrics_resource_ts_desc
    ON metrics (resource_id, ts DESC);

-- Daily continuous aggregate view for daily spend/utilization
CREATE MATERIALIZED VIEW IF NOT EXISTS metrics_daily_summary
WITH (timescaledb.continuous) AS
SELECT
    resource_id,
    time_bucket('1 day', ts) AS day,
    avg(cpu_utilization_pct) AS cpu_avg,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY cpu_utilization_pct) AS cpu_p95,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY cpu_utilization_pct) AS cpu_p99,
    avg(memory_utilization_pct) AS mem_avg,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY memory_utilization_pct) AS mem_p95,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY memory_utilization_pct) AS mem_p99,
    count(*) AS sample_count
FROM metrics
GROUP BY resource_id, time_bucket('1 day', ts)
WITH NO DATA;
