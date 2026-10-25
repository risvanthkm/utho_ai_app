"""SQLAlchemy models for FinSort TimescaleDB metrics."""

from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, DateTime, Float, Index, String, UniqueConstraint
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class MetricModel(Base):
    """SQLAlchemy model mapped 1:1 to metrics hypertable."""

    __tablename__ = "metrics"

    resource_id = Column(String(128), primary_key=True, nullable=False)
    ts = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    cpu_utilization_pct = Column(Float, nullable=False)
    memory_utilization_pct = Column(Float, nullable=False)
    disk_utilization_pct = Column(Float, nullable=False, default=25.0)
    disk_iops = Column(Float, nullable=False, default=0.0)
    network_in_mbps = Column(Float, nullable=False, default=0.0)
    network_out_mbps = Column(Float, nullable=False, default=0.0)

    __table_args__ = (
        UniqueConstraint("resource_id", "ts", name="uq_metrics_resource_ts"),
        Index("idx_metrics_resource_ts_desc", "resource_id", ts.desc()),
    )
