"""Standardized telemetry synthesizer and validator emitting FinSort schema."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from app.models.schemas import StandardMetricRecord


def generate_synthesized_metrics(output_dir: Path | None = None) -> pd.DataFrame:
    """Generate deterministic, strict Pydantic-validated metrics for 85 resources."""
    base_dir = Path(__file__).resolve().parent.parent
    if output_dir is None:
        output_dir = base_dir / "data"

    inventory_path = base_dir / "datasets" / "raw" / "inventory.csv"
    if inventory_path.is_file():
        df_inv = pd.read_csv(inventory_path)
    else:
        # Fallback inventory of 65 VMs, 12 Disks, 8 IPs = 85 resources
        vms = [{"resource_id": f"res-vm-{i:03d}", "type": "VM", "archetype": "steady_low"} for i in range(1, 66)]
        df_inv = pd.DataFrame(vms)

    vm_inv = df_inv[df_inv["type"] == "VM"].copy()
    if vm_inv.empty:
        vm_inv = df_inv.copy()

    # 30 days of 4-hour intervals = 180 points per VM
    end_time = datetime(2026, 10, 24, 20, 0, 0, tzinfo=timezone.utc)
    timestamps = pd.date_range(end=end_time, periods=180, freq="4h", tz=timezone.utc)

    rng = np.random.default_rng(42)
    records: list[dict] = []

    for _, row in vm_inv.iterrows():
        res_id = str(row["resource_id"])
        archetype = str(row.get("archetype", "steady_low"))

        if "burst" in archetype or "daily" in archetype:
            base_cpu, base_mem = 25.0, 45.0
        elif "batch" in archetype:
            base_cpu, base_mem = 15.0, 30.0
        elif "zombie" in archetype or "idle" in archetype:
            base_cpu, base_mem = 1.5, 8.0
        else:
            base_cpu, base_mem = 12.0, 24.0

        for ts in timestamps:
            hour = ts.hour
            cycle = math.sin(2 * math.pi * (hour - 6) / 24)
            noise = float(rng.normal(0, 1.5))

            if "burst" in archetype and 12 <= hour <= 18:
                cpu = min(98.0, max(2.0, base_cpu * 2.8 + noise * 3))
                mem = min(92.0, max(10.0, base_mem * 1.6 + noise * 2))
            elif "batch" in archetype and 0 <= hour <= 4:
                cpu = min(95.0, max(2.0, 85.0 + noise * 2))
                mem = min(80.0, max(10.0, 60.0 + noise * 2))
            else:
                cpu = min(95.0, max(1.0, base_cpu + cycle * (base_cpu * 0.4) + noise))
                mem = min(95.0, max(5.0, base_mem + cycle * (base_mem * 0.2) + noise))

            # Strict Pydantic model validation for each record
            record = StandardMetricRecord(
                resource_id=res_id,
                ts=ts.to_pydatetime(),
                cpu_utilization_pct=round(float(cpu), 2),
                memory_utilization_pct=round(float(mem), 2),
                disk_utilization_pct=round(float(max(5.0, min(95.0, 35.0 + noise))), 2),
                disk_iops=float(max(0.0, 20.0 + noise * 5)),
                network_in_mbps=round(float(max(0.1, 5.0 + cycle * 3.0 + noise)), 2),
                network_out_mbps=round(float(max(0.1, 8.0 + cycle * 4.0 + noise)), 2),
            )
            d = record.model_dump()
            d["timestamp"] = d["ts"]  # Guarantee 100% compatibility with timestamp column
            records.append(d)

    df = pd.DataFrame(records)

    # Emit to data/history.parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    history_parquet = output_dir / "history.parquet"
    df.to_parquet(history_parquet, index=False)

    # Also emit to datasets/raw/telemetry.parquet
    raw_dir = base_dir / "datasets" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_dir / "telemetry.parquet", index=False)

    return df


def generate_telemetry_if_needed(datasets_dir: Path | None = None) -> Path:
    base_dir = Path(__file__).resolve().parent.parent
    history_file = base_dir / "data" / "history.parquet"
    if not history_file.is_file():
        generate_synthesized_metrics()
    return history_file


if __name__ == "__main__":
    df = generate_synthesized_metrics()
    print(f"Generated and validated {len(df)} records. Range: {df['ts'].min()} to {df['ts'].max()}")
