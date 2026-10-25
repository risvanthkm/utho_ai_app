"""Seed telemetry dataset if missing to enable TFT training and inference."""

import math
from pathlib import Path
import numpy as np
import pandas as pd

def generate_telemetry_if_needed(datasets_dir: Path | None = None) -> Path:
    if datasets_dir is None:
        datasets_dir = Path(__file__).resolve().parent.parent / "datasets" / "raw"
    
    parquet_path = datasets_dir / "telemetry.parquet"
    if parquet_path.is_file():
        return parquet_path
    
    inventory_path = datasets_dir / "inventory.csv"
    if not inventory_path.is_file():
        return parquet_path
    
    df_inv = pd.read_csv(inventory_path)
    vm_inv = df_inv[df_inv["type"] == "VM"].copy()
    if vm_inv.empty:
        return parquet_path
    
    # 30 days of 4-hour intervals = 180 points
    end_time = pd.Timestamp("2026-10-24 20:00:00")
    timestamps = pd.date_range(end=end_time, periods=180, freq="4h")
    
    rng = np.random.default_rng(42)
    rows = []
    
    for _, row in vm_inv.iterrows():
        res_id = str(row["resource_id"])
        archetype = str(row.get("archetype", "steady_low"))
        
        # Base CPU and Mem levels according to archetype
        if "burst" in archetype or "daily" in archetype:
            base_cpu, base_mem = 25.0, 45.0
        elif "batch" in archetype:
            base_cpu, base_mem = 15.0, 30.0
        elif "zombie" in archetype or "idle" in archetype:
            base_cpu, base_mem = 1.5, 8.0
        else: # steady_low / standard
            base_cpu, base_mem = 12.0, 24.0
            
        for ts in timestamps:
            hour = ts.hour
            # Diurnal sine wave
            cycle = math.sin(2 * math.pi * (hour - 6) / 24)
            noise = rng.normal(0, 1.5)
            
            if "burst" in archetype and 12 <= hour <= 18:
                cpu = min(98.0, max(2.0, base_cpu * 2.8 + noise * 3))
                mem = min(92.0, max(10.0, base_mem * 1.6 + noise * 2))
            elif "batch" in archetype and 0 <= hour <= 4:
                cpu = min(95.0, max(2.0, 85.0 + noise * 2))
                mem = min(80.0, max(10.0, 60.0 + noise * 2))
            else:
                cpu = min(95.0, max(1.0, base_cpu + cycle * (base_cpu * 0.4) + noise))
                mem = min(95.0, max(5.0, base_mem + cycle * (base_mem * 0.2) + noise))
                
            rows.append({
                "resource_id": res_id,
                "timestamp": ts,
                "cpu_utilization_pct": round(float(cpu), 2),
                "memory_utilization_pct": round(float(mem), 2),
                "disk_utilization_pct": round(float(max(5.0, min(95.0, 35.0 + noise))), 2),
                "disk_iops": int(max(0, 20 + noise * 5)),
                "network_in_mbps": round(float(max(0.1, 5.0 + cycle * 3.0 + noise)), 2),
                "network_out_mbps": round(float(max(0.1, 8.0 + cycle * 4.0 + noise)), 2),
            })
            
    df_telemetry = pd.DataFrame(rows)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    df_telemetry.to_parquet(parquet_path, index=False)
    print(f"Generated {len(df_telemetry)} telemetry records at {parquet_path}")
    return parquet_path

if __name__ == "__main__":
    generate_telemetry_if_needed()
