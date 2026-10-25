"""Train Temporal Quantile Transformer on telemetry with MAE vs Baseline validation."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.services.tft_forecasting import QuantileTemporalTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def quantile_loss(preds: torch.Tensor, target: torch.Tensor, q: float) -> torch.Tensor:
    errors = target - preds
    return torch.max((q - 1) * errors, q * errors).mean()


def train():
    start_time = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Training Temporal Quantile Model on: {device} ({gpu_name})")

    parquet_path = PROJECT_ROOT / "data" / "history.parquet"
    if not parquet_path.is_file():
        parquet_path = PROJECT_ROOT / "datasets" / "raw" / "telemetry.parquet"

    df = pd.read_parquet(parquet_path)
    ts_col = "ts" if "ts" in df.columns else "timestamp"
    df["day"] = pd.to_datetime(df[ts_col]).dt.floor("1D")

    enc_len = 14
    pred_len = 90
    total_win = enc_len + pred_len

    train_x_list, train_y_list = [], []
    val_x_list, val_y_list = [], []

    for _, grp in df.groupby("resource_id"):
        daily_r = (
            grp.groupby("day")
            .agg(
                cpu_mean=("cpu_utilization_pct", "mean"),
                cpu_max=("cpu_utilization_pct", "max"),
                mem_mean=("memory_utilization_pct", "mean"),
                mem_max=("memory_utilization_pct", "max"),
            )
            .sort_index()
        )
        s = daily_r[["cpu_mean", "cpu_max", "mem_mean", "mem_max"]].values.astype(np.float32)
        if len(s) < total_win + 4:
            reps = int(np.ceil((total_win + 6) / max(1, len(s))))
            s = np.tile(s, (reps, 1))[: total_win + 6]

        n_win = len(s) - total_win + 1
        for i in range(n_win - 1):
            train_x_list.append(s[i : i + enc_len])
            train_y_list.append(s[i + enc_len : i + total_win, 0])
        # Final time window per resource is held out for validation
        val_x_list.append(s[n_win - 1 : n_win - 1 + enc_len])
        val_y_list.append(s[n_win - 1 + enc_len : n_win - 1 + total_win, 0])

    train_x = torch.tensor(np.array(train_x_list), dtype=torch.float32)
    train_y = torch.tensor(np.array(train_y_list), dtype=torch.float32)
    val_x = torch.tensor(np.array(val_x_list), dtype=torch.float32)
    val_y = torch.tensor(np.array(val_y_list), dtype=torch.float32)

    model = QuantileTemporalTransformer(in_features=4, hidden_dim=48, n_heads=4, pred_len=pred_len).to(device)
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=min(256, len(dataset)), shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    best_val_mae = float("inf")
    best_state = None
    epochs_run = 0

    for epoch in range(20):
        if time.time() - start_time > 60:
            break
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            p10, p50, p90 = model(bx)
            loss = (
                quantile_loss(p10, by, 0.1)
                + 2.5 * quantile_loss(p50, by, 0.5)
                + quantile_loss(p90, by, 0.9)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        epochs_run = epoch + 1

        model.eval()
        with torch.no_grad():
            vx, vy = val_x.to(device), val_y.to(device)
            _, vp50, _ = model(vx)
            cur_mae = float(torch.abs(vy - vp50).mean().item())
            if cur_mae < best_val_mae:
                best_val_mae = cur_mae
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        vx = val_x.to(device)
        _, p50_val, _ = model(vx)
        val_pred = p50_val.cpu().numpy()
        val_actual = val_y.numpy()
        val_mae = float(np.mean(np.abs(val_actual - val_pred)))

        # Seasonal-naive baseline: repeat last known encoder day across horizon
        last_enc_day = val_x[:, -1:, 0].numpy()
        baseline_pred = np.repeat(last_enc_day, pred_len, axis=1)
        baseline_mae = float(np.mean(np.abs(val_actual - baseline_pred)))

    elapsed = round(time.time() - start_time, 2)
    print(f"Trained {epochs_run} epochs in {elapsed}s | Val MAE: {val_mae:.4f} vs Baseline MAE: {baseline_mae:.4f}")

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "tft.ckpt"
    torch.save(model.cpu().state_dict(), ckpt_path)

    meta = {
        "name": "QuantileTemporalTransformer-90d",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "gpu_name": gpu_name,
        "in_features": 4,
        "hidden_dim": 48,
        "n_heads": 4,
        "epochs_run": epochs_run,
        "training_seconds": elapsed,
        "val_mae": round(val_mae, 4),
        "baseline_mae": round(baseline_mae, 4),
        "data_source": "telemetry_parquet",
        "encoder_days": enc_len,
        "prediction_days": pred_len,
        "horizons": [30, 35, 40, 60, 90],
        "is_trained": True,
    }
    meta_path = models_dir / "tft_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Checkpoint saved to {ckpt_path}")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    train()
