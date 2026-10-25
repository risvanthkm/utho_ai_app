"""Train Temporal Quantile Transformer on synthesized telemetry with MAE vs Baseline validation."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class QuantileTemporalTransformer(nn.Module):
    """Compact Temporal Quantile Transformer for Multi-Horizon P10/P50/P90 forecasting."""

    def __init__(self, in_features: int = 4, hidden_dim: int = 48, n_heads: int = 4, pred_len: int = 40):
        super().__init__()
        self.pred_len = pred_len
        self.embedding = nn.Linear(in_features, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 14, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # 3 quantiles: P10, P50, P90
        self.head_p10 = nn.Linear(hidden_dim, pred_len)
        self.head_p50 = nn.Linear(hidden_dim, pred_len)
        self.head_p90 = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, 14, in_features]
        h = self.embedding(x) + self.pos_encoder
        h = self.transformer(h)
        # Pool across sequence length: take last timestep
        pooled = h[:, -1, :]
        p10 = self.head_p10(pooled)
        p50 = self.head_p50(pooled)
        p90 = self.head_p90(pooled)

        # Enforce monotonic quantile invariant: p10 <= p50 <= p90
        p50_adj = torch.maximum(p50, p10)
        p90_adj = torch.maximum(p90, p50_adj)
        return p10, p50_adj, p90_adj


def quantile_loss(preds: torch.Tensor, target: torch.Tensor, q: float) -> torch.Tensor:
    errors = target - preds
    return torch.max((q - 1) * errors, q * errors).mean()


def train():
    start_time = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Temporal Quantile Model on: {device}")

    # Load daily aggregated metrics
    parquet_path = PROJECT_ROOT / "data" / "history.parquet"
    if not parquet_path.is_file():
        parquet_path = PROJECT_ROOT / "datasets" / "raw" / "telemetry.parquet"

    df = pd.read_parquet(parquet_path)
    df["ts"] = pd.to_datetime(df["ts"])
    df["day"] = df["ts"].dt.floor("1D")

    # Aggregate daily spend/utilization
    daily = df.groupby(["day"]).agg({
        "cpu_utilization_pct": ["mean", "max"],
        "memory_utilization_pct": ["mean", "max"],
    }).reset_index()
    daily.columns = ["day", "cpu_mean", "cpu_max", "mem_mean", "mem_max"]
    daily = daily.sort_values("day").reset_index(drop=True)

    series = daily[["cpu_mean", "cpu_max", "mem_mean", "mem_max"]].values.astype(np.float32)

    # If series is shorter than 54 days (14 enc + 40 pred), tile/interpolate smoothly
    if len(series) < 54:
        reps = int(np.ceil(60 / len(series)))
        series = np.tile(series, (reps, 1))[:60]

    # Create sliding windows of 14 days input -> 40 days target
    X_list, y_list = [], []
    for i in range(len(series) - 54 + 1):
        X_list.append(series[i : i + 14])
        y_list.append(series[i + 14 : i + 54, 0])  # Forecast target: cpu_mean trace

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)

    # Time-based split: last sample is validation (never shuffled across time)
    train_x, val_x = X[:-1], X[-1:]
    train_y, val_y = y[:-1], y[-1:]

    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)

    model = QuantileTemporalTransformer(in_features=4, hidden_dim=48, n_heads=4, pred_len=40).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Train for <=30 epochs with 5-minute cap
    model.train()
    for epoch in range(30):
        if time.time() - start_time > 300:
            print("Reached 5-minute wall clock cap. Stopping training.")
            break
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            p10, p50, p90 = model(bx)
            loss = (
                quantile_loss(p10, by, 0.1)
                + quantile_loss(p50, by, 0.5)
                + quantile_loss(p90, by, 0.9)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Validation Evaluation
    model.eval()
    with torch.no_grad():
        vx = val_x.to(device)
        _, p50_val, _ = model(vx)
        val_pred = p50_val.cpu().numpy()[0]
        val_actual = val_y.numpy()[0]
        val_mae = float(np.mean(np.abs(val_actual - val_pred)))

        # Seasonal-naive baseline: repeat last known day
        baseline_pred = np.full_like(val_actual, series[13, 0])
        baseline_mae = float(np.mean(np.abs(val_actual - baseline_pred)))

    print(f"Validation MAE: {val_mae:.4f} vs Seasonal-Naive Baseline MAE: {baseline_mae:.4f}")

    # Save model checkpoint and metadata
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "tft.ckpt"
    torch.save(model.state_dict(), ckpt_path)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "val_mae": round(val_mae, 4),
        "baseline_mae": round(baseline_mae, 4),
        "data_source": "simulated",
        "encoder_days": 14,
        "prediction_days": 40,
        "is_trained": True,
    }
    meta_path = models_dir / "tft_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Checkpoint saved to {ckpt_path}")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    train()
