"""Production inference service for Temporal Quantile Transformer."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class QuantileTemporalTransformer(nn.Module):
    """Compact Temporal Quantile Transformer with per-series standardization & residual quantile heads."""

    def __init__(self, in_features: int = 4, hidden_dim: int = 48, n_heads: int = 4, pred_len: int = 90):
        super().__init__()
        self.pred_len = pred_len
        self.embedding = nn.Linear(in_features, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 14, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head_p10 = nn.Linear(hidden_dim, pred_len)
        self.head_p50 = nn.Linear(hidden_dim, pred_len)
        self.head_p90 = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Per-series standardization across the 14-day encoder window
        seq_mean = x.mean(dim=1, keepdim=True)
        seq_std = torch.clamp(x.std(dim=1, keepdim=True, unbiased=False), min=1e-2)
        x_norm = (x - seq_mean) / seq_std

        # Target anchor & scale (feature 0 = cpu_mean)
        anchor = seq_mean[:, :, 0]  # (B, 1)
        scale = seq_std[:, :, 0]    # (B, 1)

        h = self.embedding(x_norm) + self.pos_encoder
        h = self.transformer(h)
        pooled = h[:, -1, :]

        p10 = anchor + self.head_p10(pooled) * scale
        p50 = anchor + self.head_p50(pooled) * scale
        p90 = anchor + self.head_p90(pooled) * scale

        p50_adj = torch.maximum(p50, p10)
        p90_adj = torch.maximum(p90, p50_adj)
        return p10, p50_adj, p90_adj


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CKPT_PATH = PROJECT_ROOT / "models" / "tft.ckpt"
META_PATH = PROJECT_ROOT / "models" / "tft_meta.json"

_cached_model: QuantileTemporalTransformer | None = None
_cached_meta: dict | None = None


def _find_checkpoint_files() -> tuple[Path | None, Path | None]:
    candidates = [
        (CKPT_PATH, META_PATH),
        (Path("/app/models/tft.ckpt"), Path("/app/models/tft_meta.json")),
    ]
    env_dir = os.environ.get("MODEL_DIR")
    if env_dir:
        candidates.append((Path(env_dir).parent / "tft.ckpt", Path(env_dir).parent / "tft_meta.json"))
        candidates.append((Path(env_dir) / "tft.ckpt", Path(env_dir) / "tft_meta.json"))

    for c, m in candidates:
        if c.is_file() and m.is_file():
            return c, m
    return None, None


def load_model() -> tuple[QuantileTemporalTransformer | None, dict | None]:
    global _cached_model, _cached_meta
    if _cached_model is not None and _cached_meta is not None:
        return _cached_model, _cached_meta

    ckpt, meta_path = _find_checkpoint_files()
    if ckpt is None or meta_path is None:
        return None, None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pred_len = int(meta.get("prediction_days", 90))
        hidden_dim = int(meta.get("hidden_dim", 48))
        n_heads = int(meta.get("n_heads", 4))
        in_features = int(meta.get("in_features", 4))
        model = QuantileTemporalTransformer(
            in_features=in_features, hidden_dim=hidden_dim, n_heads=n_heads, pred_len=pred_len
        )
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        _cached_model = model
        _cached_meta = meta
        return _cached_model, _cached_meta
    except Exception as exc:
        print(f"Error loading TFT checkpoint: {exc}")
        return None, None


def generate_forecast(
    horizon_days: int = 30,
    current_monthly_spend_inr: float = 683200.0,
    accepted_savings_inr_per_month: float = 0.0,
) -> dict:
    """Predict P10/P50/P90 cost horizons strictly without random scaling."""
    model, meta = load_model()
    if model is None or meta is None:
        raise RuntimeError("TFT checkpoint missing or failed to load. Models must be trained first.")

    max_pred = int(model.pred_len)
    if horizon_days not in (30, 35, 40, 60, 90):
        horizon_days = min(max_pred, max(7, int(horizon_days)))
    else:
        horizon_days = min(max_pred, int(horizon_days))

    parquet_path = PROJECT_ROOT / "data" / "history.parquet"
    if not parquet_path.is_file():
        parquet_path = PROJECT_ROOT / "datasets" / "raw" / "telemetry.parquet"

    df = pd.read_parquet(parquet_path)
    ts_col = "ts" if "ts" in df.columns else "timestamp"
    df["day"] = pd.to_datetime(df[ts_col]).dt.floor("1D")

    daily = df.groupby(["day"]).agg({
        "cpu_utilization_pct": ["mean", "max"],
        "memory_utilization_pct": ["mean", "max"],
    }).reset_index()
    daily.columns = ["day", "cpu_mean", "cpu_max", "mem_mean", "mem_max"]
    daily = daily.sort_values("day").reset_index(drop=True)

    series = daily[["cpu_mean", "cpu_max", "mem_mean", "mem_max"]].values.astype(np.float32)
    if len(series) < 14:
        reps = int(np.ceil(14 / len(series)))
        series = np.tile(series, (reps, 1))

    x_enc = torch.tensor(series[-14:], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        p10, p50, p90 = model(x_enc)
        p10_arr = p10[0, :horizon_days].numpy()
        p50_arr = p50[0, :horizon_days].numpy()
        p90_arr = p90[0, :horizon_days].numpy()

    p50_arr = np.maximum(p50_arr, p10_arr)
    p90_arr = np.maximum(p90_arr, p50_arr)

    daily_base = current_monthly_spend_inr / 30.0
    mean_u = float(np.mean(series[-14:, 0])) if np.mean(series[-14:, 0]) > 0 else 20.0

    b_p10 = [round(float(daily_base * (1.0 + (p10_arr[i] - mean_u) / 100.0 * 0.5)), 2) for i in range(horizon_days)]
    b_p50 = [round(float(daily_base * (1.0 + (p50_arr[i] - mean_u) / 100.0 * 0.5)), 2) for i in range(horizon_days)]
    b_p90 = [round(float(daily_base * (1.0 + (p90_arr[i] - mean_u) / 100.0 * 0.5)), 2) for i in range(horizon_days)]

    for i in range(horizon_days):
        if b_p50[i] < b_p10[i]:
            b_p50[i] = b_p10[i]
        if b_p90[i] < b_p50[i]:
            b_p90[i] = b_p50[i]

    daily_savings = max(0.0, accepted_savings_inr_per_month / 30.0)
    opt_p10 = [round(max(0.0, b_p10[i] - daily_savings), 2) for i in range(horizon_days)]
    opt_p50 = [round(max(0.0, b_p50[i] - daily_savings), 2) for i in range(horizon_days)]
    opt_p90 = [round(max(0.0, b_p90[i] - daily_savings), 2) for i in range(horizon_days)]

    start_dt = datetime.now(timezone.utc).date() + timedelta(days=1)
    dates = [(start_dt + timedelta(days=i)).isoformat() for i in range(horizon_days)]

    return {
        "horizon_days": horizon_days,
        "dates": dates,
        "baseline": {"p10": b_p10, "p50": b_p50, "p90": b_p90},
        "optimized": {"p10": opt_p10, "p50": opt_p50, "p90": opt_p90},
        "model": {**meta, "is_trained": True},
    }
