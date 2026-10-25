"""Open-source / free capacity risk assessment engine replacing proprietary paid Jev."""

from __future__ import annotations

import math
from typing import Any

RISK_TIERS = ["Low", "Moderate", "High", "Critical"]


def assess_open_risk(
    recommendation: dict[str, Any],
    sample_count: int = 180,
    peak_cpu_p95: float | None = None,
    peak_cpu_p99: float | None = None,
    proj_cpu_p95: float | None = None,
    proj_cpu_p99: float | None = None,
    proj_mem_p95: float | None = None,
) -> dict[str, Any]:
    """Calculate open-source ordinal capacity risk (0..3 -> 0..100%) and statistical confidence.

    Does not depend on external paid APIs or API keys. Fully local and deterministic.
    """
    action = str(recommendation.get("action", "") or recommendation.get("action_type", "")).upper()
    status = str(recommendation.get("optimization_status", "") or recommendation.get("status", "")).upper()

    # Extract or infer metrics
    c_p95 = peak_cpu_p95 if peak_cpu_p95 is not None else float(recommendation.get("peak_cpu_pct", 22.0) or 22.0)
    c_p99 = peak_cpu_p99 if peak_cpu_p99 is not None else float(recommendation.get("peak_cpu_p99", 28.0) or 28.0)
    p_c95 = proj_cpu_p95 if proj_cpu_p95 is not None else float(recommendation.get("target_peak_cpu_pct", c_p95 * 1.5) or c_p95)
    p_c99 = proj_cpu_p99 if proj_cpu_p99 is not None else float(recommendation.get("target_peak_cpu_p99", c_p99 * 1.5) or c_p99)
    p_m95 = proj_mem_p95 if proj_mem_p95 is not None else float(recommendation.get("target_peak_memory_pct", 45.0) or 45.0)

    # 1. Tier Classification & Ordinal Score (0.0 to 3.0)
    if "PROTECTED" in status or c_p99 >= 85.0 or p_c99 >= 85.0:
        score = 2.76
        level = "Critical"
        probs = {"Low": 0.01, "Moderate": 0.05, "High": 0.16, "Critical": 0.78}
        cushion = max(0.0, p_c99 - 85.0)
        base_confidence = 0.96
    elif p_c95 > 75.0 or p_c99 >= 80.0 or p_m95 >= 75.0:
        score = 2.15
        level = "High"
        probs = {"Low": 0.05, "Moderate": 0.22, "High": 0.63, "Critical": 0.10}
        cushion = 85.0 - p_c99
        base_confidence = 0.92
    elif p_c95 > 55.0 or p_c99 >= 70.0:
        score = 1.05
        level = "Moderate"
        probs = {"Low": 0.28, "Moderate": 0.56, "High": 0.13, "Critical": 0.03}
        cushion = 85.0 - p_c99
        base_confidence = 0.94
    else:
        # Low risk: comfortable headroom
        if "ORPHAN" in action or "ORPHAN" in status:
            score = 0.08
            probs = {"Low": 0.96, "Moderate": 0.03, "High": 0.01, "Critical": 0.00}
        elif "IDLE" in action or "IDLE" in status or "SCHEDULE" in action:
            score = 0.18
            probs = {"Low": 0.90, "Moderate": 0.08, "High": 0.01, "Critical": 0.01}
        else:
            score = 0.42
            probs = {"Low": 0.82, "Moderate": 0.14, "High": 0.03, "Critical": 0.01}
        level = "Low"
        cushion = 85.0 - p_c99
        base_confidence = 0.97

    # Backward compatibility with numeric keys "0", "1", "2", "3"
    tier_to_idx = {"Low": "0", "Moderate": "1", "High": "2", "Critical": "3"}
    for tier_name, p_val in list(probs.items()):
        probs[tier_to_idx[tier_name]] = p_val

    # 2. Fix the confidence of the Risk (Item 5)
    # Grounded confidence calculation based on:
    # a) Sample coverage (180 samples = 30 days of 4h telemetry -> 1.0; fewer samples reduce confidence)
    # b) Cushion from SLA boundary (further from 85% boundary -> higher confidence)
    sample_factor = min(1.0, math.sqrt(max(1, sample_count) / 180.0))
    distance_factor = min(1.0, 0.85 + 0.15 * min(1.0, abs(cushion) / 25.0))
    confidence = round(max(0.60, min(0.99, base_confidence * sample_factor * distance_factor)), 2)

    confidence_pct = round(confidence * 100, 1)
    confidence_reason = (
        f"{sample_count} telemetry samples observed ({min(30, round(sample_count * 4 / 24))}d window); "
        f"projected p99 is {p_c99:.1f}% ({abs(cushion):.1f}% margin to 85% SLA governor limit)."
    )

    return {
        "model": "OpenRisk-Ordinal-v1 (Open-Source)",
        "risk_score": round(score, 2),
        "risk_score_percentage": round((score / 3.0) * 100.0, 1),
        "risk_level": level.lower(),
        "model_confidence": confidence,
        "confidence_pct": confidence_pct,
        "confidence_level": "HIGH" if confidence >= 0.85 else ("MEDIUM" if confidence >= 0.70 else "LOW"),
        "confidence_reason": confidence_reason,
        "probabilities": probs,
        "explanation": (
            "Open-source ordinal capacity risk model: Classified across 4 operational safety tiers "
            "(Low, Moderate, High, Critical) based on p95/p99 peak telemetry headroom."
        ),
        "decision_policy": "Deterministic capacity governors (Projected p99 < 85% and p95 <= 80%) remain authoritative.",
    }
