"""Optional TypeSafe Jev assessment for one rightsizing recommendation."""

from __future__ import annotations

import math
from typing import Any

import httpx

from app.config import Settings

API_URL = "https://api.typesafe.ai/v1/systemone"
RISK_LEVELS = [
    "Low risk: recommendation has comfortable measured headroom and adequate history.",
    "Moderate risk: recommendation has limited headroom or incomplete evidence; review it.",
    "High risk: recommendation approaches measured capacity limits or lacks safe evidence.",
    "Critical risk: recommendation would exceed capacity or could cause a severe outage.",
]


def assess_recommendation(settings: Settings, recommendation: dict[str, Any]) -> dict[str, Any]:
    """Ask Jev to classify recommendation risk; never execute or override actions."""
    if not settings.typesafe_api_key:
        raise RuntimeError("Jev risk assessment is not configured. Set TYPESAFE_API_KEY.")
    safe_state = {
        key: recommendation.get(key)
        for key in (
            "action",
            "current_plan",
            "recommended_plan",
            "target_configuration",
            "peak_percentile",
            "peak_cpu_pct",
            "peak_memory_pct",
            "peak_disk_pct",
            "target_peak_cpu_pct",
            "target_peak_memory_pct",
            "risk_percentage",
            "risk_level",
            "confidence_level",
            "confidence_reason",
            "reason",
        )
    }
    payload = {
        "model": settings.typesafe_model,
        "state": safe_state,
        "questions": {
            "recommendation_risk": {
                "type": "score",
                "instructions": (
                    "Assess the operational risk of applying this proposed VM "
                    "rightsizing recommendation, considering the measured peak "
                    "utilization and evidence quality. Do not assume the change "
                    "has been validated or executed."
                ),
                "criteria": RISK_LEVELS,
            }
        },
    }
    try:
        response = httpx.post(
            API_URL,
            headers={"Authorization": f"Bearer {settings.typesafe_api_key}"},
            json=payload,
            timeout=20.0,
        )
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise RuntimeError("Jev risk assessment timed out.") from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Jev risk assessment failed with HTTP {exc.response.status_code}."
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError("Could not reach the Jev risk assessment API.") from exc

    try:
        body = response.json()
        answer = body["answers"]["recommendation_risk"]
        score = float(answer["score"])
        confidence = float(answer["confidence"])
        probabilities = {
            str(level): float(probability) for level, probability in answer["probabilities"].items()
        }
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError("Jev returned an invalid risk assessment response.") from exc
    if (
        not math.isfinite(score)
        or not 0 <= score <= len(RISK_LEVELS) - 1
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
        or not probabilities
        or any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values())
        or set(probabilities) != {"0", "1", "2", "3"}
        or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=0.01)
    ):
        raise RuntimeError("Jev returned out-of-range values for risk assessment.")
    return {
        "model": body.get("model", settings.typesafe_model),
        "risk_score": score,
        "risk_score_percentage": round(score / (len(RISK_LEVELS) - 1) * 100, 2),
        "risk_level": "low"
        if score < 0.75
        else "moderate"
        if score < 1.75
        else "high"
        if score < 2.75
        else "critical",
        "model_confidence": confidence,
        "probabilities": probabilities,
        "explanation": (
            "The percentage is the Jev ordinal score normalized to 0-100, not a "
            "measured probability of outage. Jev confidence describes certainty "
            "in its assessment, not the chance the recommendation is safe."
        ),
        "decision_policy": "Advisory only; deterministic capacity checks remain authoritative.",
    }
