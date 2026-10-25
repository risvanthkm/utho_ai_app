import json
from dataclasses import replace

import pytest

from app.config import settings
from app.services import chatbot


def test_chatbot_router_accepts_only_registered_tools(monkeypatch):
    monkeypatch.setattr(
        chatbot,
        "_gemini_request",
        lambda *_args, **_kwargs: json.dumps(
            {
                "tool": "resource_history",
                "arguments": {"resource_id": "vm-1", "hours": 24},
                "needs_clarification": False,
                "clarification_question": None,
            }
        ),
    )

    routed = chatbot._route_query(settings, "Show vm-1 for 24 hours", [])

    assert routed["tool"] == "resource_history"
    assert routed["arguments"]["resource_id"] == "vm-1"


def test_chatbot_router_rejects_unapproved_tool(monkeypatch):
    monkeypatch.setattr(
        chatbot,
        "_gemini_request",
        lambda *_args, **_kwargs: '{"tool":"sql","arguments":{}}',
    )

    with pytest.raises(RuntimeError, match="unsupported action"):
        chatbot._route_query(settings, "run a database query", [])


def test_chatbot_history_automatically_limits_chart_points(monkeypatch):
    seen = {}

    def fake_history(_settings, resource_id, hours, interval):
        seen.update(resource_id=resource_id, hours=hours, interval=interval)
        return {"points": []}

    monkeypatch.setattr(chatbot, "resource_history", fake_history)

    result = chatbot._bounded_resource_history(settings, "vm-1", 24 * 90, None)

    assert seen == {"resource_id": "vm-1", "hours": 24 * 90, "interval": 240}
    assert result["chart_point_limit"] == chatbot.MAX_CHART_POINTS


def test_chat_answers_from_retrieved_context_without_llm_routing(monkeypatch):
    context = {
        "resource_count": 1,
        "historical_peak_recommendations": {
            "estimated_monthly_savings_total": 125.0,
            "currency": "USD",
        },
        "forecast": {"status": "available", "horizon_hours": 24},
    }
    captured = {}

    monkeypatch.setattr(chatbot, "_retrieve_chat_context", lambda _settings: context)
    monkeypatch.setattr(
        chatbot,
        "_route_query",
        lambda *_args, **_kwargs: pytest.fail("chat must not ask the LLM to route"),
    )

    def fake_gemini(_settings, prompt, **_kwargs):
        captured["prompt"] = prompt
        return "The retrieved estimate is $125.00 USD per month."

    monkeypatch.setattr(chatbot, "_gemini_request", fake_gemini)
    result = chatbot.chat(
        replace(settings, google_api_key="test-key", database_url=None),
        "How much can we save?",
    )

    assert result["answer"] == "The retrieved estimate is $125.00 USD per month."
    assert result["tool"] == "structured_rag_retrieval"
    assert result["data"] == context
    assert "do not classify it, select a tool" in captured["prompt"]
    assert json.dumps(context) in captured["prompt"]
    assert result["model"] == settings.google_model


def test_chat_without_gemini_uses_retrieved_values_not_static_facts(monkeypatch):
    context = {
        "resource_count": 3,
        "30_day_current_cost_estimate": {
            "current_cost": {"total": 42.5},
        },
        "forecast": {"status": "unavailable", "reason": "TFT models are not trained yet."},
    }
    monkeypatch.setattr(chatbot, "_retrieve_chat_context", lambda _settings: context)
    result = chatbot.chat(replace(settings, google_api_key=None, database_url=None), "Summarize")

    assert "3 resources" in result["answer"]
    assert "$42.50 USD" in result["answer"]
    assert "TFT models are not trained yet." in result["answer"]
    assert "85 resources" not in result["answer"]
