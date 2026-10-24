import json

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
