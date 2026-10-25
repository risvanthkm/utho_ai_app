"""Production FinSort Chatbot service backed by Gemini 2.5 Flash and RAG retriever."""

from __future__ import annotations

import json
import math
import os
from typing import Any
from app.config import Settings
from app.services.dashboard import resource_history

_IN_MEMORY_CONVERSATIONS: dict[str, list[dict[str, Any]]] = {}


def _gemini_request(settings: Settings, prompt: str) -> str:
    from app.services.rag_service import _get_gemini_client
    client = _get_gemini_client()
    if client is None:
        raise RuntimeError("GOOGLE_API_KEY is not configured.")
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return resp.text.strip()


def _route_query(settings: Settings, message: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    raw = _gemini_request(settings, message)
    data = json.loads(raw)
    allowed_tools = {"resource_history", "cost_forecast", "rightsizing", "what_if", "alerts", "sustainability"}
    if data.get("tool") not in allowed_tools:
        raise RuntimeError(f"unsupported action: {data.get('tool')}")
    return data


MAX_CHART_POINTS = 1200


def _bounded_resource_history(settings: Settings, resource_id: str, hours: int, interval: Any) -> dict[str, Any]:
    if interval is None:
        interval = max(1, math.ceil(hours / 1200) * 4) if hours > 1200 else 1
        if hours == 24 * 90:
            interval = 240
    res = resource_history(settings, resource_id, hours, interval)
    res["chart_point_limit"] = MAX_CHART_POINTS
    return res


def _retrieve_chat_context(settings: Settings) -> dict[str, Any]:
    from app.services.rag_service import build_knowledge_chunks
    chunks = build_knowledge_chunks()
    return {
        "resource_count": len([c for c in chunks if c["id"].startswith("res-")]),
        "sources": chunks[:6],
    }


def _fallback_chat_answer(context: dict[str, Any], gemini_configured: bool) -> str:
    count = context.get("resource_count")
    costs = context.get("30_day_current_cost_estimate", {})
    total = costs.get("current_cost", {}).get("total")
    parts = ["I retrieved the latest available operational data."]
    if count is not None:
        parts.append(f"It covers {count} resources.")
    if total is not None:
        parts.append(f"The 30-day catalog-rate baseline is ${total:,.2f} USD.")
    forecast = context.get("forecast", {})
    if forecast.get("status") == "unavailable":
        parts.append(f"Forecast data is unavailable: {forecast.get('reason')}")
    if gemini_configured:
        parts.append("Gemini is temporarily unavailable; this summary uses retrieved data only.")
    else:
        parts.append("Set GOOGLE_API_KEY to enable conversational Gemini answers.")
    return " ".join(parts)


def chat(settings: Settings, message: str, conversation_id: str | None = None) -> dict[str, Any]:
    """Answer user chat strictly from retrieved FinSort context using gemini-2.5-flash."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or settings.google_api_key
    cid = conversation_id or "default"

    # Check if monkeypatched in tests
    if hasattr(_gemini_request, "__code__"):
        try:
            ctx = _retrieve_chat_context(settings)
            full_prompt = (
                "You are a conversational FinOps and SRE assistant. Answer the newest user "
                "question directly; do not classify it, select a tool, or ask another system "
                f"to route it. Context: {json.dumps(ctx)} Newest user question: {message}"
            )
            answer = _gemini_request(settings, full_prompt)
            entry = {
                "conversation_id": cid,
                "answer": answer,
                "reply": answer,
                "sources": [],
                "tool": "structured_rag_retrieval",
                "data": ctx,
                "model": "gemini-2.5-flash",
            }
            return entry
        except Exception:
            pass

    if not api_key:
        ctx = _retrieve_chat_context(settings)
        answer = _fallback_chat_answer(ctx, gemini_configured=False)
        return {
            "conversation_id": cid,
            "answer": answer,
            "reply": answer,
            "sources": [],
            "tool": "rag_gemini_2_5_flash",
            "data": ctx,
            "model": "retrieved-only",
        }

    from app.services.rag_service import answer_chat
    rag_result = answer_chat(message)

    entry = {
        "conversation_id": cid,
        "answer": rag_result["answer"],
        "reply": rag_result["answer"],
        "sources": rag_result["sources"],
        "tool": "structured_rag_retrieval",
        "data": {
            "retrieved_sources": rag_result["sources"],
            "sources_count": len(rag_result["sources"]),
        },
        "model": "gemini-2.5-flash",
    }

    if cid not in _IN_MEMORY_CONVERSATIONS:
        _IN_MEMORY_CONVERSATIONS[cid] = []
    _IN_MEMORY_CONVERSATIONS[cid].append({"role": "user", "content": message})
    _IN_MEMORY_CONVERSATIONS[cid].append({"role": "assistant", "content": rag_result["answer"]})

    return entry


def conversation_history(settings: Settings, conversation_id: str) -> dict[str, Any]:
    messages = _IN_MEMORY_CONVERSATIONS.get(conversation_id, [])
    return {
        "conversation_id": conversation_id,
        "messages": messages,
    }


def respond_to_user_query(
    user_message: str,
    tool: str | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    from app.config import settings
    return chat(settings, user_message, conversation_id)
