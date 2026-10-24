"""Optional Google Gemini API client for user-facing recommendation summaries."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from app.config import Settings


def generate_description(
    settings: Settings,
    title: str,
    facts: dict[str, str | int | float | bool | None],
) -> str:
    if not settings.google_api_key:
        raise RuntimeError("Google AI is not configured. Set GOOGLE_API_KEY in the environment.")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(settings.google_model, safe='')}:generateContent"
    )
    body = json.dumps(
        {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "Write a concise, factual cloud-operations explanation "
                                "from the supplied JSON facts. Do not invent savings, "
                                "risks, or actions. State uncertainty where values are "
                                f"estimates.\nTitle: {title}\nFacts: "
                                f"{json.dumps(facts, sort_keys=True)}"
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 400},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": settings.google_api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Google AI request failed ({exc.code}): {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Google AI: {exc.reason}") from exc

    try:
        return str(payload["candidates"][0]["content"]["parts"][0]["text"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Google AI returned no text description.") from exc
