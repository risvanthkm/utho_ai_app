"""Production RAG and Gemini LLM integration service."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAG_CACHE_PATH = PROJECT_ROOT / "data" / "rag_index.json"

_rag_chunks: list[dict[str, Any]] = []
_rag_embeddings: np.ndarray | None = None


class ScenarioStructuredOutput(BaseModel):
    scenario_label: str = Field(description="Brief title of the simulated scenario")
    target_environment: str | None = Field(default=None, description="Target environment: production, staging, or development")
    target_region: str | None = Field(default=None, description="Target Utho region: in-noida-1, in-bangalore-1, or in-mumbai-1")
    dev_off_hours_enabled: bool = Field(default=False, description="Whether dev/staging off-hours schedule is enabled")
    dev_shutdown_hours_per_week: int = Field(default=0, ge=0, le=168, description="Hours per week workloads are powered off")
    staging_weekend_off_enabled: bool = Field(default=False, description="Whether staging weekend shutdown is enabled")
    safety_headroom_buffer_pct: int = Field(default=30, ge=10, le=50, description="Safety headroom buffer percentage")
    purge_orphaned_disks: bool = Field(default=False, description="Whether to purge unattached block storage volumes")
    release_unused_ips: bool = Field(default=False, description="Whether to release unassociated public IPs")
    explanation: str = Field(description="Engineering rationale for the parameter choices")


def _get_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    from google import genai
    return genai.Client(api_key=api_key)


def build_knowledge_chunks() -> list[dict[str, Any]]:
    chunks = []
    # 1. FinOps Principles & Core Formulas
    chunks.append({
        "id": "finops-rule-001",
        "title": "Peak-Safety Mathematical Governor",
        "text": "FinSort enforces a strict Peak-Safety Governor: any virtual machine with p95 > 70% or p99 > 85% CPU/memory utilization is automatically excluded from downsizing to protect enterprise SLAs against latency spikes.",
    })
    chunks.append({
        "id": "finops-rule-002",
        "title": "Off-Hours Schedules (Dev/Staging)",
        "text": "Non-production workloads (dev and staging) in Noida, Bangalore, and Mumbai are idle during weeknights (20:00 to 08:00 IST) and 48 hours on weekends. Automating shutdown for 118 hours/week reduces monthly run-rate for those workloads by 70.2% without production impact.",
    })
    chunks.append({
        "id": "finops-rule-003",
        "title": "Orphaned Disks and Unattached Elastic IPs",
        "text": "Detached NVMe/SSD block volumes accrue idle tariffs at ₹18/GB/month. Unassociated static public IPs accrue idle reservation charges at ₹1,350/month each. Immediate purging carries zero SLA risk and recovers 100% of wasted spend.",
    })
    chunks.append({
        "id": "pricing-catalog-001",
        "title": "Utho Cloud INR Tariff Catalog",
        "text": "Compute plans: c4.small (1 vCPU, 2 GB) @ ₹2,600/mo (₹3.56/hr), c4.medium (2 vCPU, 4 GB) @ ₹5,000/mo (₹6.85/hr), c4.large (4 vCPU, 8 GB) @ ₹9,200/mo (₹12.60/hr), c4.xlarge (8 vCPU, 16 GB) @ ₹17,500/mo (₹23.97/hr), m4.large (4 vCPU, 16 GB) @ ₹12,800/mo (₹17.53/hr), r4.large (4 vCPU, 32 GB) @ ₹19,400/mo (₹26.58/hr). Block storage: ₹18/GB/mo. Public IP: ₹1,350/mo.",
    })

    # 2. Inventory Chunks
    inv_path = PROJECT_ROOT / "datasets" / "raw" / "inventory.csv"
    if inv_path.is_file():
        df_inv = pd.read_csv(inv_path)
        for _, r in df_inv.iterrows():
            chunks.append({
                "id": str(r.get("resource_id")),
                "title": f"Resource {r.get('resource_id')} ({r.get('name', 'node')})",
                "text": f"Resource ID: {r.get('resource_id')}, Name: {r.get('name')}, Type: {r.get('type')}, Region: {r.get('region')}, Environment: {r.get('environment')}, Team: {r.get('team')}, Plan: {r.get('plan')}, vCPU: {r.get('vcpu')}, RAM: {r.get('ram_gb')}GB, Disk: {r.get('disk_size_gb')}GB, Cost: ₹{r.get('monthly_cost_inr', 0)}/mo.",
            })

    return chunks


def init_rag_index():
    """Build or load cached RAG embeddings index at startup."""
    global _rag_chunks, _rag_embeddings
    chunks = build_knowledge_chunks()
    _rag_chunks = chunks

    # Check cache
    if RAG_CACHE_PATH.is_file():
        try:
            cached = json.loads(RAG_CACHE_PATH.read_text(encoding="utf-8"))
            if len(cached.get("chunks", [])) == len(chunks):
                _rag_embeddings = np.array(cached["embeddings"], dtype=np.float32)
                return len(_rag_chunks)
        except Exception:
            pass

    client = _get_gemini_client()
    if client is None:
        # Fallback to local TF-IDF / bag-of-words normalized vectors if no Gemini key
        _build_fallback_embeddings()
        return len(_rag_chunks)

    try:
        texts = [f"{c['title']}\n{c['text']}" for c in _rag_chunks]
        # Batch embed using Gemini
        embed_response = client.models.embed_content(
            model="text-embedding-004",
            contents=texts,
        )
        embeddings = [e.values for e in embed_response.embeddings]
        _rag_embeddings = np.array(embeddings, dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(_rag_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _rag_embeddings = _rag_embeddings / norms

        # Save cache
        RAG_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAG_CACHE_PATH.write_text(
            json.dumps({"chunks": _rag_chunks, "embeddings": _rag_embeddings.tolist()}),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"Gemini embedding API call failed: {exc}. Using lexical vector fallback.")
        _build_fallback_embeddings()

    return len(_rag_chunks)


def _build_fallback_embeddings():
    global _rag_embeddings
    vocab: dict[str, int] = {}
    for c in _rag_chunks:
        for word in (c["title"] + " " + c["text"]).lower().split():
            clean = "".join(ch for ch in word if ch.isalnum())
            if clean and clean not in vocab:
                vocab[clean] = len(vocab)
    
    vecs = np.zeros((len(_rag_chunks), max(1, len(vocab))), dtype=np.float32)
    for i, c in enumerate(_rag_chunks):
        for word in (c["title"] + " " + c["text"]).lower().split():
            clean = "".join(ch for ch in word if ch.isalnum())
            if clean in vocab:
                vecs[i, vocab[clean]] += 1.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    _rag_embeddings = vecs / norms


def retrieve(query: str, top_k: int = 6) -> list[dict[str, Any]]:
    global _rag_chunks, _rag_embeddings
    if not _rag_chunks:
        init_rag_index()
    if _rag_embeddings is None:
        return _rag_chunks[:top_k]

    client = _get_gemini_client()
    if client is not None:
        try:
            res = client.models.embed_content(model="text-embedding-004", contents=query)
            q_vec = np.array(res.embeddings[0].values, dtype=np.float32)
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm
            scores = np.dot(_rag_embeddings, q_vec)
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [_rag_chunks[i] for i in top_indices]
        except Exception:
            pass

    # Simple lexical similarity match
    q_words = set("".join(ch for ch in w if ch.isalnum()) for w in query.lower().split() if w)
    scores = []
    for c in _rag_chunks:
        c_words = set("".join(ch for ch in w if ch.isalnum()) for w in (c["title"] + " " + c["text"]).lower().split())
        overlap = len(q_words & c_words)
        scores.append(overlap)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [_rag_chunks[i] for i in top_indices]


def answer_chat(message: str) -> dict[str, Any]:
    """Answer user chat strictly from retrieved FinSort context using gemini-2.5-flash."""
    client = _get_gemini_client()
    if client is None:
        raise RuntimeError("LLM not configured: GOOGLE_API_KEY is missing or invalid.")

    retrieved = retrieve(message, top_k=6)
    sources = [{"id": r["id"], "title": r["title"]} for r in retrieved]

    context_str = "\n\n".join(f"[{r['id']}] {r['title']}:\n{r['text']}" for r in retrieved)

    system_prompt = (
        "You are FinSort Copilot, an expert cloud infrastructure and FinOps advisor for a mid-sized Indian SaaS company running across Utho Cloud data centers in Noida, Bangalore, and Mumbai.\n"
        "HARD RULES:\n"
        "1. Answer ONLY using the facts present in the provided retrieved context below.\n"
        "2. If the context does not contain enough information to answer the user's question, say exactly: 'I don't have data on that.'\n"
        "3. Always cite the relevant source ids (e.g. [res-vm-001], [finops-rule-001], [pricing-catalog-001]) in your answer.\n"
        "4. Never invent, hallucinate, or assume metrics, numbers, or infrastructure facts outside the retrieved context.\n"
    )

    user_prompt = f"Retrieved Context:\n{context_str}\n\nUser Question:\n{message}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config={"system_instruction": system_prompt},
        )
        return {
            "answer": response.text.strip(),
            "sources": sources,
        }
    except Exception as exc:
        raise RuntimeError(f"Gemini API generation failed: {exc}") from exc


def parse_scenario_structured(query: str) -> dict[str, Any]:
    """Convert natural-language scenario into validated parameters via Gemini structured output."""
    client = _get_gemini_client()
    if client is None:
        q_lower = query.lower()
        env = "development" if "dev" in q_lower else "staging" if "staging" in q_lower else "production" if "prod" in q_lower else None
        region = "in-bangalore-1" if "bangalore" in q_lower else "in-noida-1" if "noida" in q_lower else "in-mumbai-1" if "mumbai" in q_lower else None
        return {
            "scenario_label": query.strip()[:60],
            "target_environment": env,
            "target_region": region,
            "dev_off_hours_enabled": "shut down" in q_lower or "off" in q_lower or "night" in q_lower,
            "dev_shutdown_hours_per_week": 84 if "8 pm" in q_lower or "8pm" in q_lower else 60,
            "staging_weekend_off_enabled": "weekend" in q_lower,
            "safety_headroom_buffer_pct": 30,
            "purge_orphaned_disks": "disk" in q_lower or "storage" in q_lower or "orphan" in q_lower,
            "release_unused_ips": "ip" in q_lower,
            "explanation": f"Grounded scenario parsed from natural query: {query.strip()}",
        }

    system_prompt = (
        "You are a FinOps systems engineer for FinSort on Utho Cloud. Convert the user's natural language scenario query "
        "into strict structured configuration parameters adhering to the schema. The Go engine will calculate all costs."
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"User scenario query: {query}",
            config={
                "system_instruction": system_prompt,
                "response_mime_type": "application/json",
                "response_schema": ScenarioStructuredOutput,
            },
        )
        data = json.loads(response.text)
        return data
    except Exception as exc:
        raise RuntimeError(f"Gemini structured output scenario parsing failed: {exc}") from exc
