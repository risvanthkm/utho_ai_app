# FinSort — Python AI, TFT Forecasting & SRE Intelligence (`utho_ai_app`)

> **Jury Overview:** `utho_ai_app` is the AI, machine-learning forecasting, and SRE root-cause intelligence microservice for **FinSort** on **Utho Cloud** (`in-mumbai-1`, `in-noida-1`, `in-bangalore-1`). It provides **Temporal Fusion Transformer (TFT) P10/P50/P90 quantile usage & cost forecasting**, **SRE telemetry audits with mathematical headroom proofs**, **ordinal capacity risk scoring (Jev)**, **natural-language What-If scenario parsing**, **Terraform HCL generation**, and a **retrieval-grounded FinOps Copilot** powered by Google Gemini.

---

## 3-Repository System Architecture

| Repository | Tech Stack | Port | Responsibility |
| :--- | :--- | :--- | :--- |
| **`utho_ai_app`** *(this repo)* | Python 3.11+, FastAPI, PyTorch TFT, Google Gemini | `:8001` | Pooled TFT `P10/P50/P90` forecasting, SRE root-cause audits, Jev ordinal risk, NL What-If parser & grounded chat |
| **`utho_backend`** | Go 1.22+ (`net/http`), `sync.RWMutex` Seed 42 Store | `:8080` | Core FinOps REST API, `p95`/`p99` peak-safety optimizer, What-If engine, Terraform/CLI exporter, AI proxy |
| **`utho_frontend`** | React 19, TypeScript, Vite 8, Tailwind CSS 4 | `:5173` | Executive FinOps Dashboard, Resource Drill-Down, Recommendations, Forecast, What-If & Terraform Modal |

---

## Full-Stack Setup Instructions (All 3 Repos)

### 1. Start Python AI & Forecasting Service (`utho_ai_app` — Port `:8001`)
```bash
cd utho_ai_app
python -m venv .venv
# Windows: .venv\Scripts\activate | macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env   # Optional: set GOOGLE_API_KEY for live Gemini chat synthesis

# Run on port 8001 (expected by utho_backend)
uvicorn app.main:app --host 0.0.0.0 --port 8001
```
- **Interactive Swagger Docs:** `http://localhost:8001/docs`
- **Healthcheck:** `GET http://localhost:8001/healthz` and `GET http://localhost:8001/api/v1/health`

### 2. Start Go FinOps Engine & API (`utho_backend` — Port `:8080`)
```bash
cd utho_backend
go run ./cmd/server
# Binds to http://localhost:8080
```

### 3. Start React FinOps Frontend (`utho_frontend` — Port `:5173`)
```bash
cd utho_frontend
npm install
npm run dev
# Open http://localhost:5173
```

---

## Core AI & ML Capabilities (For Jury Review)

### 1. Pooled Temporal Fusion Transformer (TFT) Forecasting (`app/services/forecasting.py`, `smartcloud_ai.py`)
- Learns pooled multi-horizon representations across VM telemetry with `P10`, `P50`, and `P90` quantile heads (`PinballLoss`).
- Produces **30 / 60 / 90-day P10–P50–P90 spend & utilization fan trajectories**, comparing **Baseline Unoptimized Run-Rate** against the **FinOps Optimized Trajectory** (subtracting accepted recommendation savings) and reporting validation MAE vs. seasonal-naive baseline MAE.

### 2. SRE Telemetry Audit & Peak-Safety Proof (`POST /ai/explain-resource`)
- Evaluates `cpu_avg`, `cpu_p95`, `cpu_p99`, `mem_p95`, and `net_mbps_p95` to generate:
  - **Mathematical Peak Headroom Proof:** Exact projected `p95`/`p99` utilization under candidate smaller plans.
  - **Peak-Safety Trap Detection:** Blocks downsizing when bursty peaks (`p99 >= 85%` or `p95 > 75%`) threaten SLA latency, even if 30-day average CPU is `<15%`.
  - **Ordinal Capacity Risk Score (`JevAssessment`):** 4-tier ordinal risk probability distribution (`0: Minimal` → `3: Critical`) alongside a 3-step Utho CLI rollback runbook and per-resource Terraform HCL snippet.

### 3. Natural-Language What-If Scenario Parser (`POST /ai/parse-what-if`, `POST /api/v1/scenarios/simulate`)
- Translates plain-English prompts (e.g., *"Hibernate all dev and staging VMs on nights and weekends with a 20% headroom buffer and clean up orphans older than 7 days"*) into structured simulation parameters (`dev_off_hours_enabled`, `dev_shutdown_hours_per_week`, `safety_headroom_buffer_pct`, `purge_orphaned_older_than_days`) and computes exact before/after INR cost and carbon deltas.

### 4. Retrieval-Grounded FinOps Copilot (`POST /api/v1/chat`, `POST /ai/chat`)
- Answers operational questions strictly grounded in live inventory, `p95`/`p99` telemetry, active anomaly alerts, and Utho Cloud catalog rates, returning structured source citations (`resource_id`, metric values, and savings math). Operates deterministically even when `GOOGLE_API_KEY` is not set.

---

## Key API Endpoints

| Method & Path | Purpose |
| :--- | :--- |
| `GET /healthz` & `GET /api/v1/health` | Service health & active AI feature capabilities |
| `POST /ai/explain-resource` | Deep SRE audit, headroom math proof, Jev ordinal risk, 24h quantiles & Utho Terraform HCL |
| `POST /ai/forecast` | 30/60/90-day P10/P50/P90 cost & utilization forecast with baseline vs. optimized bands |
| `POST /ai/parse-what-if` | Natural-language FinOps policy parser & impact explanation |
| `POST /ai/anomalies` | Root-cause correlation & time-to-failure (TTF) math for memory leaks, cost spikes & disk trends |
| `POST /ai/chat` & `POST /api/v1/chat` | Grounded FinOps Copilot chat with evidence citations |
| `POST /api/v1/forecast/train` | Train pooled CPU & memory PyTorch TFT checkpoints |
| `POST /api/v1/forecast/usage` | Multi-horizon `q10`/`q50`/`q90` VM utilization inference |
| `POST /api/v1/optimization/actionable` | `p95`/`p99` peak-based rightsizing, idle VM detection, orphan cleanup & off-hours schedules |
| `POST /api/v1/terraform` | Reviewable Terraform HCL configuration generator |

---

## Verification & Test Suite

```bash
pytest
```
Covers `p99`-based rightsizing, scheduled shutdown arithmetic, NL scenario parsing, TFT forecast schema validation, Terraform HCL generation safety, sustainability calculations, Jev risk validation, and API health.
