# Cloud Resource Advisor

FastAPI backend for forecasting VM utilization, estimating resource costs,
recommending VM sizes and shutdown schedules, simulating what-if scenarios,
estimating sustainability impact, generating Terraform, and optionally creating
plain-language summaries with Google AI.

The service does **not** connect to a cloud account, stop instances, or apply
Terraform. Cost, power, and carbon outputs are estimates based on the supplied
inventory, price catalog, telemetry, and documented assumptions.

## Project layout

```text
app/
  api/routes.py             HTTP endpoints and request handling
  config.py                 Environment-based paths and service settings
  data.py                   Dataset loading and schema checks
  main.py                   FastAPI application
  schemas.py                Validated API request and response models
  services/
    descriptions.py         Optional Google Gemini API integration
    forecasting.py          Pooled TFT training and inference
    optimization.py         Plan selection and cost calculations
    scenarios.py            Shutdown what-if simulation
    sustainability.py       Energy/carbon estimates
    terraform.py            Terraform text generation
    dashboard.py            Dashboard metrics, historical percentiles, alerts
    jev.py                  Optional TypeSafe Jev risk assessment
datasets/
  raw/                      Put input inventory, pricing, and telemetry here
  generated/                Optional local/generated data; ignored by Git
models/                     Locally trained checkpoints; ignored by Git
tests/                      Unit and API tests
```

## Setup

Use Python 3.11 or newer. For a CPU-only machine, install the matching CPU
PyTorch wheel before the other dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r requirements.txt
cp .env.example .env
```

Put your own input files in `datasets/raw/` or set `INVENTORY_PATH`,
`PRICING_PATH`, and `TELEMETRY_PATH` in `.env`. No resource IDs, sample
inventory, or generated telemetry are hardcoded in the application. The
inventory and pricing CSVs in `datasets/raw/` are replaceable examples.
Telemetry is not committed because it can be large; provide it at the configured
path or set `TELEMETRY_PATH`. The API accepts CSV inventory/pricing and CSV or
Parquet telemetry.

Start the API:

```bash
uvicorn app.main:app --reload --env-file .env
```

Open interactive API docs at `http://127.0.0.1:8000/docs`; health check is
`GET /api/v1/health`.

## Run with Docker Compose

Install Docker Engine and the Docker Compose plugin. Keep your inventory,
pricing, and telemetry files in `datasets/raw/`; Compose mounts this directory
read-only into the container, so large or private telemetry is not copied into
the image. The trained model is saved in a persistent Docker volume.

```bash
cp .env.example .env
docker compose up --build
```

The API is available at `http://localhost:8000/docs`. To use another host port,
set `API_PORT` in `.env`. Optional Google AI and TypeSafe credentials can also
be configured there. Stop the service with `docker compose down`; model
checkpoints remain in the volume. To remove them as well, run
`docker compose down --volumes`.

To build and run the API image without Compose, first ensure the configured
datasets exist in `datasets/raw/`, then run:

```bash
docker build -t cloud-resource-advisor .
docker run --rm -p 8000:8000 \
  -v "$PWD/datasets/raw:/app/datasets/raw:ro" \
  -v cloud-resource-advisor-models:/app/models \
  cloud-resource-advisor
```

## Dataset contracts

Inventory is one row per resource and must contain:

```text
resource_id,type,plan,region,environment,vcpu,ram_gb,disk_size_gb
```

`team` is optional. `type` values currently supported for costing are `VM`,
`block_volume`, and `public_ip`. VM rows need telemetry to receive model-based
recommendations. Inventory resource IDs must be unique. To allow the API to
identify orphaned storage/IP resources, include an `attached_to`,
`attached_resource_id`, or `attachment_id` column. An empty value is treated as
unattached; verify with your cloud provider before cleanup.

Pricing is one row per plan and must contain:

```text
plan,type,vcpu,ram_gb,rate,storage_rate
```

`rate` is interpreted as USD per hour for VMs and public IPs. `storage_rate` is
interpreted as USD per GB-month for storage. Provide the pricing units and
currency that match your provider; the example calculations assume USD.

Telemetry is one row per resource and timestamp and must contain:

```text
resource_id,timestamp,cpu_utilization_pct,memory_utilization_pct
```

Utilization values are percentages from 0 to 100. Timestamps must parse as
datetimes. The forecast model aggregates input telemetry into configurable
time buckets before training. The dashboard also reads these optional columns
when supplied: `disk_utilization_pct` (0–100%), `disk_iops` (IOPS),
`network_in_mbps` and `network_out_mbps` (Mbps). Metrics missing from the file
are explicitly omitted rather than fabricated.

## Train and forecast

Training learns **one pooled TFT for CPU and one pooled TFT for memory**, using
all inventoried VMs with matching telemetry. Training and inference are
separate: checkpoints are saved under `MODEL_DIR`, and later forecast requests
load them without retraining.

1. Call `POST /api/v1/forecast/train`, for example:

   ```json
   {
     "aggregation_hours": 4,
     "prediction_hours": 24,
     "epochs": 5,
     "batch_size": 64
   }
   ```

2. Call `POST /api/v1/forecast/usage` with `{"hours": 24}`. The requested
   prediction horizon must match the horizon used to train the current model.
   The result contains q10, q50, and q90 CPU/memory forecasts per VM and
   timestamp.

Training uses a time-ordered validation split and writes model checkpoints plus
a small manifest to the configured model directory. It does not write plot
files or training logs.

## API features

| Endpoint | Purpose |
| --- | --- |
| `GET /api/v1/resources` | Read the configured inventory |
| `GET /api/v1/catalog` | Read configured plan prices/capacities |
| `POST /api/v1/forecast/train` | Train pooled CPU and memory TFT models |
| `POST /api/v1/forecast/usage` | Forecast all VMs using saved models |
| `POST /api/v1/optimization/recommendations` | Recommend maintain/downsize/upsize plans and show a demand-based safety margin |
| `POST /api/v1/costs/forecast` | Estimate current and recommended costs over 30, 60, or 90 days, by component and resource |
| `POST /api/v1/scenarios/simulate` | Simulate a recurring shutdown schedule, with optional natural-language scenario text |
| `POST /api/v1/terraform` | Return reviewable Terraform text for one VM plan |
| `POST /api/v1/descriptions` | Generate a factual summary with Google Gemini |
| `GET /api/v1/dashboard/summary` | Dashboard totals and metric rollups, with filters |
| `GET /api/v1/dashboard/resources` | Filterable per-resource utilization and cost estimates |
| `GET /api/v1/resources/{resource_id}/history` | Drill down into a resource's downsampled history |
| `POST /api/v1/optimization/actionable` | Peak-based resize, idle-resource, orphan, and schedule suggestions |
| `POST /api/v1/optimization/jev-risk` | Optional Jev risk assessment for one resource |
| `GET /api/v1/alerts` | Capacity, spike, trend, budget, and supplied cost-change alerts |

### Cost and capacity recommendations

The original `POST /api/v1/optimization/recommendations` uses forecast q90
utilization. The dashboard's `POST /api/v1/optimization/actionable` instead
uses historical p95 or p99 CPU and memory (p99 by default), and a configurable
headroom factor (1.3 by default). It calculates the capacity required at that
peak percentile, then selects the least expensive catalog plan meeting both
CPU and RAM requirements. It reports a specific plan, required capacity,
estimated monthly savings, confidence in the data coverage, and a capacity-risk
percentage. This is intended to avoid resizing based only on averages.

The dashboard endpoints accept exact `region`, `environment`, and `team`
filters. Resource summaries include latest, mean, p95, p99, and maximum values
for every telemetry metric present. `network_in_mbps` and
`network_out_mbps` are throughput, not a percent of network capacity; a
network-capacity percentage requires a provider-specific link limit, which is
not currently in the input schema. The resource history endpoint defaults to
the latest seven days and hourly buckets; choose a 5-, 15-, 60-, 240-, or
1440-minute interval. Up to 90 days can be requested.

The actionable endpoint also flags non-production VMs that are idle across
CPU, memory, disk IOPS, and network, and suggests shutdown windows when hourly
CPU p95 stays under 5%. Volumes and IPs are only labelled orphaned when the
inventory includes an explicit, empty `attached_to`, `attached_resource_id`,
or `attachment_id` field. Without this metadata, the API returns an attachment
review item rather than guessing. All actions are suggestions: the API does
not stop, resize, or delete cloud resources.

`risk_percentage` is an explainable **capacity-pressure index**: the maximum of
the target plan's projected peak CPU/memory utilization and the existing disk
utilization percentile. It is not the probability that a VM will fail.
`confidence_level` is based on the amount and duration of telemetry available:
at least seven days is high, at least one day is medium, and less than one day
or missing data is low. These confidence labels describe evidence coverage,
not a guarantee the recommendation is correct.

Example dashboard calls:

```bash
# Summary across the last 30 days; filter by any combination of dimensions.
curl 'http://127.0.0.1:8000/api/v1/dashboard/summary?hours=720&region=mumbai&environment=prod'

# Per-resource utilization rows; add team=Payments to filter by owner.
curl 'http://127.0.0.1:8000/api/v1/dashboard/resources?hours=720&team=Payments'

# Drill into one VM; points are averaged into hourly buckets by default.
curl 'http://127.0.0.1:8000/api/v1/resources/res-00001/history?hours=168&interval_minutes=60'

# Use p99 peaks and 30% headroom for each resize recommendation.
curl -X POST 'http://127.0.0.1:8000/api/v1/optimization/actionable' \
  -H 'Content-Type: application/json' \
  -d '{"hours":720,"peak_percentile":99,"headroom":1.3,"environment":"prod"}'

# Compare current-rate monthly cost with budget and a caller-supplied last-month bill.
curl 'http://127.0.0.1:8000/api/v1/alerts?hours=720&monthly_budget_usd=5000&previous_month_cost_usd=4200'
```

### Dashboard alerts

`GET /api/v1/alerts` reports:

- Capacity alerts when the selected-window p99 CPU, memory, or disk utilization
  reaches 85% (high severity at 95%).
- Usage spikes when the latest hourly p95 is at least 15 percentage points
  above the prior hourly baseline and exceeds it by three standard deviations;
  network spikes use a 10 Mbps minimum increase.
- Capacity trends when daily p95 CPU or memory is rising by at least 0.5
  percentage points per day and is already at 70% or higher.
- Budget overruns when `monthly_budget_usd` is below the current-rate monthly
  estimate.
- Cost increases when `previous_month_cost_usd` is provided and the catalog-rate
  estimate is at least 20% above it.

The sample inventory has no billing-history dataset. Therefore the cost-change
alert compares the catalog-rate projection against a caller-supplied previous
month amount; it does not claim to be a provider's billed-cost history. Every
alert includes a reason and supporting values.

### Optional Jev risk assessment

Set `TYPESAFE_API_KEY` in `.env` to enable the
`POST /api/v1/optimization/jev-risk` endpoint. For one resource, it sends the
recommendation and derived capacity metrics to TypeSafe's System One API and
returns Jev's ordinal risk score, level probabilities, and model confidence.
The service does not send the resource ID or raw telemetry. The
`risk_score_percentage` is the Jev score normalized to 0–100, **not an outage
probability**; Jev confidence is confidence in its classification, not the
chance a recommendation is safe. Jev is advisory and never changes or executes
the deterministic recommendation. Without an API key, the endpoint returns an
explicit service-unavailable response.
The model is configured with `TYPESAFE_MODEL` (default `jev-latest`). This
integration calls TypeSafe's documented `POST https://api.typesafe.ai/v1/systemone`
interface using a typed Score question; it is deliberately separate from the
fast, deterministic dashboard and recommendation endpoints.

The cost endpoint projects the selected plan and current rates across the
requested 30/60/90-day period. It uses the horizon saved with the trained usage
model to select recommended plans, so it works with any supported training
horizon. It is a **constant-rate run-rate projection**, not an ML prediction of
future prices or utilization for all those days. It includes VM compute, public
IP, and storage charges available in the provided catalog. The full
resource-level calculation and units are returned in JSON.

Non-production VMs can also include a suggested low-load window when consecutive
forecast intervals have q90 CPU below 5%. This is a suggestion only; apply a
shutdown window through the scenario simulator to see its assumed savings.

### What-if shutdown simulation

Use structured fields for exact, repeatable inputs:

```json
{
  "days": 30,
  "environment": "dev",
  "shutdown_start": "20:00:00",
  "shutdown_end": "08:00:00",
  "weekdays_only": false,
  "use_recommended_plans": true
}
```

Or use the supported simple text form:

```json
{
  "days": 30,
  "scenario_text": "What if we shut down all dev VMs from 8 PM to 8 AM?"
}
```

The parser intentionally accepts a limited, explicit format (environment plus
start/end time); it is not an unconstrained natural-language planner. Overnight
windows are supported. The simulation assumes VM compute billing stops during
the window; storage and public IP costs continue. Weekday-only schedules count
five shutdown days per seven-day week. No actual cloud actions are taken.

### Terraform generation

`POST /api/v1/terraform` needs `resource_id`, `plan`, `ami_id`, and `subnet_id`.
The generated HCL creates a **new** AWS EC2 instance using the catalog plan.
Review AMI compatibility, networking, security groups, storage, IAM, import or
migration strategy, and Terraform state before applying. The service never runs
Terraform.

### Google AI descriptions

Set `GOOGLE_API_KEY` (and optionally `GOOGLE_MODEL`) in the process environment
or `.env` to enable the descriptions endpoint. The key is not checked into the
repository. The endpoint sends the requested title and facts to Google AI;
avoid sending confidential inventory or customer data. Without a configured
key, it returns an explicit service-unavailable response.

### Sustainability estimate

Optimization and scenario responses estimate energy and operational carbon
changes from vCPU capacity, forecast CPU utilization, power-per-vCPU, data-center
PUE, grid carbon intensity, and (for scenarios) selected VM shutdown hours.
Defaults are configurable with `WATTS_PER_VCPU_IDLE`,
`WATTS_PER_VCPU_FULL`, `DATA_CENTER_PUE`, and `GRID_CARBON_KG_PER_KWH`.
The sustainability score is the estimated percentage reduction in operational
carbon versus the current-plan estimate, clamped to 0–100. These are transparent
planning estimates, not measured energy or a lifecycle assessment.

## Configuration

See `.env.example` for all paths and optional integration/energy settings.
Relative dataset and model paths are resolved from this project directory.
The API validates required columns and fails with an actionable response when
configured files or model checkpoints are missing.

## Tests

```bash
pytest
```

Tests cover plan selection, per-component costs, scheduled shutdown arithmetic,
scenario parsing, Terraform input safety, sustainability output, dashboard
filtering and history, p99-based rightsizing, idle/orphan suggestions, alert
reasons, Jev response validation, and API health.
