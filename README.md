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

## Dataset contracts

Inventory is one row per resource and must contain:

```text
resource_id,type,plan,region,environment,vcpu,ram_gb,disk_size_gb
```

`team` is optional. `type` values currently supported for costing are `VM`,
`block_volume`, and `public_ip`. VM rows need telemetry to receive model-based
recommendations. Inventory resource IDs must be unique.

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
time buckets before training.

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

### Cost and capacity recommendations

Recommendations use the peak forecast q90 CPU/memory utilization and a
configurable capacity headroom factor. The least expensive catalog plan meeting
both estimated demands is selected. If it costs more than the current plan, the
action is `UPSCALE`; if cheaper, `DOWNSIZE`; otherwise `MAINTAIN`. Missing
forecasts leave a VM at its existing plan and are explicitly marked.

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
scenario parsing, Terraform input safety, sustainability output, and API health.
