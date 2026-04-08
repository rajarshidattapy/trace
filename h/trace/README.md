# TRACE — OpenEnv DevOps Response Environment

**TRACE** (Triage, Response, Action, Cause, Evaluation) is a deterministic, partial-observability RL environment for **incident response in production infrastructure**. Built for the [Meta × PyTorch × Hugging Face OpenEnv Hackathon](https://huggingface.co/).

## Overview

An AI agent interacts with realistic infrastructure incidents by observing system metrics, running diagnostic actions, and executing remediation. The environment features:

- **OpenEnv-compliant** — `pyproject.toml`, `server/app.py`, `openenv.yaml`
- **Deterministic** — 3 hand-crafted scenarios, reproducible by seed
- **Partially observable** — ground truth hidden behind `inspect_*` actions
- **Action-structured** — all actions use `(action_type, target, value)` format
- **Outcome-graded** — resolution success + efficiency (no diagnosis_accuracy)

## Scenarios

| Scenario | Difficulty | Root Cause | Typical Fix | Max Steps |
|----------|-----------|------------|-------------|-----------|
| `easy_cpu_spike` | Easy | Worker overload | `scale_workers` | 5 |
| `medium_cascade` | Medium | Queue deadlock/leak | `restart_service` | 7 |
| `hard_mixed` | Hard | DB + release regression | `restart_database` | 8 |

## Quick Start

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Run the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 3. Run inference

```bash
# Set environment variables
export HF_TOKEN=your-token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

python inference.py
```

### 4. Run benchmark (heuristic agent)

```bash
python scripts/run_benchmark.py
```

### 5. Run tests

```bash
python -m pytest tests/ -v
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Reset environment for new episode |
| `POST` | `/step` | Execute one action |
| `GET` | `/state` | Get current state |
| `GET` | `/health` | Health check |

### POST /reset

```json
{
    "task_id": "easy_cpu_spike",
    "seed": 42
}
```

### POST /step

```json
{
    "action": {
        "action_type": "scale_workers",
        "target": "api_workers",
        "value": 5
    }
}
```

## Action Space

| Action | Target | Value | Effect |
|--------|--------|-------|--------|
| `inspect_logs` | service_name | — | Returns log snippet |
| `inspect_metrics` | metric_name | — | Returns metric data |
| `inspect_alert` | alert_id | — | Returns alert details |
| `restart_service` | service_name | — | Resets service state |
| `scale_workers` | service_name | count | Scales horizontally |
| `restart_database` | — | — | Resets DB state |
| `rollback_release` | — | — | Reverts deployment |
| `clear_queue` | — | — | Clears message backlog |
| `declare_healthy` | — | — | End episode (resolved) |
| `declare_unfixable` | — | — | End episode (give up) |

## Scoring

```
score = 0.6 × resolution_success + 0.4 × efficiency
```

- **resolution_success**: 1.0 if incident resolved, 0.0 otherwise
- **efficiency**: `1.0 - (steps_taken / max_steps)`

## Docker

```bash
docker build -t trace-env .
docker run -p 7860:7860 trace-env
```

## Project Structure

```
TRACE/
├── pyproject.toml          # OpenEnv spec
├── openenv.yaml            # Environment metadata
├── Dockerfile
├── requirements.txt
├── inference.py            # Agent policy loop
├── trace/                  # Core module
│   ├── env.py              # TraceEnv class
│   ├── models.py           # Pydantic schemas
│   ├── scenarios.py        # Scenario generators
│   ├── simulator.py        # State transitions
│   ├── rewards.py          # Reward engine
│   ├── graders.py          # Grading logic
│   └── utils.py            # Helpers
├── server/                 # FastAPI app
│   └── app.py              # Routes + server
├── tests/                  # Unit + API tests
│   ├── test_env.py
│   ├── test_api.py
│   ├── test_rewards.py
│   ├── test_graders.py
│   └── test_scenarios.py
└── scripts/
    └── run_benchmark.py    # Local evaluation
```

## Validation

```bash
openenv validate
./validate-submission.sh <your-hf-space-url>
```

## License

MIT

## Author

Rajarshi Datta
