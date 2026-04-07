# TRACE

**Build-ready production v1 for Meta × PyTorch × Hugging Face OpenEnv Hackathon**

TRACE is a **deterministic, partially-observable RL environment** for incident response in production infrastructure. An AI agent learns to:

1. **Observe** basic telemetry (CPU, memory, latency, error_rate, queue_depth)
2. **Inspect** hidden details (logs, metrics, alert context)
3. **Diagnose** root causes (implicitly through remediation success)
4. **Remediate** incidents (restart services, scale workers, restart databases)

## Key Design Decisions (v1)

✅ **Partial Observability** — Ground truth hidden behind `inspect_*` actions  
✅ **Deterministic Scenarios** — Reproducible by seed; 3 hand-crafted incidents  
✅ **Action Structure** — All actions use `(action_type, target, value)` format  
✅ **Cumulative Rewards** — No per-step clamping; normalized at episode end  
✅ **Outcome-Based Grading** — Score = 0.6×resolution + 0.4×efficiency (no diagnosis_accuracy)  
✅ **OpenEnv Compliant** — `pyproject.toml`, `server/app.py`, `openenv.yaml`  

## Quick Start

### Prerequisites

```bash
python -m venv venv
./venv/Scripts/activate  # Windows
pip install -e .
```

### Run Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run Inference

```bash
python inference.py
```

### Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
TRACE/
├── pyproject.toml           # Project definition
├── requirements.txt         # Dependencies
├── openenv.yaml             # Environment metadata
├── Dockerfile               # Containerization
├── README.md                # This file
│
├── agent.md                 # Build-ready v1 spec
├── inference.py             # Agent policy demo
│
├── trace/                   # Core environment package
│   ├── __init__.py
│   ├── models.py            # Pydantic schemas
│   ├── env.py               # TraceEnv gym-like interface
│   ├── scenarios.py         # Deterministic scenario generators
│   ├── simulator.py         # State transition engine
│   ├── rewards.py           # Cumulative reward calculator
│   ├── graders.py           # Outcome-based grader
│   └── utils.py             # Helpers
│
├── server/                  # FastAPI server
│   ├── __init__.py
│   └── app.py               # Routes: /reset, /step, /state, /health
│
└── tests/                   # Unit tests
    ├── __init__.py
    ├── test_env.py          # Environment interface
    ├── test_scenarios.py    # Scenario determinism
    ├── test_rewards.py      # Cumulative rewards
    └── test_graders.py      # Scoring logic
```

## Three Tasks

### 1. `easy_cpu_spike` (Max: 5 steps)

**Scenario:** Traffic spike causes CPU spike  
**Observable:** CPU ↑85%, latency ↑500ms, error_rate ↑5%  
**Hidden Root Cause:** Workload surge  
**Solution:** `scale_workers("api_workers", 5)`  

### 2. `medium_cascade` (Max: 7 steps)

**Scenario:** Queue memory leak cascades to workers  
**Observable:** Queue_depth rising, services degrading  
**Hidden Root Cause:** Queue service memory leak  
**Solution:** `restart_service("queue_service")`  

### 3. `hard_mixed` (Max: 8 steps)

**Scenario:** DB pool exhausted + bad release  
**Observable:** Error spike (20%), latency spike (2000ms), CPU high  
**Hidden Root Cause:** DB connection pool + inefficient queries  
**Solution:** `restart_database()` + optional `rollback_release()`  

## API Specification

### POST /reset

```json
Request:
{
    "task_id": "easy_cpu_spike",
    "seed": 42
}

Response:
{
    "observation": {...},
    "info": {"task_id": "easy_cpu_spike", "max_steps": 5}
}
```

### POST /step

```json
Request:
{
    "action": {
        "action_type": "scale_workers",
        "target": "api_workers",
        "value": 5
    }
}

Response:
{
    "observation": {...},
    "reward": 5.0,
    "done": false,
    "info": {...}
}
```

### GET /state

```json
Response:
{
    "observation": {...},
    "episode_reward": 5.0,
    "steps": 1,
    "done": false
}
```

### GET /health

```json
Response:
{
    "status": "healthy",
    "version": "0.1.0"
}
```

## Reward Mechanism

**Rewards accumulate and are only normalized at episode end (NOT per-step clamped).**

Step rewards:
- +1.0 for relevant inspection
- +5.0 for successful remediation
- -0.5 for duplicate action
- -2.0 for harmful action
- +10.0 bonus for correct `declare_healthy`

Final score: `0.6 * resolution + 0.4 * efficiency`

## Validation

```bash
openenv validate
./validate-submission.sh
```

## Deployment

### Docker

```bash
docker build -t trace:latest .
docker run -p 7860:7860 trace:latest
```

### HF Spaces

Push to `meta-trace` Space, enable auto-deploy.

## Success Criteria

✅ `openenv validate` passes  
✅ 3 deterministic scenarios  
✅ Cumulative-normalized rewards  
✅ Partial observability  
✅ (action_type, target, value) format  
✅ Grader: 0.6×success + 0.4×efficiency  
✅ All tests pass  

---

**Status:** Build-ready. See [agent.md](agent.md) for full spec.
