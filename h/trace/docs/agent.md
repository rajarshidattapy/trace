# TRACE v1 Spec — OpenEnv Incident Response Environment

**Status:** Build-ready production v1  
**Owner:** Rajarshi Datta  
**Timeline:** 7 days  
**Target:** Meta × PyTorch × Hugging Face OpenEnv Hackathon

This spec incorporates critical feedback on the v2.0 PRD. It is **narrowed, execution-ready, and removes all ambiguities.**

---

## Executive Summary

TRACE is a **deterministic, partial-observability RL environment** for **incident response in production infrastructure**. An AI agent interacts with realistic infrastructure incidents by observing systems, running diagnostic actions, and executing remediation. The environment is:

- **OpenEnv-compliant** (pyproject.toml, server/app.py, openenv.yaml)
- **Deterministic** (3 hand-crafted scenarios)
- **Partially observable** (ground truth hidden behind `inspect_*` actions)
- **Action-structured** (action_type + target + value)
- **Outcome-graded** (no diagnosis_accuracy; only resolution success + efficiency)

**Verdict:** This v1 is buildable, complies with validator, and remains challenging.

---

## 1. Problem Statement

Production engineers spend significant time on:

1. **Triage** — filtering false positives from real alerts
2. **Inspection** — digging through logs and metrics
3. **Diagnosis** — identifying root cause
4. **Remediation** — executing fixes (scale, restart, rollback)
5. **Validation** — confirming recovery

Current RL benchmarks do **not** simulate this workflow. TRACE fills that gap.

---

## 2. Design Principles (v1)

### P1 — Partial Observability (FIX #1)

**Previous problem:** Observations exposed `db_status`, `worker_health`, `recent_logs`, `alerts` directly. This leaked too much ground truth.

**Fix:** Observation shows only:
- Generic telemetry (CPU, memory, latency, error_rate, queue_depth)
- Alert names (no context)
- Service status enums (healthy, degraded, down)

Ground truth details (logs, detailed metrics, alert context) are hidden behind inspection actions.

### P2 — Deterministic Scenarios

Exactly 3 hand-crafted incident types, all **reproducible**:

| Scenario        | Root Cause              | Typical Fix              |
|-----------------|-------------------------|--------------------------|
| easy_cpu_spike  | Worker overload         | scale_workers            |
| medium_cascade  | Queue deadlock cascades | restart_service          |
| hard_mixed      | DB + release regression | restart_database + wait  |

### P3 — Action Structure (FIX #2)

**Previous problem:** Actions had no target or magnitude (`restart_service`, `scale_workers` with no arity).

**Fix:** All actions use **triple format:**

```python
(action_type, target, value)
```

Examples:
- `("restart_service", "api_workers", None)`
- `("scale_workers", "api_workers", 5)`
- `("inspect_logs", "database", None)`

### P4 — Reward: Cumulative + Normalized (FIX #3)

**Previous problem:** Rewards clamped to [0,1] per step, causing penalties to collapse to 0.

**Fix:** 
- Collect all step rewards (no per-step clamping)
- Normalize **only at episode end**
- Ensures agent learns long-horizon causality

### P5 — Discovery Action for Diagnosis (FIX #4)

**Previous problem:** Grader includes `diagnosis_accuracy`, but action space has no way to state a diagnosis.

**Fix:** Remove `diagnosis_accuracy` from final grade. Grade only:
- **Resolution success** (binary: incident resolved or not)
- **Efficiency** (steps vs max_steps)

Agent learns diagnosis implicitly through remediation actions.

---

## 3. Environment Architecture

```
┌──────────────────────────────┐
│     inference.py             │
│    (LLM Agent Loop)          │
└──────────┬───────────────────┘
           │ HTTP
    ┌──────┴──────┐
    ▼             ▼
POST /step   GET /state
POST /reset   GET /health
    │             ▲
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │  TraceEnv   │
    │ (gym-like)  │
    └──────┬──────┘
           │
    ┌──────┴──────┬──────────┬─────────┐
    ▼             ▼          ▼         ▼
 scenarios   simulator    rewards   graders
```

---

## 4. Observation Space

```python
class Observation(BaseModel):
    timestamp: str                      # ISO8601
    
    # Metrics (always visible)
    cpu_usage_pct: float               # [0, 100]
    memory_usage_pct: float
    error_rate_pct: float
    api_latency_ms: float
    queue_depth: int
    
    # Service status (always visible, generic)
    services: dict[str, str]           # e.g., {"api_workers": "healthy"}
    
    # Alerts (names only, no context)
    active_alerts: list[str]           # e.g., ["alert_001", "alert_002"]
    
    # Inspection results (populated by inspect_* actions)
    last_inspection: Optional[dict]    # {"type": "logs", "target": "api_workers", "data": "..."}
```

**Key:** Root cause is hidden until agent calls `inspect_logs`, `inspect_metrics`, `inspect_alert`.

---

## 5. Action Space

```python
class Action(BaseModel):
    action_type: str
    target: Optional[str]   # service/metric/alert_id
    value: Optional[float]  # scaling factor, count, etc.
```

**Valid actions:**

| Action | Target | Value | Effect |
|--------|--------|-------|--------|
| `inspect_logs` | service_name | None | Returns log snippet (reveals cause) |
| `inspect_metrics` | metric_name | None | Returns metric timeseries |
| `inspect_alert` | alert_id | None | Returns alert details |
| `restart_service` | service_name | None | Resets service state |
| `scale_workers` | service_name | worker_count | Scales horizontally |
| `restart_database` | None | None | Resets DB state |
| `rollback_release` | None | None | Undoes recent deployment |
| `clear_queue` | None | None | Clears backlog |
| `declare_healthy` | None | None | Declare incident resolved (terminal) |
| `declare_unfixable` | None | None | Give up (terminal) |

---

## 6. Scenario Design

### Scenario 1: `easy_cpu_spike`

**Difficulty:** Beginner (2–4 steps)

**Trigger:** Sudden traffic spike floods API workers.

**Observable symptoms:**
- `cpu_usage_pct` → 85%
- `api_latency_ms` → 500ms
- `error_rate_pct` → 5%
- `active_alerts` → ["alert_cpu_high"]
- `services.api_workers` → "degraded"

**Hidden root cause:** Workload surge, solvable by horizontal scaling

**Optimal trajectory:**
```
1. Observe metrics (CPU high is visible)
2. inspect_logs("api_workers") → reveals "traffic spike, need more workers"
3. scale_workers("api_workers", 5) → CPU → 60%, incident recovers
4. declare_healthy() → DONE
```

**Reward:** Inspection (+1), Remediation (+5), Declare (+10) = success

---

### Scenario 2: `medium_cascade`

**Difficulty:** Intermediate (3–6 steps)

**Trigger:** Queue service memory leak + cascading worker failures.

**Observable symptoms (evolve over steps):**
- Step 1: `queue_depth` rising slowly
- Step 3: `queue_depth` > 500, `error_rate_pct` rising
- Step 5: `services.queue_service` → "degraded", worker timeouts begin
- Step 7: Multiple services → "degraded"

**Hidden root cause:** Queue memory leak; fixable by restart

**Optimal trajectory:**
```
1. Observe metrics (queue_depth unusual)
2. inspect_metrics("queue_depth") → "backlog critical"
3. inspect_logs("queue_service") → "memory usage high, leak suspected"
4. restart_service("queue_service") → queue resets, backlog clears
5. declare_healthy()
```

**Reward:** 2× Inspection (+2), Remediation (+5), Declare (+10) = strong success

---

### Scenario 3: `hard_mixed`

**Difficulty:** Advanced (4–8 steps)

**Trigger:** Recent release + DB connection pool exhaustion + cascading errors.

**Observable symptoms:**
- `error_rate_pct` spiking (5% → 20%)
- `api_latency_ms` very high (100 → 2000ms)
- Multiple alerts: `["alert_high_error_rate", "alert_db_slow", "alert_pool_exhaustion"]`
- `services.database` → "degraded"
- False lead: CPU is high (symptom, not cause)

**Hidden root cause:** DB pool exhausted (release added inefficient queries + not enough connections)

**Optimal trajectory:**
```
1. Observe metrics (error spike, latency spike)
2. inspect_alert("alert_pool_exhaustion") → "DB connection pool at 100%"
3. inspect_logs("database") → "recent release queries inefficient"
4. inspect_metrics("db_connections") → confirms pool exhaustion
5. restart_database() → pool resets, errors drop
6. [optional] rollback_release() if still degraded → teaches causality
7. declare_healthy()
```

**Reward:** 3+ Inspections (+3), Remediation (+8), Declare (+10) = strong success

---

## 7. Reward Structure (FIXED)

### Step-wise Rewards (Accumulated, No Per-Step Clamping)

```python
reward = 0

# Inspection
if action == inspect_logs and target is relevant:
    reward += 1.0
if action == inspect_metrics and target is relevant:
    reward += 1.0
if action == inspect_alert:
    reward += 0.5

# Remediation
if action solves active problem:
    reward += 5.0

# Penalties
if action is duplicate_recent:
    reward -= 0.5
if action worsens incident:
    reward -= 2.0
if action is irrelevant:
    reward -= 0.1

# Terminal
if declare_healthy() and incident_resolved:
    reward += 10.0
if declare_healthy() and NOT incident_resolved:
    reward -= 5.0
```

**All rewards summed across episode. No clamping until end.**

### Final Score (Outcome-Based)

```python
# Normalize accumulated reward
episode_reward = sum(step_rewards) / max_possible_reward
final_reward = min(max(episode_reward, 0), 1.0)

# Grading (NO diagnosis_accuracy)
score = (
    0.6 * (1.0 if incident_resolved else 0.0)  # binary success
    + 0.4 * (1.0 - steps_taken / max_steps)     # efficiency
)
```

**Example:** 
- Easy task: max_steps=5, agent solves in 3 → score = 0.6×1.0 + 0.4×(1 - 3/5) = 0.76
- Hard task: max_steps=8, agent solves in 8 → score = 0.6×1.0 + 0.4×(1 - 8/8) = 0.60

---

## 8. State Transition Logic

### Deterministic Stepping

Each episode uses a **scenario_clock** that progresses deterministically. Same seed → same trajectory.

```python
def transition(state, action) -> (next_state, reward, done):
    # Advance time
    state.timestamp = increment_time(state.seed)
    
    # Apply scenario progression (if no action taken)
    if action not relevant:
        state = apply_scenario_step(state)  # e.g., queue_depth grows
    
    # If remediation action, apply fix
    if action == restart_service:
        state.services[target] = "healthy"
        state = reset_related_metrics()
    
    # Check terminal
    if incident_resolved_enough():
        done = True
    
    return state, reward, done
```

**Property:** `transition(state, a, seed=42)` is deterministic.

---

## 9. API Routes

### POST /reset

Request:
```json
{
    "task_id": "easy_cpu_spike" | "medium_cascade" | "hard_mixed",
    "seed": 42
}
```

Response:
```json
{
    "observation": {...},
    "info": {
        "task_id": "easy_cpu_spike",
        "episode_id": "uuid",
        "max_steps": 5,
        "root_cause": "traffic_spike"
    }
}
```

### POST /step

Request:
```json
{
    "action": {
        "action_type": "scale_workers",
        "target": "api_workers",
        "value": 5
    }
}
```

Response:
```json
{
    "observation": {...},
    "reward": 5.0,
    "done": false,
    "info": {
        "step": 1,
        "episode_reward": 5.0,
        "message": "Workers scaled to 5"
    }
}
```

### GET /state

Response:
```json
{
    "observation": {...},
    "episode_reward": 5.0,
    "steps": 1,
    "done": false
}
```

### GET /health

Response:
```json
{
    "status": "healthy",
    "version": "0.1.0"
}
```

---

## 10. Project Structure

```text
TRACE/
├── pyproject.toml              # OpenEnv spec
├── uv.lock                     # Dependencies locked
├── README.md
├── openenv.yaml                # Environment metadata
├── Dockerfile
├── requirements.txt
├── .env.example
│
├── trace/                      # Core module
│   ├── __init__.py
│   ├── env.py                  # TraceEnv class
│   ├── models.py               # Pydantic schemas
│   ├── scenarios.py            # Scenario generators
│   ├── simulator.py            # State transitions
│   ├── rewards.py              # Reward engine
│   ├── graders.py              # Grading logic
│   └── utils.py                # Helpers
│
├── server/                     # FastAPI app
│   ├── __init__.py
│   └── app.py                  # Routes + server
│
├── inference.py                # Agent policy loop
│
├── tests/
│   ├── __init__.py
│   ├── test_env.py
│   ├── test_api.py
│   ├── test_rewards.py
│   ├── test_graders.py
│   └── test_scenarios.py
│
└── scripts/
    └── run_benchmark.py        # Local evaluation
```

---

## 11. Testing

### Unit Tests

1. **test_scenarios.py:** Seed determinism — verify same seed produces same trajectory
2. **test_rewards.py:** Cumulative rewards (no per-step clamping)
3. **test_graders.py:** Final score calculation
4. **test_env.py:** State transitions

### API Tests

1. POST /reset returns valid Observation
2. POST /step accepts valid Action, returns next state
3. GET /health returns 200 OK
4. Invalid action → 400 Bad Request

### Validation

```bash
openenv validate
./validate-submission.sh
```

---

## 12. Docker & Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

**HF Spaces:** Push to `meta-trace` repo, enable auto-deploy.

---

## 13. Inference Pipeline

**File:** `inference.py`

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("API_BASE_URL", "http://localhost:7860"),
    api_key=os.getenv("HF_TOKEN")
)

print("[START]")

# Agent loop
response = client.post("/reset", json={"task_id": "easy_cpu_spike", "seed": 0})
state = response.json()["observation"]
done = False

for step in range(MAX_STEPS):
    # LLM decides next action
    action = agent_policy(state)
    
    response = client.post("/step", json={"action": action})
    state = response.json()["observation"]
    reward = response.json()["reward"]
    done = response.json()["done"]
    
    if done:
        break

print("[END]")
```

**Emit exactly:**
- `[START]` before first step
- `[END]` after completion

---

## 14. Risk Register

| Risk | Mitigation |
|------|-----------|
| Validator fails on structure | Continuous `openenv validate` during dev |
| Scenarios become random | Seed-based RNG, determinism tests |
| Reward instability | No per-step clamp, cumulative only |
| Observability too opaque | 3 simple scenarios + dense inspection rewards |
| Diagnosis is ungraded | Removed from final score; implicit in remediation |

---

## 15. Success Criteria (v1 Complete)

✅ `pyproject.toml` + `uv.lock` present  
✅ `openenv validate` passes  
✅ 3 deterministic scenarios reproducible by seed  
✅ API: /reset, /step, /state, /health working  
✅ Rewards cumulative-normalized, no per-step clamp  
✅ Observations hide ground truth (partial observability)  
✅ Actions all use (type, target, value) format  
✅ Grader: 0.6×success + 0.4×efficiency (no diagnosis_accuracy)  
✅ `inference.py` runs, emits `[START]` and `[END]`  
✅ Docker builds and serves  
✅ All tests pass  

---

## 16. Execution Plan (7 Days)

| Day | Milestone |
|-----|-----------|
| 1–2 | Models + scenarios + simulator (determinism verified) |
| 3 | Rewards (cumulative logic) + graders |
| 4 | FastAPI server + Docker + `openenv validate` |
| 5 | `inference.py` + logging + tests |
| 6 | Deploy to HF Spaces |
| 7 | Polish + final validation |

---

**Status:** This spec is **build-ready**. Execute continuously against validator. No further design changes.
