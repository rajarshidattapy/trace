# Trace

**AI-native incident response runtime for engineering teams**
OpenEnv benchmark environment for training and evaluating autonomous SRE / DevOps on-call agents.

---

## Overview

Trace is a **real-world OpenEnv environment** that simulates the workflow of a **Site Reliability Engineer (SRE) / DevOps on-call engineer** responding to production incidents.

This environment is designed for **agent learning via supervised fine-tuning (SFT) warm-starts and reinforcement learning (RL)** using the standard OpenEnv API:

```python
reset()
step(action)
state()
```

The benchmark simulates realistic production failure scenarios such as:

* service degradation
* cascading system failures
* noisy alerts
* false positives
* rollback decisions
* resource scaling
* root-cause diagnosis

This is **not a toy environment**.
It models a genuine production engineering workflow used in modern software teams.

---

## Motivation

Modern engineering teams rely heavily on:

* observability systems
* on-call workflows
* incident triage
* root cause analysis
* automated remediation

Trace provides a **reproducible benchmark** for evaluating how well AI agents can operate in this high-stakes environment.

The benchmark is useful for:

* RL post-training
* agent benchmarking
* SFT trajectory generation
* reward modeling
* systems-agent evaluation

---

## OpenEnv Specification Compliance

Trace implements the **full OpenEnv interface**.

### Typed Models

All interfaces are implemented using **Pydantic typed models**.

### Observation Model

```python
class Observation(BaseModel):
    cpu_usage: int
    memory_usage: int
    error_rate: float
    queue_backlog: int
    api_status: str
    db_status: str
    recent_logs: list[str]
```

### Action Model

```python
class Action(BaseModel):
    action_type: str
```

Supported actions:

* `inspect_logs`
* `inspect_metrics`
* `restart_service`
* `scale_workers`
* `rollback_release`
* `clear_queue`
* `escalate_incident`

### Reward Model

```python
class Reward(BaseModel):
    score: float
```

### Required API

```python
reset() -> Observation
step(action) -> (Observation, reward, done, info)
state() -> Observation
```

---

## Real-World Task Simulation

Trace simulates **production incident response workflows**.

The agent operates as an on-call engineer responsible for restoring service health.

Example production incident:

```json
{
  "cpu_usage": 96,
  "error_rate": 0.21,
  "queue_backlog": 1400,
  "api_status": "degraded",
  "db_status": "healthy"
}
```

Agent objective:

* diagnose root cause
* take correct remediation steps
* restore service
* minimize destructive actions
* resolve incident efficiently

---

## Tasks (Easy → Medium → Hard)

Trace includes **3 deterministic tasks with programmatic graders**.

---

### Task 1 — Easy

**Single Service CPU Spike**

Scenario:

* one service unhealthy
* CPU spike
* queue backlog increasing

Expected resolution:

```text
inspect_metrics → scale_workers
```

Target difficulty: **Easy**

---

### Task 2 — Medium

**Cascading API + Queue Failure**

Scenario:

* API degraded
* worker queue stalled
* rising latency

Expected resolution:

```text
inspect_logs → clear_queue → restart_service
```

Target difficulty: **Medium**

---

### Task 3 — Hard

**Adversarial Noisy Incident**

Scenario:

* false alerts
* partial logs
* misleading CPU spike
* actual issue = DB timeout

Expected resolution requires:

* multiple inspections
* root cause reasoning
* rollback or escalation

Target difficulty: **Hard**

---

## Deterministic Agent Graders

Each task includes a deterministic grader.

Score range:

```text
0.0 → 1.0
```

### Grading Criteria

```python
score =
    0.4 * diagnosis_accuracy +
    0.4 * remediation_correctness +
    0.2 * efficiency
```

Examples:

* correct root cause → +0.4
* correct fix → +0.4
* minimal step count → +0.2

Incorrect destructive actions:

* wrong restart → penalty
* repeated invalid action → penalty
* infinite loops → penalty

---

## Reward Function

Trace uses **dense reward shaping** across the full trajectory.

This is intentionally **not sparse binary reward**.

### Reward Logic

```python
reward =
    +0.20 for correct inspection step
    +0.30 for correct diagnosis
    +0.40 for successful remediation
    -0.20 for destructive action
    -0.10 per wasted step
```

This supports RL learning with:

* partial progress signals
* meaningful trajectory rewards
* stepwise optimization

Final reward is normalized:

```python
score = clamp(score, 0.0, 1.0)
```

---

## SFT + RL Pipeline Compatibility

Trace is designed for:

### Stage 1 — SFT

Successful trajectories can be converted into:

```text
(state, optimal_action)
```

datasets for supervised fine-tuning.

### Stage 2 — RL

The same environment provides reward-based learning using:

```python
env.step(action)
```

This supports:

* PPO
* policy gradient methods
* offline RL
* reward modeling

---

## Baseline Inference Script

A reproducible inference runner is included:

```text
inference.py
```

This file is located in the **repo root as required**.

The script:

* uses `OpenAI` client
* reads env variables
* runs all 3 tasks
* emits strict structured logs

### Required Environment Variables

```bash
API_BASE_URL
MODEL_NAME
HF_TOKEN
IMAGE_NAME
```

---

## Mandatory Structured STDOUT Logging

Trace strictly follows the required output format:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<...>
```

This format is required for automated scoring.

---

## Baseline Scores

Reference baseline (Qwen 72B instruct):

| Task   | Score |
| ------ | ----: |
| Easy   |  0.91 |
| Medium |  0.78 |
| Hard   |  0.62 |

Average:

```text
0.77
```

All scores are reproducible.

---

## Deployment

Trace is deployable as a **containerized Hugging Face Space**.

Required endpoint health checks:

```text
POST /reset
POST /step
GET /state
```

The HF Space must respond:

```text
HTTP 200
```

for:

```text
/reset
```

---

## Docker

Containerized execution is fully supported.

### Build

```bash
docker build -t trace .
```

### Run

```bash
docker run -p 7860:7860 trace
```

---

## Validation

Before submission, run the official validator:

```bash
chmod +x scripts/validate-submission.sh
./scripts/validate-submission.sh <hf-space-url>
```

This validates:

* HF Space health
* Docker build
* OpenEnv compliance

Exactly matching the provided validator workflow. 

---

## Local Validation

```bash
openenv validate
```

This must pass before submission.

---

## Runtime Constraints

Trace is optimized for:

```text
vCPU = 2
RAM = 8GB
runtime < 20 minutes
```

Inference is designed to complete within the hackathon infra limits.

---

## Project Structure

```text
trace/
├── README.md
├── openenv.yaml
├── Dockerfile
├── inference.py
├── app.py
├── trace/
├── tasks/
├── tests/
└── scripts/
```

---

## Why Trace Matters

Trace benchmarks a **real-world engineering workflow** that is directly useful for:

* autonomous incident response
* infrastructure agents
* DevOps copilots
* production SRE evaluation

This fills a meaningful gap in current agent benchmarking environments.
