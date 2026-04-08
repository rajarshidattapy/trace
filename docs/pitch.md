# TRACE — Teaching AI to Fix Production Incidents

> **Triage · Response · Action · Cause · Evaluation**

---

## 🔥 The Problem

**Production incidents cost companies millions per hour.**

When systems break at 3 AM, engineers scramble through dashboards, comb through logs, and guess at root causes. The average MTTR (Mean Time to Resolution) for critical incidents is **4+ hours** — and it's getting worse as systems grow more complex.

Current AI coding agents can write code, but **none of them can operate production infrastructure under pressure.**

Why? Because there's no training ground for it.

---

## 💡 The Solution: TRACE

TRACE is the **first RL environment designed to teach AI agents how to respond to production incidents.**

Think of it as a **flight simulator for Site Reliability Engineers** — but for AI.

```
Agent observes metrics → inspects systems → diagnoses root cause → executes fix → validates recovery
```

One environment. Three difficulty levels. Infinite training runs.

---

## 🎯 How It Works

### The Agent Loop

```
┌─────────────────────────────────────────────┐
│                  LLM Agent                  │
│        "CPU is at 85%... let me check       │
│         the logs for api_workers"           │
└──────────────────┬──────────────────────────┘
                   │ action: inspect_logs("api_workers")
                   ▼
┌─────────────────────────────────────────────┐
│              TRACE Environment              │
│                                             │
│  📊 Metrics    🖥️ Services    🚨 Alerts      │
│  CPU: 85%     api: degraded   cpu_high      │
│  Latency: 500ms                             │
│                                             │
│  🔍 "Traffic spike detected. Scale workers" │
└──────────────────┬──────────────────────────┘
                   │ reward: +1.0 (useful inspection)
                   ▼
          Agent learns to diagnose → fix → validate
```

### Partial Observability — Like Real Life

The agent sees **symptoms**, not causes:
- ✅ CPU usage, memory, latency, error rates
- ✅ Service statuses (healthy / degraded / down)
- ✅ Alert names

But **root cause is hidden** behind inspection actions — just like a real engineer who has to `grep` the logs.

### Three Scenarios, Increasing Complexity

| Scenario | What Breaks | Root Cause | How to Fix |
|----------|-------------|------------|------------|
| 🟢 **Easy** | Traffic spike | Worker overload | Scale horizontally |
| 🟡 **Medium** | Cascading failures | Queue memory leak | Restart service |
| 🔴 **Hard** | Multi-service outage | DB pool + bad release | Restart DB + rollback |

### Smart Grading

```python
score = 0.6 × did_you_fix_it + 0.4 × how_fast_were_you
```

No hand-wavy metrics. Binary success + speed. That's it.

---

## 🏗️ Technical Design

### OpenEnv Compliant

TRACE is built to the **OpenEnv specification** — the emerging standard for RL environment benchmarks:

- `pyproject.toml` + `openenv.yaml` — auto-validated
- 4 REST endpoints: `/reset`, `/step`, `/state`, `/health`
- Docker-ready, deploys to HF Spaces in one push

### Deterministic & Reproducible

Same seed → same metrics → same trajectory. Every time.

This isn't a toy random environment. The scenarios are **hand-crafted to test real incident response patterns** — triage, diagnosis, remediation, validation.

### Action Space

10 structured actions using `(type, target, value)` triples:

```
inspect_logs("database")           → reveals root cause
scale_workers("api_workers", 5)    → horizontal scaling
restart_database()                 → resets connection pool
declare_healthy()                  → "I fixed it" (terminal)
```

---

## 📊 Results

Running our benchmark with heuristic agents:

| Scenario | Steps | Grade | Status |
|----------|-------|-------|--------|
| Easy CPU Spike | 3/5 | 0.76 | ✅ Solved |
| Medium Cascade | 3/7 | 0.83 | ✅ Solved |
| Hard Mixed | 4/8 | 0.80 | ✅ Solved |

**Average: 0.80** — and that's with hand-coded heuristics. The ceiling for LLM agents is much higher.

---

## 🦾 Why This Matters

### For AI Research
- First standardized benchmark for **operational AI** (not just coding)
- Partial observability forces genuine **reasoning under uncertainty**  
- Structured action space enables **reward shaping** without reward hacking

### For the Industry
- Train agents to reduce MTTR from hours to minutes
- Build **autonomous incident response** systems
- Bridge the gap between "AI writes code" and "AI runs production"

### For the Hackathon
- ✅ OpenEnv validator passes
- ✅ 28 tests, all green
- ✅ Full inference pipeline with LLM agent
- ✅ Interactive Gradio demo
- ✅ Docker builds and serves
- ✅ Push-to-deploy HF Spaces ready

---

## 🚀 Demo

### Live Interactive UI

```
python ui.py
→ Opens Gradio dashboard at localhost:7861
→ Select scenario, take actions, watch metrics change in real-time
```

### LLM Agent Run

```
export HF_TOKEN=your-token
python inference.py

[START] task=easy_cpu_spike env=trace model=openai/gpt-oss-20b
[STEP] step=1 action=inspect_logs(api_workers,) reward=1.00 done=false error=null
[STEP] step=2 action=scale_workers(api_workers,5) reward=5.00 done=false error=null
[STEP] step=3 action=declare_healthy(,) reward=10.00 done=true error=null
[END] success=true steps=3 score=0.840 rewards=1.00,5.00,10.00
```

---

## 👤 Team

**Rajarshi Datta** — Builder, designer, engineer.

---

## 📦 Stack

| Component | Technology |
|-----------|-----------|
| Environment | Python, Pydantic |
| Server | FastAPI, Uvicorn |
| Inference | OpenAI Client, HF Router |
| Model | openai/gpt-oss-20b |
| Deployment | Docker, HF Spaces |
| Demo | Gradio |
| Spec | OpenEnv |

---

## One Line

**TRACE teaches AI agents to fix production incidents — the missing benchmark between "AI writes code" and "AI runs production."**

---

*Built for the Meta × PyTorch × Hugging Face OpenEnv Hackathon.*
