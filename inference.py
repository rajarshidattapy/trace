"""
TRACE v1 — Inference Script
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]
"""

import json
import os
import textwrap
from typing import List, Optional

import httpx
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# TRACE environment server URL (local or remote)
TRACE_SERVER_URL = os.getenv("TRACE_SERVER_URL", "http://localhost:7860")

TASK_NAME = os.getenv("TRACE_TASK", "easy_cpu_spike")
BENCHMARK = "trace"
SEED = int(os.getenv("TRACE_SEED", "0"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "300"))

# Max steps per scenario
MAX_STEPS_MAP = {
    "easy_cpu_spike": 5,
    "medium_cascade": 7,
    "hard_mixed": 8,
}

# ── Logging helpers ──────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── TRACE HTTP client ────────────────────────────────────────────────────────

class TraceClient:
    """Simple HTTP client for the TRACE environment server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)

    def reset(self, task_id: str, seed: int = 0) -> dict:
        resp = self.client.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action: dict) -> dict:
        resp = self.client.post(
            f"{self.base_url}/step",
            json={"action": action},
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict:
        resp = self.client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        resp = self.client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self.client.close()


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert production incident response agent. You are connected to the
TRACE environment, which simulates real infrastructure incidents. Your goal is to:

1. Observe the system metrics and alerts.
2. Inspect logs, metrics, and alerts to discover the root cause.
3. Take remediation actions to resolve the incident.
4. Declare healthy once the incident is resolved.

You MUST respond with a JSON object containing exactly these fields:
{
    "action_type": "<one of: inspect_logs, inspect_metrics, inspect_alert, restart_service, scale_workers, restart_database, rollback_release, clear_queue, declare_healthy, declare_unfixable>",
    "target": "<service name, metric name, or alert id — or null>",
    "value": <numeric value for scale_workers, or null>
}

Available services: api_workers, queue_service, database
Available metrics for inspect_metrics: cpu_usage_pct, memory_usage_pct, error_rate_pct, api_latency_ms, queue_depth, db_connections

Strategy tips:
- Start by inspecting logs or metrics of services that appear degraded.
- Look at active alerts and inspect them for context.
- Once you identify the root cause, take the appropriate remediation action.
- After remediation, declare_healthy if metrics have improved.

Respond ONLY with the JSON object. No explanation, no markdown, no extra text.
""")


def format_observation(obs: dict) -> str:
    """Format observation into a readable string for the LLM."""
    lines = [
        f"Timestamp: {obs.get('timestamp', 'N/A')}",
        f"CPU Usage: {obs.get('cpu_usage_pct', 0):.1f}%",
        f"Memory Usage: {obs.get('memory_usage_pct', 0):.1f}%",
        f"Error Rate: {obs.get('error_rate_pct', 0):.1f}%",
        f"API Latency: {obs.get('api_latency_ms', 0):.0f} ms",
        f"Queue Depth: {obs.get('queue_depth', 0)}",
        f"Services: {json.dumps(obs.get('services', {}))}",
        f"Active Alerts: {json.dumps(obs.get('active_alerts', []))}",
    ]
    inspection = obs.get("last_inspection")
    if inspection:
        lines.append(f"Last Inspection Result: {json.dumps(inspection)}")
    return "\n".join(lines)


def build_user_prompt(step: int, obs: dict, history: List[str]) -> str:
    """Build the user prompt with current observation and history."""
    obs_text = format_observation(obs)
    history_block = "\n".join(history[-5:]) if history else "None"
    return textwrap.dedent(f"""\
Step {step} — Current System State:
{obs_text}

Previous actions:
{history_block}

Decide your next action. Respond with a JSON object only.
""")


def get_llm_action(
    client: OpenAI,
    step: int,
    obs: dict,
    history: List[str],
) -> dict:
    """Query the LLM for the next action."""
    user_prompt = build_user_prompt(step, obs, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines).strip()

        action = json.loads(raw)
        # Ensure required fields
        action.setdefault("action_type", "inspect_logs")
        action.setdefault("target", None)
        action.setdefault("value", None)
        return action

    except (json.JSONDecodeError, Exception) as exc:
        print(f"[DEBUG] LLM parse error: {exc} — raw: {raw if 'raw' in dir() else 'N/A'}", flush=True)
        # Fallback: inspect logs as a safe default
        return {"action_type": "inspect_logs", "target": "api_workers", "value": None}


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    trace = TraceClient(TRACE_SERVER_URL)

    max_steps = MAX_STEPS_MAP.get(TASK_NAME, 8)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset the environment
        reset_resp = trace.reset(task_id=TASK_NAME, seed=SEED)
        obs = reset_resp["observation"]

        for step in range(1, max_steps + 1):
            # Get action from LLM
            action = get_llm_action(llm_client, step, obs, history)

            # Format action string for logging
            action_str = f"{action['action_type']}({action.get('target', '')},{action.get('value', '')})"

            # Execute step
            step_resp = trace.step(action)
            obs = step_resp["observation"]
            reward = float(step_resp.get("reward", 0.0))
            done = step_resp.get("done", False)
            info = step_resp.get("info", {})
            error = info.get("error", None)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {action_str} -> reward={reward:+.2f} done={done}"
            )

            if done:
                # Extract final grade if available
                if "final_grade" in info:
                    score = float(info["final_grade"])
                    success = info.get("success", False)
                else:
                    # Estimate score from rewards
                    score = max(0.0, min(1.0, sum(rewards) / (max_steps * 5.0)))
                    success = info.get("is_resolved", False)
                break

        if not done:
            # Episode ended by hitting max_steps without terminal action
            state_resp = trace.state()
            score = 0.0
            success = False

    except Exception as exc:
        print(f"[DEBUG] Exception during episode: {exc}", flush=True)
        score = 0.0
        success = False

    finally:
        try:
            trace.close()
        except Exception as e:
            print(f"[DEBUG] trace.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()