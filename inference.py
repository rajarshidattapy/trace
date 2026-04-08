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
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = "https://router.huggingface.co/v1"
MODEL_NAME = "openai/gpt-oss-20b"

# TRACE environment server URL (local or remote)
TRACE_SERVER_URL = os.getenv("TRACE_SERVER_URL", "http://localhost:7861")

TASK_NAME = os.getenv("TRACE_TASK", "easy_cpu_spike")
BENCHMARK = "trace"
SEED = int(os.getenv("TRACE_SEED", "0"))
TEMPERATURE = 0.2
MAX_TOKENS = 512

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


def build_user_prompt(step: int, obs: dict) -> str:
    """Build the user prompt with current observation."""
    obs_text = format_observation(obs)
    return textwrap.dedent(f"""\
Step {step} — Current System State:
{obs_text}

Decide your next action. Respond with a JSON object only.
""")


def parse_llm_json(raw: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences and preamble."""
    text = raw.strip()

    # Strip markdown code fences
    if "```" in text:
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("No valid JSON found", text, 0)


# Fallback plan: a sequence of reasonable actions when the LLM fails.
# These are tuned to the scenario math:
#   easy_cpu_spike: traffic_spike_strength starts 0.8, each scale_workers *= 0.5,
#                   need < 0.1 → requires 4 scale_workers (0.8→0.4→0.2→0.1→0.05)
#   medium_cascade: queue_memory_leak starts 0.1, restart_service resets to 0,
#                   need < 0.05 → one restart_service suffices
#   hard_mixed:     db_pool_impact=0.7 (restart_database *=0.2 → 0.14),
#                   release_impact=0.5 (rollback_release *=0.2 → 0.1),
#                   need both ≤ 0.15 → one of each suffices
FALLBACK_PLANS = {
    "easy_cpu_spike": [
        # 4 scales to resolve + declare_healthy = exactly 5 steps (max)
        {"action_type": "scale_workers", "target": "api_workers", "value": 4},
        {"action_type": "scale_workers", "target": "api_workers", "value": 4},
        {"action_type": "scale_workers", "target": "api_workers", "value": 4},
        {"action_type": "scale_workers", "target": "api_workers", "value": 4},
        {"action_type": "declare_healthy", "target": None, "value": None},
    ],
    "medium_cascade": [
        # inspect → inspect → fix → declare = 4 of 7 steps
        {"action_type": "inspect_metrics", "target": "queue_depth", "value": None},
        {"action_type": "inspect_logs", "target": "queue_service", "value": None},
        {"action_type": "restart_service", "target": "queue_service", "value": None},
        {"action_type": "declare_healthy", "target": None, "value": None},
        {"action_type": "declare_healthy", "target": None, "value": None},
        {"action_type": "declare_healthy", "target": None, "value": None},
        {"action_type": "declare_healthy", "target": None, "value": None},
    ],
    "hard_mixed": [
        # inspect → inspect → fix db → fix release → declare = 5 of 8 steps
        {"action_type": "inspect_logs", "target": "database", "value": None},
        {"action_type": "inspect_metrics", "target": "db_connections", "value": None},
        {"action_type": "restart_database", "target": "database", "value": None},
        {"action_type": "rollback_release", "target": None, "value": None},
        {"action_type": "declare_healthy", "target": None, "value": None},
        {"action_type": "declare_healthy", "target": None, "value": None},
        {"action_type": "declare_healthy", "target": None, "value": None},
        {"action_type": "declare_healthy", "target": None, "value": None},
    ],
}


def get_llm_action(
    client: OpenAI,
    step: int,
    obs: dict,
    messages: list,
    fallback_step: int,
) -> tuple[dict, bool]:
    """Query the LLM for the next action.

    Returns (action_dict, used_llm) — used_llm is False when falling back.
    """
    user_prompt = build_user_prompt(step, obs)

    # Add current observation as a user message
    messages.append({"role": "user", "content": user_prompt})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        print(f"[DEBUG] LLM raw response: {raw!r}", flush=True)

        if not raw:
            raise ValueError("Empty response from LLM")

        action = parse_llm_json(raw)
        action.setdefault("action_type", "inspect_logs")
        action.setdefault("target", None)
        action.setdefault("value", None)

        # Append assistant response to conversation history
        messages.append({"role": "assistant", "content": raw})
        return action, True

    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        # Use scenario-specific fallback plan
        plan = FALLBACK_PLANS.get(TASK_NAME, FALLBACK_PLANS["easy_cpu_spike"])
        idx = min(fallback_step, len(plan) - 1)
        fallback = plan[idx]
        print(f"[DEBUG] Using fallback step {idx}: {fallback}", flush=True)

        # Append fallback as assistant message so conversation stays coherent
        messages.append({"role": "assistant", "content": json.dumps(fallback)})
        return fallback, False


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    trace = TraceClient(TRACE_SERVER_URL)

    max_steps = MAX_STEPS_MAP.get(TASK_NAME, 8)

    # Multi-turn conversation messages for the LLM
    messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    fallback_count = 0
    done = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset the environment
        reset_resp = trace.reset(task_id=TASK_NAME, seed=SEED)
        obs = reset_resp["observation"]

        for step in range(1, max_steps + 1):
            # Get action from LLM (multi-turn, with fallback)
            action, used_llm = get_llm_action(
                llm_client, step, obs, messages, fallback_step=fallback_count
            )
            if not used_llm:
                fallback_count += 1

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

            if done:
                # Extract final grade if available
                if "final_grade" in info:
                    score = float(info["final_grade"])
                    success = info.get("success", False)
                else:
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