"""
TRACE v1 — Gradio Demo UI
==========================
Interactive UI that lets you play through TRACE scenarios step-by-step.
No need to run the FastAPI server separately — this embeds TraceEnv directly.

Usage:
    pip install gradio
    python ui.py
"""

import time
import gradio as gr
import json
from trace.env import TraceEnv
from trace.models import Action

# ── Global state ─────────────────────────────────────────────────────────────

env = TraceEnv()
episode_log = []
current_obs = None

# ── Scenario descriptions (shown on reset, no spoilers) ─────────────────────

SCENARIO_BRIEFINGS = {
    "easy_cpu_spike": {
        "title": "CPU Spike Recovery",
        "briefing": (
            "**Incident alert:** API latency has spiked and CPU is critically high. "
            "Users are reporting slow responses. Your monitoring shows elevated load "
            "across the API worker fleet.\n\n"
            "**Your mission:** Investigate the cause and bring the system back to healthy."
        ),
        "difficulty": "Easy",
    },
    "medium_cascade": {
        "title": "Cascading Queue Failure",
        "briefing": (
            "**Incident alert:** Error rates are climbing and the message queue is backing up. "
            "Multiple downstream services are starting to degrade. Memory usage is trending upward.\n\n"
            "**Your mission:** Find the failing component and stop the cascade before it takes "
            "down the whole system."
        ),
        "difficulty": "Medium",
    },
    "hard_mixed": {
        "title": "Multi-Service Outage",
        "briefing": (
            "**Incident alert:** A major outage is in progress. Database queries are timing out, "
            "error rates are through the roof, and latency is in the seconds. "
            "This started shortly after a deployment.\n\n"
            "**Your mission:** Multiple things are broken. You'll need to diagnose and fix "
            "more than one root cause to fully resolve this."
        ),
        "difficulty": "Hard",
    },
}

# ── Optimal demo paths (for Auto Demo) ──────────────────────────────────────

AUTO_DEMO_PATHS = {
    "easy_cpu_spike": [
        ("inspect_logs", "api_workers", None, "First, check the API worker logs to understand what's happening..."),
        ("scale_workers", "api_workers", 5.0, "Logs say traffic spike! Scale out the workers to handle the load..."),
        ("declare_healthy", None, None, "Metrics look good. Declare the incident resolved!"),
    ],
    "medium_cascade": [
        ("inspect_metrics", "queue_depth", None, "Queue depth is suspicious. Let's inspect that metric..."),
        ("inspect_logs", "queue_service", None, "Queue is backing up. Check the queue service logs for the root cause..."),
        ("restart_service", "queue_service", None, "Memory leak found! Restart the queue service to clear it..."),
        ("declare_healthy", None, None, "Queue is draining, errors dropping. Declare healthy!"),
    ],
    "hard_mixed": [
        ("inspect_logs", "database", None, "DB is timing out. Check the database logs..."),
        ("restart_database", None, None, "Connection pool exhausted! Restart the database to reset it..."),
        ("rollback_release", None, None, "Logs mention a bad release. Roll back to the previous version..."),
        ("declare_healthy", None, None, "Both issues fixed. Declare the incident resolved!"),
    ],
}

# ── Target options per action type ───────────────────────────────────────────

TARGET_OPTIONS = {
    "inspect_logs": ["api_workers", "queue_service", "database"],
    "inspect_metrics": ["cpu_usage_pct", "memory_usage_pct", "error_rate_pct", "api_latency_ms", "queue_depth", "db_connections"],
    "inspect_alert": ["alert_cpu_high", "alert_high_error_rate", "alert_queue_backlog", "alert_db_slow", "alert_pool_exhaustion"],
    "restart_service": ["api_workers", "queue_service", "database"],
    "scale_workers": ["api_workers"],
    "restart_database": [],
    "rollback_release": [],
    "clear_queue": ["queue_service"],
    "declare_healthy": [],
    "declare_unfixable": [],
}

NEEDS_VALUE = {"scale_workers"}

# ── Formatting helpers ───────────────────────────────────────────────────────


def format_metrics(obs):
    """Format observation metrics as readable markdown."""
    if obs is None:
        return "No observation yet. Reset the environment first."

    d = obs.model_dump() if hasattr(obs, 'model_dump') else obs

    def bar(val, max_val=100, width=10):
        filled = int((val / max_val) * width)
        return "\u2588" * filled + "\u2591" * (width - filled)

    def status_color(val, warn, crit):
        if val > crit:
            return "\U0001f534"
        elif val > warn:
            return "\U0001f7e1"
        return "\U0001f7e2"

    cpu = d.get('cpu_usage_pct', 0)
    mem = d.get('memory_usage_pct', 0)
    err = d.get('error_rate_pct', 0)
    lat = d.get('api_latency_ms', 0)
    queue = d.get('queue_depth', 0)

    metrics_md = f"""| Metric | Value | Bar | |
|--------|------:|-----|---|
| CPU Usage | {cpu:.1f}% | `{bar(cpu)}` | {status_color(cpu, 50, 75)} |
| Memory | {mem:.1f}% | `{bar(mem)}` | {status_color(mem, 60, 80)} |
| Error Rate | {err:.1f}% | `{bar(err, 30)}` | {status_color(err, 2, 5)} |
| Latency | {lat:.0f}ms | `{bar(min(lat,2000), 2000)}` | {status_color(lat, 300, 1000)} |
| Queue | {queue} | `{bar(min(queue,1000), 1000)}` | {status_color(queue, 100, 500)} |"""

    return metrics_md


def format_services(obs):
    """Format services status."""
    if obs is None:
        return ""
    d = obs.model_dump() if hasattr(obs, 'model_dump') else obs

    def svc_emoji(s):
        return {"healthy": "\U0001f7e2", "degraded": "\U0001f7e1", "down": "\U0001f534"}.get(s, "\u26aa")

    lines = []
    for k, v in d.get("services", {}).items():
        lines.append(f"{svc_emoji(v)} **{k}** \u2014 {v}")
    return "\n\n".join(lines) if lines else "_No services_"


def format_alerts(obs):
    """Format active alerts."""
    if obs is None:
        return ""
    d = obs.model_dump() if hasattr(obs, 'model_dump') else obs
    alerts = d.get("active_alerts", [])
    if not alerts:
        return "\u2705 No active alerts"
    return "\n\n".join(f"\U0001f6a8 `{a}`" for a in alerts)


def format_inspection(obs):
    """Format last inspection."""
    if obs is None:
        return "_No inspection yet_"
    d = obs.model_dump() if hasattr(obs, 'model_dump') else obs
    inspection = d.get("last_inspection")
    if inspection and inspection.get("message"):
        return f'\U0001f4a1 **{inspection["message"]}**'
    return "_No inspection results \u2014 use inspect actions to investigate_"


def format_episode_log():
    """Format episode log as markdown table."""
    if not episode_log:
        return "_No actions taken yet._"

    rows = ["| # | Action | Reward | Status |", "|---|--------|--------|--------|"]
    for entry in episode_log:
        emoji = "\u2705" if entry.get("reward", 0) > 0 else "\u274c" if entry.get("reward", 0) < 0 else "\u2796"
        status = "\U0001f3c1 Done" if entry.get("done") else "\u25b6\ufe0f"
        rows.append(
            f"| {entry['step']} | `{entry['action']}` | {emoji} {entry['reward']:+.1f} | {status} |"
        )
    return "\n".join(rows)


def format_score_bar():
    """Format current episode score."""
    if not episode_log:
        return ""

    total = sum(e.get("reward", 0) for e in episode_log)
    steps = len(episode_log)
    return f"**Episode Reward:** {total:+.1f} &nbsp;|&nbsp; **Steps:** {steps}"


def format_grade():
    """Format final grade if episode is done."""
    if not env.done:
        return ""

    state = env.state()
    total_reward = state.get("episode_reward", 0)
    steps = state.get("steps", 0)

    return f"\U0001f3c6 **Total Reward:** {total_reward:.2f} | **Steps:** {steps}"


# ── Core actions ─────────────────────────────────────────────────────────────


def build_all_outputs(obs, status_msg, grade_md="", narrator=""):
    """Build the full tuple of outputs for Gradio."""
    return (
        format_metrics(obs),
        format_services(obs),
        format_alerts(obs),
        format_inspection(obs),
        status_msg,
        format_episode_log(),
        format_score_bar(),
        grade_md,
        narrator,
        gr.update(interactive=not env.done),
    )


def reset_env(task_id, seed):
    """Reset the environment."""
    global current_obs, episode_log

    episode_log = []

    try:
        seed_int = int(seed)
    except ValueError:
        seed_int = 0

    obs = env.reset(task_id=task_id, seed=seed_int)
    current_obs = obs

    briefing = SCENARIO_BRIEFINGS[task_id]
    max_steps = {"easy_cpu_spike": 5, "medium_cascade": 7, "hard_mixed": 8}[task_id]

    status_msg = f"\u2705 **Reset** \u2014 `{task_id}` | Seed: {seed_int} | Max steps: {max_steps}"
    narrator = (
        f"### \U0001f4cb {briefing['title']}  \u2014  Difficulty: **{briefing['difficulty']}**\n\n"
        f"{briefing['briefing']}\n\n"
        f"---\n"
        f"*Start by using **inspect_logs** or **inspect_metrics** on a service that looks unhealthy.*"
    )

    return build_all_outputs(obs, status_msg, narrator=narrator)


def take_action(action_type, target, value):
    """Execute one step."""
    global current_obs

    if env.simulator is None:
        return build_all_outputs(
            None,
            "\u26a0\ufe0f **Reset the environment first!**",
            narrator="\u261d\ufe0f Pick a scenario from the dropdown and click **Reset Environment** to start.",
        )

    if env.done:
        return build_all_outputs(
            current_obs,
            "\U0001f3c1 **Episode over.** Reset to play again.",
            grade_md=format_grade(),
            narrator="This incident is closed. Click **Reset Environment** to try another scenario.",
        )

    # Parse value
    val = None
    if value and value.strip():
        try:
            val = float(value)
        except ValueError:
            pass

    # Parse target
    tgt = target.strip() if target and target.strip() else None

    action = Action(action_type=action_type, target=tgt, value=val)
    obs, reward, done, info = env.step(action)
    current_obs = obs

    # Log
    action_str = f"{action_type}({tgt or ''}, {val or ''})" if tgt or val else action_type
    episode_log.append({
        "step": info.get("step", len(episode_log) + 1),
        "action": action_str,
        "reward": reward,
        "done": done,
    })

    # Status + narrator
    if done:
        if info.get("success"):
            status = f"\U0001f389 **Resolved!** Grade: {info.get('final_grade', 0):.3f} | Efficiency: {info.get('efficiency', 0):.2f}"
            narrator = (
                f"\U0001f389 **Incident resolved!**\n\n"
                f"**Grade:** {info.get('final_grade', 0):.3f} &nbsp;|&nbsp; "
                f"**Efficiency:** {info.get('efficiency', 0):.1%}\n\n"
                f"You diagnosed and fixed the issue in **{info.get('step', '?')} steps**. "
                f"Try a harder scenario or a different approach to improve your score!"
            )
        else:
            status = f"\U0001f480 **Failed** \u2014 Grade: {info.get('final_grade', 0):.3f}"
            narrator = (
                f"\U0001f480 **Incident not resolved.**\n\n"
                f"The system is still broken. Try resetting and using inspection actions "
                f"to find the root cause before applying fixes."
            )
        grade_md = format_grade()
    else:
        resolved_icon = "\u2705" if info.get('is_resolved') else "\u274c"
        status = f"\u25b6\ufe0f Step {info.get('step', '?')} \u2014 Reward: {reward:+.1f} | Resolved: {resolved_icon}"
        grade_md = ""

        # Contextual narrator hints based on what just happened
        if reward > 0 and action_type.startswith("inspect_"):
            insp = obs.model_dump().get("last_inspection", {})
            msg = insp.get("message", "") if insp else ""
            narrator = f"\U0001f50d **Good inspection!** (+{reward:.1f})\n\n> {msg}\n\n*Use this clue to decide your next action.*"
        elif reward > 0:
            narrator = f"\u2705 **Effective action!** (+{reward:.1f})\n\n*The system is responding. Check if metrics have improved, then consider declaring healthy.*"
        elif reward < 0:
            narrator = f"\u26a0\ufe0f **That didn't help.** ({reward:+.1f})\n\n*Try inspecting logs or metrics first to find the root cause before applying fixes.*"
        else:
            narrator = "*Action executed. Check the metrics and decide your next move.*"

    return build_all_outputs(obs, status, grade_md=grade_md, narrator=narrator)


def update_targets(action_type):
    """Update target dropdown choices when action type changes."""
    targets = TARGET_OPTIONS.get(action_type, [])
    needs_val = action_type in NEEDS_VALUE
    if targets:
        return gr.update(choices=targets, value=targets[0], interactive=True), gr.update(
            visible=needs_val, value="5" if needs_val else ""
        )
    return gr.update(choices=[], value="", interactive=False), gr.update(
        visible=needs_val, value=""
    )


def run_auto_demo(task_id, seed):
    """Run the optimal path for the selected scenario, yielding step-by-step."""
    global current_obs, episode_log

    episode_log = []
    try:
        seed_int = int(seed)
    except ValueError:
        seed_int = 0

    obs = env.reset(task_id=task_id, seed=seed_int)
    current_obs = obs

    briefing = SCENARIO_BRIEFINGS[task_id]
    max_steps = {"easy_cpu_spike": 5, "medium_cascade": 7, "hard_mixed": 8}[task_id]

    status_msg = f"\u2705 **Auto Demo** \u2014 `{task_id}` | Seed: {seed_int}"
    narrator = (
        f"### \U0001f3ac Auto Demo: {briefing['title']}\n\n"
        f"{briefing['briefing']}\n\n"
        f"---\n*Watch the optimal solution unfold step by step...*"
    )

    yield build_all_outputs(obs, status_msg, narrator=narrator)

    path = AUTO_DEMO_PATHS.get(task_id, [])
    for action_type, target, value, narration in path:
        time.sleep(2)

        if env.done:
            break

        action = Action(action_type=action_type, target=target, value=value)
        obs, reward, done, info = env.step(action)
        current_obs = obs

        action_str = f"{action_type}({target or ''})" if target else action_type
        if value is not None:
            action_str = f"{action_type}({target or ''}, {value})"
        episode_log.append({
            "step": info.get("step", len(episode_log) + 1),
            "action": action_str,
            "reward": reward,
            "done": done,
        })

        if done:
            if info.get("success"):
                status = f"\U0001f389 **Resolved!** Grade: {info.get('final_grade', 0):.3f}"
                narrator = (
                    f"\U0001f3ac **{narration}**\n\n"
                    f"\U0001f389 **Demo complete!** Grade: **{info.get('final_grade', 0):.3f}** in "
                    f"**{info.get('step', '?')} steps**.\n\n"
                    f"*Now try it yourself \u2014 can you beat this score?*"
                )
            else:
                status = f"\U0001f480 **Failed** \u2014 Grade: {info.get('final_grade', 0):.3f}"
                narrator = f"\U0001f3ac **{narration}**\n\n\U0001f480 Demo ended without resolution."
            grade_md = format_grade()
        else:
            status = f"\u25b6\ufe0f Step {info.get('step', '?')} \u2014 Reward: {reward:+.1f}"
            grade_md = ""
            narrator = f"\U0001f3ac **Step {info.get('step', '?')}:** {narration}"

        yield build_all_outputs(obs, status, grade_md=grade_md, narrator=narrator)


# ── Gradio UI ────────────────────────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

CSS = """
.gradio-container { max-width: 1400px !important; margin: auto !important; }
.metric-panel { min-height: 200px; }
.action-btn { min-height: 50px !important; font-weight: 700 !important; font-size: 16px !important; }
.narrator-box { background: #f0f4ff; border-left: 4px solid #3b82f6; padding: 12px 16px; border-radius: 8px; }
"""

with gr.Blocks(title="TRACE v1 \u2014 Incident Response Demo") as demo:

    # ── Header ──
    gr.Markdown("""
# TRACE v1 \u2014 Incident Response Environment
**Teaching AI to fix production incidents.** Pick a scenario, investigate the system, diagnose the root cause, apply a fix, and declare healthy.

> **How to play:** 1) Select a scenario  2) Click **Reset**  3) Use **inspect** actions to find the root cause  4) Apply the right **fix**  5) Click **declare_healthy** when metrics recover.
> Or click **Auto Demo** to watch the optimal solution.
    """)

    # ── Row 1: Status bar ──
    status_display = gr.Markdown("\U0001f449 Select a scenario and click **Reset** to begin.")

    # ── Row 2: Main layout ──
    with gr.Row(equal_height=True):

        # ── Column 1: Controls ──
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### Controls")
            task_dropdown = gr.Dropdown(
                choices=["easy_cpu_spike", "medium_cascade", "hard_mixed"],
                value="easy_cpu_spike",
                label="Scenario",
            )
            seed_input = gr.Textbox(value="42", label="Seed")
            with gr.Row():
                reset_btn = gr.Button("Reset Environment", variant="primary", elem_classes=["action-btn"])
                auto_btn = gr.Button("Auto Demo", variant="secondary", elem_classes=["action-btn"])

            gr.Markdown("---")
            gr.Markdown("### Take Action")
            action_dropdown = gr.Dropdown(
                choices=list(TARGET_OPTIONS.keys()),
                value="inspect_logs",
                label="Action Type",
            )
            target_dropdown = gr.Dropdown(
                choices=TARGET_OPTIONS["inspect_logs"],
                value="api_workers",
                label="Target",
                interactive=True,
            )
            value_input = gr.Textbox(value="", label="Value (for scale_workers)", placeholder="e.g. 5", visible=False)
            action_btn = gr.Button("Execute Action", variant="primary", elem_classes=["action-btn"])

        # ── Column 2: Metrics Dashboard ──
        with gr.Column(scale=2, min_width=400):
            gr.Markdown("### System Metrics")
            metrics_display = gr.Markdown("_Reset to see metrics_", elem_classes=["metric-panel"])

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Services")
                    services_display = gr.Markdown("")
                with gr.Column(scale=1):
                    gr.Markdown("#### Alerts")
                    alerts_display = gr.Markdown("")

            gr.Markdown("#### Inspection Result")
            inspection_display = gr.Markdown("_No inspection yet_")

        # ── Column 3: Log + Narrator ──
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### Narrator")
            narrator_display = gr.Markdown(
                "Welcome! This is a **flight simulator for incident response**.\n\n"
                "You'll see a production system that's broken. Your job is to:\n"
                "1. **Inspect** logs, metrics, and alerts\n"
                "2. **Diagnose** the root cause\n"
                "3. **Fix** it with the right remediation\n"
                "4. **Declare healthy** once it's resolved\n\n"
                "*Start by selecting a scenario and clicking Reset.*",
                elem_classes=["narrator-box"],
            )

            gr.Markdown("---")
            gr.Markdown("### Episode Log")
            score_display = gr.Markdown("")
            log_display = gr.Markdown("_No actions yet_")
            grade_display = gr.Markdown("")

    # ── All outputs (shared by reset, action, auto demo) ──
    all_outputs = [
        metrics_display, services_display, alerts_display,
        inspection_display, status_display, log_display,
        score_display, grade_display, narrator_display, action_btn,
    ]

    # ── Event handlers ──
    action_dropdown.change(
        fn=update_targets,
        inputs=[action_dropdown],
        outputs=[target_dropdown, value_input],
    )

    reset_btn.click(
        fn=reset_env,
        inputs=[task_dropdown, seed_input],
        outputs=all_outputs,
    )

    action_btn.click(
        fn=take_action,
        inputs=[action_dropdown, target_dropdown, value_input],
        outputs=all_outputs,
    )

    auto_btn.click(
        fn=run_auto_demo,
        inputs=[task_dropdown, seed_input],
        outputs=all_outputs,
    )


if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
