"""
TRACE v1 — Gradio Demo UI
==========================
Interactive UI that lets you play through TRACE scenarios step-by-step.
No need to run the FastAPI server separately — this embeds TraceEnv directly.

Usage:
    pip install gradio
    python ui.py
"""

import gradio as gr
import json
from trace.env import TraceEnv
from trace.models import Action

# ── Global state ─────────────────────────────────────────────────────────────

env = TraceEnv()
episode_log = []
current_obs = None


def format_metrics(obs):
    """Format observation metrics as readable markdown."""
    if obs is None:
        return "No observation yet. Reset the environment first."
    
    d = obs.model_dump() if hasattr(obs, 'model_dump') else obs
    
    def bar(val, max_val=100, width=10):
        filled = int((val / max_val) * width)
        return "█" * filled + "░" * (width - filled)
    
    def status_color(val, warn, crit):
        if val > crit:
            return "🔴"
        elif val > warn:
            return "🟡"
        return "🟢"
    
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
        return {"healthy": "🟢", "degraded": "🟡", "down": "🔴"}.get(s, "⚪")
    
    lines = []
    for k, v in d.get("services", {}).items():
        lines.append(f"{svc_emoji(v)} **{k}** — {v}")
    return "\n\n".join(lines) if lines else "_No services_"


def format_alerts(obs):
    """Format active alerts."""
    if obs is None:
        return ""
    d = obs.model_dump() if hasattr(obs, 'model_dump') else obs
    alerts = d.get("active_alerts", [])
    if not alerts:
        return "✅ No active alerts"
    return "\n\n".join(f"🚨 `{a}`" for a in alerts)


def format_inspection(obs):
    """Format last inspection."""
    if obs is None:
        return "_No inspection yet_"
    d = obs.model_dump() if hasattr(obs, 'model_dump') else obs
    inspection = d.get("last_inspection")
    if inspection and inspection.get("message"):
        return f'💡 **{inspection["message"]}**'
    return "_No inspection results — use inspect_logs, inspect_metrics, or inspect_alert_"


def format_episode_log():
    """Format episode log as markdown table."""
    if not episode_log:
        return "_No actions taken yet. Reset the environment and start taking actions._"
    
    rows = ["| # | Action | Reward | Status |", "|---|--------|--------|--------|"]
    for entry in episode_log:
        emoji = "✅" if entry.get("reward", 0) > 0 else "❌" if entry.get("reward", 0) < 0 else "➖"
        status = "🏁 Done" if entry.get("done") else "▶️"
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
    
    max_steps = {"easy_cpu_spike": 5, "medium_cascade": 7, "hard_mixed": 8}[task_id]
    
    status_msg = f"✅ **Reset** — `{task_id}` | Seed: {seed_int} | Max steps: {max_steps}"
    
    return (
        format_metrics(obs),
        format_services(obs),
        format_alerts(obs),
        format_inspection(obs),
        status_msg,
        format_episode_log(),
        format_score_bar(),
        "",
        gr.update(interactive=True),
    )


def take_action(action_type, target, value):
    """Execute one step."""
    global current_obs
    
    if env.simulator is None:
        return (
            "⚠️ **Reset the environment first!**",
            "", "", "", "⚠️ Reset required",
            format_episode_log(), "", "",
            gr.update(),
        )
    
    if env.done:
        return (
            format_metrics(current_obs),
            format_services(current_obs),
            format_alerts(current_obs),
            format_inspection(current_obs),
            "🏁 **Episode over.** Reset to play again.",
            format_episode_log(),
            format_score_bar(),
            format_grade(),
            gr.update(interactive=False),
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
    action_str = f"{action_type}({tgt or ''}, {val or ''})"
    episode_log.append({
        "step": info.get("step", len(episode_log) + 1),
        "action": action_str,
        "reward": reward,
        "done": done,
    })
    
    # Status
    if done:
        if info.get("success"):
            status = f"🎉 **Resolved!** Grade: {info.get('final_grade', 0):.3f} | Efficiency: {info.get('efficiency', 0):.2f}"
        else:
            status = f"💀 **Failed** — Grade: {info.get('final_grade', 0):.3f}"
    else:
        status = f"▶️ Step {info.get('step', '?')} — Reward: {reward:+.1f} | Resolved: {'✅' if info.get('is_resolved') else '❌'}"
    
    grade_md = format_grade() if done else ""
    
    return (
        format_metrics(obs),
        format_services(obs),
        format_alerts(obs),
        format_inspection(obs),
        status,
        format_episode_log(),
        format_score_bar(),
        grade_md,
        gr.update(interactive=not done),
    )


def format_grade():
    """Format final grade if episode is done."""
    if not env.done:
        return ""
    
    state = env.state()
    total_reward = state.get("episode_reward", 0)
    steps = state.get("steps", 0)
    
    return f"🏆 **Total Reward:** {total_reward:.2f} | **Steps:** {steps}"


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
"""

with gr.Blocks(title="TRACE v1 — Incident Response Demo") as demo:
    
    # ── Header ──
    gr.Markdown("""
# 🔍 TRACE v1 — Incident Response Environment
_Observe → Inspect → Diagnose → Remediate → Validate_
    """)
    
    # ── Row 1: Status bar ──
    status_display = gr.Markdown("👋 Select a scenario and click **Reset** to begin.")
    
    # ── Row 2: Main 3-column layout ──
    with gr.Row(equal_height=True):
        
        # ── Column 1: Controls ──
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### ⚙️ Controls")
            task_dropdown = gr.Dropdown(
                choices=["easy_cpu_spike", "medium_cascade", "hard_mixed"],
                value="easy_cpu_spike",
                label="Scenario"
            )
            seed_input = gr.Textbox(value="42", label="Seed")
            reset_btn = gr.Button("🔄 Reset Environment", variant="primary", elem_classes=["action-btn"])
            
            gr.Markdown("---")
            gr.Markdown("### 🎮 Action")
            action_dropdown = gr.Dropdown(
                choices=[
                    "inspect_logs", "inspect_metrics", "inspect_alert",
                    "restart_service", "scale_workers", "restart_database",
                    "rollback_release", "clear_queue",
                    "declare_healthy", "declare_unfixable",
                ],
                value="inspect_logs",
                label="Action Type"
            )
            target_input = gr.Textbox(value="api_workers", label="Target")
            value_input = gr.Textbox(value="", label="Value", placeholder="e.g. 5")
            action_btn = gr.Button("⚡ Execute", variant="secondary", elem_classes=["action-btn"])
        
        # ── Column 2: Metrics Dashboard ──
        with gr.Column(scale=2, min_width=400):
            gr.Markdown("### 📊 System Metrics")
            metrics_display = gr.Markdown("_Reset to see metrics_", elem_classes=["metric-panel"])
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 🖥️ Services")
                    services_display = gr.Markdown("")
                with gr.Column(scale=1):
                    gr.Markdown("#### 🚨 Alerts")
                    alerts_display = gr.Markdown("")
            
            gr.Markdown("#### 🔍 Inspection Result")
            inspection_display = gr.Markdown("_No inspection yet_")
        
        # ── Column 3: Episode Log ──
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### 📜 Episode Log")
            score_display = gr.Markdown("")
            log_display = gr.Markdown("_No actions yet_")
            grade_display = gr.Markdown("")
            
            gr.Markdown("---")
            gr.Markdown("""#### 💡 Hints
- **Easy**: `inspect_logs(api_workers)` → `scale_workers(api_workers, 5)` → `declare_healthy`
- **Medium**: `inspect_metrics(queue_depth)` → `inspect_logs(queue_service)` → `restart_service(queue_service)` → `declare_healthy`
- **Hard**: `inspect_alert(alert_pool_exhaustion)` → `inspect_logs(database)` → `restart_database` → `restart_database` → `declare_healthy`
            """)
    
    # ── Event handlers ──
    reset_btn.click(
        fn=reset_env,
        inputs=[task_dropdown, seed_input],
        outputs=[
            metrics_display, services_display, alerts_display,
            inspection_display, status_display, log_display,
            score_display, grade_display, action_btn
        ],
    )
    
    action_btn.click(
        fn=take_action,
        inputs=[action_dropdown, target_input, value_input],
        outputs=[
            metrics_display, services_display, alerts_display,
            inspection_display, status_display, log_display,
            score_display, grade_display, action_btn
        ],
    )


if __name__ == "__main__":
    demo.launch(server_port=7861, share=True, theme=THEME, css=CSS)
