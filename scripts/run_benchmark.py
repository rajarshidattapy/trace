"""
TRACE v1 — Local Benchmark Runner
==================================
Runs all 3 scenarios with a deterministic heuristic agent.
Prints per-scenario scores and overall average.

Usage:
    python scripts/run_benchmark.py
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trace.env import TraceEnv
from trace.models import Action


# ── Heuristic agents ─────────────────────────────────────────────────────────

def heuristic_easy_cpu_spike(env: TraceEnv) -> dict:
    """Optimal heuristic for easy_cpu_spike scenario."""
    obs = env.reset(task_id="easy_cpu_spike", seed=42)
    total_reward = 0.0
    steps = 0

    actions = [
        Action(action_type="inspect_logs", target="api_workers", value=None),
        Action(action_type="scale_workers", target="api_workers", value=5),
        Action(action_type="scale_workers", target="api_workers", value=5),
        Action(action_type="declare_healthy", target=None, value=None),
    ]

    for action in actions:
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break

    return {
        "task_id": "easy_cpu_spike",
        "steps": steps,
        "total_reward": total_reward,
        "done": done,
        "info": info,
    }


def heuristic_medium_cascade(env: TraceEnv) -> dict:
    """Optimal heuristic for medium_cascade scenario."""
    obs = env.reset(task_id="medium_cascade", seed=42)
    total_reward = 0.0
    steps = 0

    actions = [
        Action(action_type="inspect_metrics", target="queue_depth", value=None),
        Action(action_type="inspect_logs", target="queue_service", value=None),
        Action(action_type="restart_service", target="queue_service", value=None),
        Action(action_type="declare_healthy", target=None, value=None),
    ]

    for action in actions:
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break

    return {
        "task_id": "medium_cascade",
        "steps": steps,
        "total_reward": total_reward,
        "done": done,
        "info": info,
    }


def heuristic_hard_mixed(env: TraceEnv) -> dict:
    """Optimal heuristic for hard_mixed scenario."""
    obs = env.reset(task_id="hard_mixed", seed=42)
    total_reward = 0.0
    steps = 0

    actions = [
        Action(action_type="inspect_alert", target="alert_pool_exhaustion", value=None),
        Action(action_type="inspect_logs", target="database", value=None),
        Action(action_type="inspect_metrics", target="db_connections", value=None),
        Action(action_type="restart_database", target=None, value=None),
        Action(action_type="restart_database", target=None, value=None),
        Action(action_type="declare_healthy", target=None, value=None),
    ]

    for action in actions:
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break

    return {
        "task_id": "hard_mixed",
        "steps": steps,
        "total_reward": total_reward,
        "done": done,
        "info": info,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    env = TraceEnv()

    print("=" * 60)
    print("  TRACE v1 — Local Benchmark")
    print("=" * 60)
    print()

    scenarios = [
        ("easy_cpu_spike", heuristic_easy_cpu_spike),
        ("medium_cascade", heuristic_medium_cascade),
        ("hard_mixed", heuristic_hard_mixed),
    ]

    results = []
    for name, heuristic_fn in scenarios:
        result = heuristic_fn(env)
        results.append(result)

        final_grade = result["info"].get("final_grade", "N/A")
        success = result["info"].get("success", "N/A")
        efficiency = result["info"].get("efficiency", "N/A")

        print(f"  Scenario: {name}")
        print(f"    Steps taken:   {result['steps']}")
        print(f"    Total reward:  {result['total_reward']:.2f}")
        print(f"    Final grade:   {final_grade}")
        print(f"    Success:       {success}")
        print(f"    Efficiency:    {efficiency}")
        print(f"    Episode done:  {result['done']}")
        print()

    # Overall summary
    grades = [r["info"].get("final_grade", 0) for r in results]
    valid_grades = [g for g in grades if isinstance(g, (int, float))]
    avg_grade = sum(valid_grades) / len(valid_grades) if valid_grades else 0

    print("-" * 60)
    print(f"  Average Grade: {avg_grade:.3f}")
    print(f"  Scenarios Passed: {sum(1 for r in results if r['info'].get('success', False))}/{len(results)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
