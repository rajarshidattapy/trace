"""Test scenario determinism."""

import pytest
from trace.scenarios import create_scenario


def test_easy_cpu_spike_determinism():
    """Test that easy_cpu_spike is deterministic with same seed."""

    # Run scenario 1 with seed=42
    scenario1 = create_scenario("easy_cpu_spike", seed=42)
    scenario1.reset()
    metrics1 = []
    for _ in range(5):
        m = scenario1.step()
        metrics1.append(m)

    # Run scenario 2 with same seed=42
    scenario2 = create_scenario("easy_cpu_spike", seed=42)
    scenario2.reset()
    metrics2 = []
    for _ in range(5):
        m = scenario2.step()
        metrics2.append(m)

    # Should be identical
    for m1, m2 in zip(metrics1, metrics2):
        assert abs(m1["cpu_usage_pct"] - m2["cpu_usage_pct"]) < 0.01
        assert m1["api_latency_ms"] == m2["api_latency_ms"]


def test_medium_cascade_resolution():
    """Test that cascade scenario resolves when action is taken."""
    scenario = create_scenario("medium_cascade", seed=42)
    scenario.reset()

    # Initially not resolved
    assert not scenario.is_resolved()

    # Simulate without fix
    for _ in range(5):
        scenario.step()

    # Still degraded
    assert not scenario.is_resolved()

    # Fix (restart_service)
    scenario.step(resolved_action="restart_service")

    # Should improve
    assert scenario.is_resolved()


def test_hard_mixed_multi_step():
    """Test that hard scenario requires both restart_database and rollback_release."""
    scenario = create_scenario("hard_mixed", seed=42)
    scenario.reset()

    # Take multiple steps without actions — should stay unresolved
    for _ in range(5):
        scenario.step()
        assert not scenario.is_resolved()

    # Only restart_database — db_pool fixed but release still high
    scenario.step(resolved_action="restart_database")
    assert not scenario.is_resolved(), "restart_database alone should not resolve hard scenario"

    # Now rollback_release — both factors resolved
    scenario.step(resolved_action="rollback_release")
    assert scenario.is_resolved(), "Both restart_database + rollback_release should resolve hard scenario"
