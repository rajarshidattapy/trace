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
        m = scenario1.step(action_taken=False)
        metrics1.append(m)
    
    # Run scenario 2 with same seed=42
    scenario2 = create_scenario("easy_cpu_spike", seed=42)
    scenario2.reset()
    metrics2 = []
    for _ in range(5):
        m = scenario2.step(action_taken=False)
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
    
    # Simulate fix (scale up queue_memory_leak)
    for _ in range(5):
        scenario.step(action_taken=False)
    
    # Still degraded
    assert not scenario.is_resolved()
    
    # Fix (restart)
    scenario.step(action_taken=True)
    
    # Should improve
    assert scenario.is_resolved()


def test_hard_mixed_multi_step():
    """Test that hard scenario requires multiple remediation steps."""
    scenario = create_scenario("hard_mixed", seed=42)
    scenario.reset()
    
    # Take multiple steps without actions — should stay unresolved
    for _ in range(5):
        scenario.step(action_taken=False)
        assert not scenario.is_resolved()
    
    # First remediation: 0.7 * 0.4 = 0.28 (still above 0.15 threshold)
    scenario.step(action_taken=True)
    assert not scenario.is_resolved(), "One action should not be enough for hard scenario"
    
    # Second remediation: 0.28 * 0.4 = 0.112 (below 0.15 threshold)
    scenario.step(action_taken=True)
    assert scenario.is_resolved(), "Two remediation actions should resolve hard scenario"
