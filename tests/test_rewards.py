"""Test reward calculation (cumulative, not per-step clamped)."""

import pytest
from trace.rewards import RewardCalculator, calculate_final_score
from trace.models import Action


def test_cumulative_rewards_not_clamped():
    """Test that rewards accumulate and are not clamped per step."""
    calc = RewardCalculator("easy_cpu_spike", root_cause="traffic_spike")
    
    # Give rewards that normally would be clamped
    calc.episode_rewards = [5.0, 5.0, -2.0, 10.0, 5.0]
    
    # Sum should not be clamped (should be 23.0)
    total = calc.get_episode_reward()
    assert total == 23.0, f"Expected 23.0, got {total}"
    
    # Only at the end is normalization applied
    normalized = calc.get_normalized_reward(max_possible=50)
    assert 0 <= normalized <= 1, "Normalized should be in [0,1]"


def test_inspection_rewards():
    """Test that inspection actions give rewards."""
    calc = RewardCalculator("easy_cpu_spike", root_cause="traffic_spike")
    
    action = Action(action_type="inspect_logs", target="api_workers", value=None)
    reward = calc.calculate_step_reward(
        action=action,
        is_relevant=True,
        solves_issue=False,
        incident_resolved=False,
        step=0
    )
    
    assert reward > 0, "Inspection should give positive reward"


def test_duplicate_action_penalty():
    """Test that duplicate recent actions get penalized."""
    calc = RewardCalculator("easy_cpu_spike", root_cause="traffic_spike")
    
    action = Action(action_type="inspect_logs", target="api_workers", value=None)
    
    # First action
    first_reward = calc.calculate_step_reward(
        action=action,
        is_relevant=True,
        solves_issue=False,
        incident_resolved=False,
        step=0
    )
    
    # Duplicate action (same type+target consecutively)
    dup_reward = calc.calculate_step_reward(
        action=action,
        is_relevant=True,
        solves_issue=False,
        incident_resolved=False,
        step=1
    )
    
    # Duplicate should get less reward than non-duplicate due to penalty
    assert dup_reward < first_reward, "Duplicate should receive less reward than first action"


def test_final_score_calculation():
    """Test outcome-based scoring (no diagnosis_accuracy)."""
    
    # Success in max steps
    score = calculate_final_score(
        incident_resolved=True,
        steps_taken=5,
        max_steps=5
    )
    assert score == 0.6, f"Expected 0.6, got {score}"
    
    # Success in half steps
    score = calculate_final_score(
        incident_resolved=True,
        steps_taken=2,
        max_steps=5
    )
    expected = 0.6 * 1.0 + 0.4 * (1.0 - 2/5)
    assert abs(score - expected) < 0.01, f"Expected {expected}, got {score}"
    
    # Failure
    score = calculate_final_score(
        incident_resolved=False,
        steps_taken=5,
        max_steps=5
    )
    assert score == 0.0, f"Expected 0.0, got {score}"
