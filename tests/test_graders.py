"""Test grading engine."""

import pytest
from trace.graders import Grader


def test_grader_success_max_efficiency():
    """Test grader with success and max efficiency."""
    grader = Grader("easy_cpu_spike")
    
    grade = grader.grade(
        incident_resolved=True,
        steps_taken=1  # Very efficient
    )
    
    assert grade.score > 0.9, "Should be high score"
    assert grade.success is True
    assert grade.efficiency > 0.8


def test_grader_success_poor_efficiency():
    """Test grader with success but poor efficiency."""
    grader = Grader("easy_cpu_spike")
    
    grade = grader.grade(
        incident_resolved=True,
        steps_taken=5  # At max
    )
    
    assert grade.score == 0.6, f"Expected 0.6, got {grade.score}"
    assert grade.success is True
    assert grade.efficiency == 0.0


def test_grader_failure():
    """Test grader when incident not resolved."""
    grader = Grader("hard_mixed")
    
    grade = grader.grade(
        incident_resolved=False,
        steps_taken=3
    )
    
    assert grade.score == 0.0, "Should be 0 if not resolved"
    assert grade.success is False


def test_grader_tasks_have_different_max_steps():
    """Test that tasks have correct max_steps."""
    graders = {
        "easy_cpu_spike": Grader("easy_cpu_spike"),
        "medium_cascade": Grader("medium_cascade"),
        "hard_mixed": Grader("hard_mixed"),
    }
    
    assert graders["easy_cpu_spike"].max_steps == 5
    assert graders["medium_cascade"].max_steps == 7
    assert graders["hard_mixed"].max_steps == 8
