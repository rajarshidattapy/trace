"""Grading engine for TRACE environment."""

from dataclasses import dataclass


@dataclass
class Grade:
    """Episode grade."""
    score: float
    success: bool
    efficiency: float
    steps_taken: int
    max_steps: int


class Grader:
    """Deterministic grader (outcome-based, no diagnosis_accuracy)."""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.max_steps = {
            "easy_cpu_spike": 5,
            "medium_cascade": 7,
            "hard_mixed": 8,
        }[task_id]
    
    def grade(
        self,
        incident_resolved: bool,
        steps_taken: int
    ) -> Grade:
        """
        Grade an episode (more lenient).
        
        Final score = 0.5 * resolution_success + 0.5 * efficiency
        where:
        - resolution_success is 1.0 if incident_resolved else 0.0
        - efficiency is max(0, 1.0 - (steps_taken / (max_steps * 1.5)))
          (more forgiving for going over max_steps)
        """
        resolution_score = 1.0 if incident_resolved else 0.0
        
        # Efficiency: more lenient, doesn't hit 0 until 1.5x max_steps
        efficiency = max(0.0, 1.0 - (steps_taken / (self.max_steps * 1.5)))
        
        # More balanced weighting: 50% resolution, 50% efficiency
        final_score = 0.5 * resolution_score + 0.5 * efficiency
        final_score = min(max(final_score, 0.0), 1.0)
        
        return Grade(
            score=final_score,
            success=incident_resolved,
            efficiency=efficiency,
            steps_taken=steps_taken,
            max_steps=self.max_steps
        )
