"""Reward calculation engine for TRACE."""

from enum import Enum
from trace.models import Action


class RewardCalculator:
    """Cumulative reward engine (not per-step clamped)."""
    
    # Action rewards (incremental, not clamped per step)
    INSPECTION_REWARD = 1.0
    DIAGNOSIS_REWARD = 0.5  # for relevant inspections
    REMEDIATION_REWARD = 5.0
    DECLARE_HEALTHY_BONUS = 10.0
    DECLARE_UNHEALTHY_PENALTY = -5.0
    
    # Penalties
    DUPLICATE_ACTION_PENALTY = -0.5
    HARMFUL_ACTION_PENALTY = -2.0
    IRRELEVANT_ACTION_PENALTY = -0.1
    
    def __init__(self, task_id: str, root_cause: str):
        self.task_id = task_id
        self.root_cause = root_cause
        self.episode_rewards = []
        self.action_history = []
    
    def calculate_step_reward(
        self,
        action: Action,
        is_relevant: bool,
        solves_issue: bool,
        incident_resolved: bool,
        step: int
    ) -> float:
        """Calculate reward for single step. Does NOT clamp."""
        reward = 0.0
        
        # Check for duplicate recent action
        if step > 0 and self.action_history[-1] == (action.action_type, action.target):
            reward -= self.DUPLICATE_ACTION_PENALTY
        
        # Inspection actions
        if action.action_type.startswith("inspect_"):
            if is_relevant:
                reward += self.INSPECTION_REWARD + self.DIAGNOSIS_REWARD
            else:
                reward -= self.IRRELEVANT_ACTION_PENALTY
        
        # Remediation actions
        elif action.action_type.startswith(("restart_", "scale_", "rollback_", "clear_")):
            if solves_issue:
                reward += self.REMEDIATION_REWARD
            else:
                reward -= self.HARMFUL_ACTION_PENALTY
        
        # Terminal actions
        elif action.action_type == "declare_healthy":
            if incident_resolved:
                reward += self.DECLARE_HEALTHY_BONUS
            else:
                reward += self.DECLARE_UNHEALTHY_PENALTY
        
        elif action.action_type == "declare_unfixable":
            reward -= self.HARMFUL_ACTION_PENALTY
        
        else:
            reward -= self.IRRELEVANT_ACTION_PENALTY
        
        # Record action
        self.action_history.append((action.action_type, action.target))
        self.episode_rewards.append(reward)
        
        return reward
    
    def get_episode_reward(self) -> float:
        """Get cumulative episode reward (not clamped per step)."""
        return sum(self.episode_rewards)
    
    def get_normalized_reward(self, max_possible: float = None) -> float:
        """Normalize episode reward to [0, 1] at end of episode."""
        episode_total = sum(self.episode_rewards)
        
        if max_possible is None:
            # Auto-estimate based on action count
            max_possible = len(self.episode_rewards) * 5.0
        
        if max_possible == 0:
            return 0.0
        
        normalized = episode_total / max_possible
        return min(max(normalized, 0.0), 1.0)
    
    def reset(self):
        """Reset for new episode."""
        self.episode_rewards = []
        self.action_history = []


def get_max_steps(task_id: str) -> int:
    """Get max steps for a task."""
    return {
        "easy_cpu_spike": 5,
        "medium_cascade": 7,
        "hard_mixed": 8,
    }[task_id]


def calculate_final_score(
    incident_resolved: bool,
    steps_taken: int,
    max_steps: int
) -> float:
    """Calculate final outcome-based score (no diagnosis_accuracy)."""
    resolution_score = 1.0 if incident_resolved else 0.0
    efficiency_score = 1.0 - (steps_taken / max_steps) if steps_taken <= max_steps else 0.0
    
    final_score = 0.6 * resolution_score + 0.4 * efficiency_score
    return min(max(final_score, 0.0), 1.0)
