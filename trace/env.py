"""Main TraceEnv gym-like environment."""

from typing import Optional, Tuple
from trace.models import Observation, Action, StepResponse
from trace.simulator import Simulator
from trace.rewards import RewardCalculator, get_max_steps
from trace.graders import Grader


class TraceEnv:
    """OpenEnv-compatible incident response environment."""
    
    def __init__(self):
        self.current_task_id: Optional[str] = None
        self.simulator: Optional[Simulator] = None
        self.reward_calc: Optional[RewardCalculator] = None
        self.grader: Optional[Grader] = None
        self.episode_reward: float = 0.0
        self.steps: int = 0
        self.done: bool = False
    
    def reset(self, task_id: str = "easy_cpu_spike", seed: int = 0) -> Observation:
        """Reset environment for new episode."""
        self.current_task_id = task_id
        self.simulator = Simulator(task_id, seed)
        self.reward_calc = RewardCalculator(task_id, root_cause=self._get_root_cause())
        self.grader = Grader(task_id)
        self.episode_reward = 0.0
        self.steps = 0
        self.done = False
        
        obs = self.simulator.reset()
        return obs
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """Execute one step in the environment."""
        if self.done:
            raise RuntimeError("Episode already done. Call reset() first.")
        
        if self.simulator is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        # Progress simulator
        obs, simulator_reward, sim_done, sim_info = self.simulator.step(action)
        
        # Calculate step reward
        is_relevant = self.simulator._is_action_relevant(action)
        solves_issue = self.simulator._does_action_solve(action)
        incident_resolved = self.simulator.scenario.is_resolved()
        
        step_reward = self.reward_calc.calculate_step_reward(
            action=action,
            is_relevant=is_relevant,
            solves_issue=solves_issue,
            incident_resolved=incident_resolved,
            step=self.steps
        )
        
        self.episode_reward += step_reward
        self.steps += 1
        self.done = sim_done or self.steps >= get_max_steps(self.current_task_id)
        
        info = {
            "step": self.steps,
            "episode_reward": self.episode_reward,
            "step_reward": step_reward,
            **sim_info
        }
        
        if self.done:
            grade = self.grader.grade(
                incident_resolved=sim_info.get("is_resolved", False),
                steps_taken=self.steps
            )
            info["final_grade"] = grade.score
            info["success"] = grade.success
            info["efficiency"] = grade.efficiency
        
        return obs, step_reward, self.done, info
    
    def state(self) -> dict:
        """Get current state without progressing."""
        if self.simulator is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        return {
            "observation": self.simulator.current_obs,
            "episode_reward": self.episode_reward,
            "steps": self.steps,
            "done": self.done
        }
    
    def _get_root_cause(self) -> str:
        """Get root cause for current task."""
        if self.current_task_id == "easy_cpu_spike":
            return "traffic_spike"
        elif self.current_task_id == "medium_cascade":
            return "queue_memory_leak"
        elif self.current_task_id == "hard_mixed":
            return "db_connection_pool + release_regression"
        return "unknown"
