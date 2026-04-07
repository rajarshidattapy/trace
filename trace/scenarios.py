"""Scenario generators for TRACE environment."""

import random
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ScenarioState:
    """Mutable scenario-specific state."""
    
    task_id: str
    seed: int
    step: int = 0
    
    # Scenario-specific counters/flags
    traffic_spike_strength: float = 0
    queue_memory_leak: float = 0
    db_release_impact: float = 0
    
    def reset(self):
        """Reset step counter and RNG."""
        self.step = 0
        random.seed(self.seed)


class ScenarioEasyCPUSpike:
    """Easy scenario: traffic spike requiring horizontal scaling."""
    
    MAX_STEPS = 5
    
    def __init__(self, seed: int = 0):
        self.state = ScenarioState(
            task_id="easy_cpu_spike",
            seed=seed,
            traffic_spike_strength=0.8  # 80% spike at start
        )
    
    def reset(self):
        self.state.reset()
    
    def step(self, action_taken: bool = False) -> dict:
        """Progress scenario. Return metrics."""
        self.state.step += 1
        rng = random.Random(self.seed + self.state.step)
        
        # If fix applied (scale_workers), decay traffic
        if action_taken:
            self.state.traffic_spike_strength *= 0.5
        else:
            # Spike persists but doesn't worsen
            pass
        
        cpu_usage = 30 + self.state.traffic_spike_strength * 60 + rng.gauss(0, 2)
        error_rate = 2 + self.state.traffic_spike_strength * 10
        latency = 50 + self.state.traffic_spike_strength * 450
        
        return {
            "cpu_usage_pct": min(100, max(0, cpu_usage)),
            "error_rate_pct": min(100, max(0, error_rate)),
            "api_latency_ms": max(50, latency),
            "queue_depth": 10,
            "memory_usage_pct": 40 + rng.gauss(0, 2),
        }
    
    def is_resolved(self) -> bool:
        """Check if incident is resolved."""
        return self.state.traffic_spike_strength < 0.1


class ScenarioMediumCascade:
    """Medium scenario: queue memory leak causing cascading failures."""
    
    MAX_STEPS = 7
    
    def __init__(self, seed: int = 0):
        self.state = ScenarioState(
            task_id="medium_cascade",
            seed=seed,
            queue_memory_leak=0.1  # starts small, grows
        )
    
    def reset(self):
        self.state.reset()
    
    def step(self, action_taken: bool = False) -> dict:
        """Progress scenario. Return metrics."""
        self.state.step += 1
        rng = random.Random(self.seed + self.state.step)
        
        # If fix applied (restart_service), reset memory leak
        if action_taken:
            self.state.queue_memory_leak = 0
        else:
            # Leak grows exponentially
            self.state.queue_memory_leak = min(1.0, self.state.queue_memory_leak * 1.3)
        
        queue_depth = 100 + self.state.queue_memory_leak * 900
        error_rate = 1 + self.state.queue_memory_leak * 20
        
        return {
            "cpu_usage_pct": 45 + self.state.queue_memory_leak * 20 + rng.gauss(0, 2),
            "error_rate_pct": min(100, error_rate),
            "api_latency_ms": 100 + self.state.queue_memory_leak * 1000,
            "queue_depth": int(queue_depth),
            "memory_usage_pct": 50 + self.state.queue_memory_leak * 30,
        }
    
    def is_resolved(self) -> bool:
        """Check if incident is resolved."""
        return self.state.queue_memory_leak < 0.05


class ScenarioHardMixed:
    """Hard scenario: DB connection pool + release regression."""
    
    MAX_STEPS = 8
    
    def __init__(self, seed: int = 0):
        self.state = ScenarioState(
            task_id="hard_mixed",
            seed=seed,
            db_release_impact=0.7  # starts high
        )
    
    def reset(self):
        self.state.reset()
    
    def step(self, action_taken: bool = False) -> dict:
        """Progress scenario. Return metrics."""
        self.state.step += 1
        rng = random.Random(self.seed + self.state.step)
        
        # If fix applied (restart_database), decay impact
        if action_taken:
            self.state.db_release_impact *= 0.4
        else:
            # Impact persists
            pass
        
        error_rate = 5 + self.state.db_release_impact * 25
        latency = 100 + self.state.db_release_impact * 1900
        
        return {
            "cpu_usage_pct": 55 + self.state.db_release_impact * 30 + rng.gauss(0, 3),
            "error_rate_pct": min(100, error_rate),
            "api_latency_ms": max(100, latency),
            "queue_depth": 50 + int(self.state.db_release_impact * 200),
            "memory_usage_pct": 60 + self.state.db_release_impact * 20,
        }
    
    def is_resolved(self) -> bool:
        """Check if incident is resolved."""
        return self.state.db_release_impact < 0.15


def create_scenario(task_id: str, seed: int = 0):
    """Factory for scenario objects."""
    if task_id == "easy_cpu_spike":
        return ScenarioEasyCPUSpike(seed)
    elif task_id == "medium_cascade":
        return ScenarioMediumCascade(seed)
    elif task_id == "hard_mixed":
        return ScenarioHardMixed(seed)
    else:
        raise ValueError(f"Unknown task_id: {task_id}")
