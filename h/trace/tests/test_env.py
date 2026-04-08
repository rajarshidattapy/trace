"""Test main TraceEnv interface."""

import pytest
from trace.env import TraceEnv
from trace.models import Action, Observation


def test_env_reset():
    """Test environment reset."""
    env = TraceEnv()
    obs = env.reset(task_id="easy_cpu_spike", seed=42)
    
    assert isinstance(obs, Observation)
    assert obs.cpu_usage_pct >= 0
    assert obs.memory_usage_pct >= 0


def test_env_step():
    """Test environment step."""
    env = TraceEnv()
    env.reset(task_id="easy_cpu_spike", seed=42)
    
    action = Action(
        action_type="scale_workers",
        target="api_workers",
        value=5
    )
    
    obs, reward, done, info = env.step(action)
    
    assert isinstance(obs, Observation)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert "step" in info


def test_env_state():
    """Test environment state query."""
    env = TraceEnv()
    env.reset(task_id="easy_cpu_spike", seed=42)
    
    state = env.state()
    
    assert "observation" in state
    assert "episode_reward" in state
    assert "steps" in state
    assert "done" in state


def test_env_episode_progression():
    """Test that episode progresses correctly."""
    env = TraceEnv()
    env.reset(task_id="easy_cpu_spike", seed=42)
    
    initial_reward = env.episode_reward
    assert initial_reward == 0.0
    
    # Take action
    action = Action(
        action_type="inspect_logs",
        target="api_workers",
        value=None
    )
    
    obs, reward, done, info = env.step(action)
    
    # Reward should accumulate
    assert env.episode_reward >= initial_reward
    assert env.steps == 1
