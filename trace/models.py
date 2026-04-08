"""Pydantic schemas for TRACE environment."""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Current state observation visible to agent."""
    
    timestamp: str = Field(description="ISO8601 timestamp")
    
    # Metrics (always visible)
    cpu_usage_pct: float = Field(ge=0, le=100, description="CPU usage [0, 100]")
    memory_usage_pct: float = Field(ge=0, le=100, description="Memory usage [0, 100]")
    error_rate_pct: float = Field(ge=0, le=100, description="Error rate [0, 100]")
    api_latency_ms: float = Field(ge=0, description="API latency in ms")
    queue_depth: int = Field(ge=0, description="Queue backlog depth")
    
    # Service status (always visible)
    services: dict[str, str] = Field(default_factory=dict, description="Service health: healthy|degraded|down")
    
    # Alerts (names only, no context)
    active_alerts: list[str] = Field(default_factory=list, description="Active alert IDs")
    
    # Inspection results (populated by inspect_* actions)
    last_inspection: Optional[dict] = Field(default=None, description="Last inspection result")


class Action(BaseModel):
    """Agent action (type, target, value)."""
    
    action_type: str = Field(description="Action type: inspect_logs, inspect_metrics, etc.")
    target: Optional[str] = Field(default=None, description="Target service/metric/alert")
    value: Optional[float] = Field(default=None, description="Numeric value (scale factor, count, etc.)")


class StepResponse(BaseModel):
    """Response from /step endpoint."""
    
    observation: Observation
    reward: float = Field(description="Reward for this step")
    done: bool = Field(description="Episode terminal flag")
    info: dict = Field(default_factory=dict)


class ResetRequest(BaseModel):
    """Request to /reset endpoint."""
    
    task_id: Literal["easy_cpu_spike", "medium_cascade", "hard_mixed"] = Field(default="easy_cpu_spike")
    seed: int = Field(default=0)


class StateResponse(BaseModel):
    """Response from /state endpoint."""
    
    observation: Observation
    episode_reward: float
    steps: int
    done: bool


class HealthResponse(BaseModel):
    """Response from /health endpoint."""
    
    status: str = "healthy"
    version: str = "0.1.0"
