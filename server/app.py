"""FastAPI server for TRACE environment."""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from trace.env import TraceEnv
from trace.utils import generate_episode_id
from trace.models import (
    Observation, Action, StepResponse, ResetRequest,
    StateResponse, HealthResponse
)

app = FastAPI(
    title="TRACE",
    version="0.1.0",
    description="OpenEnv-compatible incident response environment"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env = TraceEnv()
current_task_id: str = None


@app.post("/reset")
async def reset_endpoint(request: ResetRequest) -> dict:
    """Reset environment for new episode."""
    global current_task_id
    
    try:
        obs = env.reset(task_id=request.task_id, seed=request.seed)
        current_task_id = request.task_id
        
        return {
            "observation": obs.model_dump(),
            "info": {
                "task_id": request.task_id,
                "episode_id": generate_episode_id(),
                "max_steps": {
                    "easy_cpu_spike": 5,
                    "medium_cascade": 7,
                    "hard_mixed": 8,
                }[request.task_id],
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step_endpoint(request: dict) -> dict:
    """Execute one step."""
    try:
        action_data = request.get("action", {})
        action = Action(
            action_type=action_data.get("action_type"),
            target=action_data.get("target"),
            value=action_data.get("value")
        )
        
        obs, reward, done, info = env.step(action)
        
        return {
            "observation": obs.model_dump(),
            "reward": float(reward),
            "done": done,
            "info": {
                **info,
                "message": f"Action {action.action_type} executed"
            }
        }
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
async def state_endpoint() -> StateResponse:
    """Get current state."""
    try:
        state = env.state()
        return StateResponse(
            observation=state["observation"],
            episode_reward=state["episode_reward"],
            steps=state["steps"],
            done=state["done"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health_endpoint() -> HealthResponse:
    """Health check."""
    return HealthResponse(
        status="healthy",
        version="0.1.0"
    )


def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()