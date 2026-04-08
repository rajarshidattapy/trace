"""FastAPI server for TRACE environment."""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
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

# Mount Gradio UI alongside FastAPI so both share port 7860 on HF Spaces
try:
    import gradio as gr
    import importlib.util
    _ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui.py")
    if os.path.exists(_ui_path):
        spec = importlib.util.spec_from_file_location("ui", _ui_path)
        ui_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ui_mod)
        app = gr.mount_gradio_app(app, ui_mod.demo, path="/ui")
        print("[INFO] Gradio UI mounted at /ui", flush=True)
except Exception as e:
    print(f"[INFO] Gradio UI not mounted: {e}", flush=True)

# Global environment instance
env = TraceEnv()
current_task_id: str = None


@app.get("/")
async def root():
    """Serve the HTML UI."""
    index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"message": "TRACE v1 API is running. Use /docs for API docs."}


@app.get("/index.html")
async def index():
    """Serve the HTML UI at /index.html."""
    index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="index.html not found")


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