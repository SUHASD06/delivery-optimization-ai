"""
server/app.py — OpenEnv-compliant HTTP server entry point.

Exposes standard endpoints: POST /reset, POST /step, GET /state, GET /health
Runs on port 7860 (shared with Gradio UI in production via app.py at repo root).

The [project.scripts] entry point 'server' calls main() in this module.
"""
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Any, Dict
from env.environment import DeliveryEnv

app = FastAPI(
    title="Delivery Optimization OpenEnv",
    description="OpenEnv-compliant RL environment for last-mile delivery optimization.",
    version="1.0.0",
)

# Global environment instance
_env: Optional[DeliveryEnv] = None


# --- Request/Response Schemas ---

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    options: Optional[Dict[str, Any]] = None
    task: Optional[str] = None  # "easy", "medium", "hard"


class StepRequest(BaseModel):
    action: int  # 0=up, 1=down, 2=left, 3=right, 4=refuel


# --- Endpoints ---

@app.get("/")
def root():
    return {
        "name": "delivery-optimization-env",
        "description": "OpenEnv-compliant delivery optimization RL environment",
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(request: ResetRequest = None):
    global _env

    # Determine phase from task name if provided
    phase = 1
    if request and request.task:
        task_map = {"easy": 1, "medium": 2, "hard": 3}
        phase = task_map.get(request.task, 1)

    _env = DeliveryEnv(phase=phase)
    seed = request.seed if request else None
    obs, info = _env.reset(seed=seed)

    return JSONResponse(content={
        "observation": obs.tolist(),
        "info": info,
    })


@app.post("/step")
def step(request: StepRequest):
    global _env

    if _env is None:
        _env = DeliveryEnv(phase=1)
        _env.reset()

    obs, reward, terminated, truncated, info = _env.step(request.action)

    return JSONResponse(content={
        "observation": obs.tolist(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "done": bool(terminated or truncated),
        "info": info,
    })


@app.get("/state")
def state():
    global _env

    if _env is None:
        _env = DeliveryEnv(phase=1)
        _env.reset()

    s = _env.state()
    return JSONResponse(content={
        "location": list(s["location"]),
        "fuel": float(s["fuel"]),
        "time_elapsed": float(s["time"]),
        "pending_deliveries": [list(d) for d in s["pending"]],
        "deadlines": list(s["deadlines"]),
        "steps_taken": int(_env.steps_taken),
    })


def main():
    """Entry point used by [project.scripts] server = 'server.app:main'."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
