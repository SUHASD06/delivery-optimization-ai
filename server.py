"""
OpenEnv-compliant HTTP server for the Delivery Optimization environment.
Exposes standard endpoints: POST /reset, POST /step, GET /state
This is required by the Scaler Meta PyTorch Hackathon automated evaluator.
"""
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Any, Dict
from env.environment import DeliveryEnv

app = FastAPI(
    title="Delivery Optimization OpenEnv",
    description="OpenEnv-compliant RL environment for last-mile delivery optimization.",
    version="1.0.0",
)

# Global environment instance
env: Optional[DeliveryEnv] = None


# --- Request/Response Schemas ---

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    options: Optional[Dict[str, Any]] = None
    task: Optional[str] = None  # "easy", "medium", "hard"


class StepRequest(BaseModel):
    action: int  # 0=up, 1=down, 2=left, 3=right, 4=refuel


class ResetResponse(BaseModel):
    observation: list
    info: dict


class StepResponse(BaseModel):
    observation: list
    reward: float
    terminated: bool
    truncated: bool
    done: bool
    info: dict


class StateResponse(BaseModel):
    location: list
    fuel: float
    time_elapsed: float
    pending_deliveries: list
    deadlines: list
    steps_taken: int


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


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = None):
    global env

    # Determine phase from task name if provided
    phase = 1
    if request and request.task:
        task_map = {"easy": 1, "medium": 2, "hard": 3}
        phase = task_map.get(request.task, 1)

    env = DeliveryEnv(phase=phase)
    seed = request.seed if request else None
    obs, info = env.reset(seed=seed)

    return ResetResponse(
        observation=obs.tolist(),
        info=info,
    )


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    global env

    if env is None:
        # Auto-initialize if reset was not called
        env = DeliveryEnv(phase=1)
        env.reset()

    obs, reward, terminated, truncated, info = env.step(request.action)

    return StepResponse(
        observation=obs.tolist(),
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        done=bool(terminated or truncated),
        info=info,
    )


@app.get("/state", response_model=StateResponse)
def state():
    global env

    if env is None:
        env = DeliveryEnv(phase=1)
        env.reset()

    s = env.state()
    return StateResponse(
        location=list(s["location"]),
        fuel=float(s["fuel"]),
        time_elapsed=float(s["time"]),
        pending_deliveries=[list(d) for d in s["pending"]],
        deadlines=list(s["deadlines"]),
        steps_taken=int(env.steps_taken),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
