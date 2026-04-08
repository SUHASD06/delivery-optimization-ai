"""
inference.py — OpenEnv Hackathon Inference Script.

Connects to the running environment server (or falls back to running
the environment in-process), runs an agent for one episode, and prints
[START]/[STEP]/[END] structured output blocks required by the Phase 2 evaluator.
"""
import os
import sys

# ── Task / config ──────────────────────────────────────────────
task_name = os.getenv("OPENENV_TASK", "delivery-optimization")
base_url   = os.getenv("OPENENV_URL", "http://localhost:7860")
max_steps  = 200

# Simple cycling action pattern: up, right, down, left, refuel
ACTIONS = [0, 3, 1, 2, 4]

# ── [START] must be the very first line of structured output ───
print(f"[START] task={task_name} step=0 reward=0.0", flush=True)


def run_via_http():
    """Try to run one episode through the HTTP server."""
    import requests

    reset_resp = requests.post(f"{base_url}/reset", json={}, timeout=30)
    reset_resp.raise_for_status()

    total_reward = 0.0
    done = False
    step = 0

    while not done and step < max_steps:
        action = ACTIONS[step % len(ACTIONS)]
        step_resp = requests.post(
            f"{base_url}/step",
            json={"action": action},
            timeout=30,
        )
        step_resp.raise_for_status()
        res = step_resp.json()

        reward = float(res.get("reward", 0.0))
        total_reward += reward
        done = bool(res.get("done") or res.get("terminated") or res.get("truncated"))
        step += 1

        print(f"[STEP] task={task_name} step={step} reward={reward:.4f}", flush=True)

    return total_reward, step


def run_locally():
    """Fallback: run the environment in-process without a server."""
    sys.path.insert(0, os.path.dirname(__file__))
    from env.environment import DeliveryEnv
    from agent.baseline import choose_best

    env = DeliveryEnv(phase=1)
    env.reset(seed=42)

    total_reward = 0.0
    done = False
    step = 0

    while not done and step < max_steps:
        # Use heuristic agent for best possible score
        action = choose_best(env)
        if action is None:
            action = ACTIONS[step % len(ACTIONS)]

        _, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = bool(terminated or truncated)
        step += 1

        print(f"[STEP] task={task_name} step={step} reward={reward:.4f}", flush=True)

    return total_reward, step


def main():
    total_reward = 0.0
    step = 0

    # Try HTTP server first, fall back to in-process execution
    try:
        total_reward, step = run_via_http()
    except Exception as e:
        print(f"# HTTP server not available ({e}), running locally", flush=True)
        try:
            total_reward, step = run_locally()
        except Exception as e2:
            print(f"# Local fallback also failed: {e2}", flush=True)
            # Still emit a valid [END] so the evaluator can parse something
            step = 1

    # ── [END] block ────────────────────────────────────────────
    score = total_reward / max(step, 1)
    print(f"[END] task={task_name} score={score:.4f} steps={step}", flush=True)


if __name__ == "__main__":
    main()
