"""
inference.py — OpenEnv Hackathon Inference Script.

Connects to the running environment server, resets the environment,
runs an agent for one episode, and prints [START]/[STEP]/[END] structured
output blocks required by the Phase 2 evaluator.
"""
import os
import sys
import requests


def main():
    base_url = os.getenv("OPENENV_URL", "http://localhost:8000")

    # ── Reset ──────────────────────────────────────────────────
    try:
        reset_resp = requests.post(f"{base_url}/reset", json={}, timeout=30)
        reset_resp.raise_for_status()
        reset_data = reset_resp.json()
    except Exception as e:
        print(f"[ERROR] Could not reset environment: {e}", flush=True)
        sys.exit(1)

    task_name = os.getenv("OPENENV_TASK", "delivery-optimization")

    # ── Print [START] block ────────────────────────────────────
    print(f"[START] task={task_name} step=0 reward=0.0", flush=True)

    total_reward = 0.0
    done = False
    step = 0
    max_steps = 200

    # Simple greedy heuristic: cycle through actions
    # Action: 0=up, 1=down, 2=left, 3=right, 4=refuel
    actions = [0, 3, 1, 2, 0, 3]  # basic pattern

    while not done and step < max_steps:
        action = actions[step % len(actions)]

        try:
            step_resp = requests.post(
                f"{base_url}/step",
                json={"action": action},
                timeout=30,
            )
            step_resp.raise_for_status()
            res = step_resp.json()
        except Exception as e:
            print(f"[ERROR] Step {step} failed: {e}", flush=True)
            break

        reward = float(res.get("reward", 0.0))
        total_reward += reward
        done = bool(res.get("done") or res.get("terminated") or res.get("truncated"))

        step += 1

        # ── Print [STEP] block for every step ─────────────────
        print(
            f"[STEP] task={task_name} step={step} reward={reward:.4f}",
            flush=True,
        )

    # ── Print [END] block ──────────────────────────────────────
    score = total_reward / max(step, 1)
    print(
        f"[END] task={task_name} score={score:.4f} steps={step}",
        flush=True,
    )


if __name__ == "__main__":
    main()
