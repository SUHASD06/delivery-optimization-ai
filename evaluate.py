"""
Evaluation: Heuristic Baseline vs PPO Agent
Generates a comprehensive comparison table and saves results.

Usage:  python evaluate.py --phase 1
        python evaluate.py --phase 2
        python evaluate.py --phase 3
        python evaluate.py --phase all
"""
import argparse
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from env.environment import DeliveryEnv
from agent.baseline import choose_best


def evaluate_heuristic(phase, episodes=100):
    env = DeliveryEnv(phase=phase)
    results = {"rewards": [], "deliveries": [], "fuel": [], "steps": []}

    for _ in range(episodes):
        obs, _ = env.reset()
        done   = False
        info   = {}
        ep_r   = 0

        while not done:
            action = choose_best(env)
            if action is None:
                break
            obs, reward, terminated, truncated, info = env.step(action)
            ep_r += reward
            done  = terminated or truncated

        results["rewards"].append(ep_r)
        results["deliveries"].append(info.get("deliveries_completed", 0))
        results["fuel"].append(info.get("fuel_used", 0))
        results["steps"].append(info.get("steps_taken", 0))

    return results


def evaluate_ppo(model_path, phase, episodes=100):
    from stable_baselines3 import PPO
    env = DeliveryEnv(phase=phase)
    results = {"rewards": [], "deliveries": [], "fuel": [], "steps": []}

    try:
        model = PPO.load(model_path, env=env)
    except FileNotFoundError:
        print(f"  ⚠️  Model {model_path}.zip not found, skipping PPO eval.")
        return results

    for _ in range(episodes):
        obs, _ = env.reset()
        done   = False
        info   = {}
        ep_r   = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_r += reward
            done  = terminated or truncated

        results["rewards"].append(ep_r)
        results["deliveries"].append(info.get("deliveries_completed", 0))
        results["fuel"].append(info.get("fuel_used", 0))
        results["steps"].append(info.get("steps_taken", 0))

    return results


def plot_comparison(heuristic, ppo, phase):
    """Generate side-by-side comparison chart."""
    plt.style.use('seaborn-v0_8-darkgrid')

    metrics = ["deliveries", "fuel", "steps", "rewards"]
    titles  = ["Deliveries Completed", "Fuel Used", "Steps Taken", "Total Reward"]
    colors  = [("#55A868", "#4C72B0"), ("#DD8452", "#C44E52"),
               ("#8172B2", "#937860"), ("#4C72B0", "#55A868")]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (metric, title, (c1, c2)) in enumerate(zip(metrics, titles, colors)):
        ax = axes[i]
        h_vals = heuristic[metric]
        p_vals = ppo[metric]

        if h_vals and p_vals:
            ax.hist(h_vals, bins=20, alpha=0.6, color=c1, label="Heuristic", edgecolor="white")
            ax.hist(p_vals, bins=20, alpha=0.6, color=c2, label="PPO Agent", edgecolor="white")
            ax.axvline(np.mean(h_vals), color=c1, linestyle="--", linewidth=2)
            ax.axvline(np.mean(p_vals), color=c2, linestyle="--", linewidth=2)
        elif h_vals:
            ax.hist(h_vals, bins=20, alpha=0.7, color=c1, label="Heuristic", edgecolor="white")
            ax.axvline(np.mean(h_vals), color=c1, linestyle="--", linewidth=2)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Phase {phase} — Heuristic vs PPO Agent Comparison",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fname = f"comparison_p{phase}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved {fname}")


def summarize(results):
    if not results["rewards"]:
        return 0, 0, 0, 0
    return (
        round(np.mean(results["rewards"]), 2),
        round(np.mean(results["deliveries"]), 2),
        round(np.mean(results["fuel"]), 2),
        round(np.mean(results["steps"]), 2),
    )


def run_evaluation(phase, episodes=100):
    model_path = f"delivery_ppo_p{phase}"

    print(f"\n{'='*60}")
    print(f"  📊 EVALUATING PHASE {phase}  ({episodes} episodes each)")
    print(f"{'='*60}")

    print("\n  Running heuristic baseline...")
    h_results = evaluate_heuristic(phase, episodes)
    h_r, h_d, h_f, h_s = summarize(h_results)

    print("  Running PPO agent...")
    p_results = evaluate_ppo(model_path, phase, episodes)
    p_r, p_d, p_f, p_s = summarize(p_results)

    rows = {
        "Method":              ["Heuristic Baseline", f"PPO Agent (Phase {phase})"],
        "Avg Reward":          [h_r, p_r],
        "Avg Deliveries":      [h_d, p_d],
        "Completion Rate (%)": [round(h_d / 3 * 100, 1), round(p_d / 3 * 100, 1)],
        "Avg Fuel Used":       [h_f, p_f],
        "Avg Steps":           [h_s, p_s],
    }

    df = pd.DataFrame(rows)
    print(f"\n{'='*65}")
    print("  EVALUATION COMPARISON")
    print(f"{'='*65}")
    print(df.to_string(index=False))
    print(f"{'='*65}")

    # Generate comparison chart
    plot_comparison(h_results, p_results, phase)

    # Save results to JSON for README embedding
    eval_results = {
        "phase": phase,
        "episodes": episodes,
        "heuristic": {"reward": h_r, "deliveries": h_d, "fuel": h_f, "steps": h_s},
        "ppo":       {"reward": p_r, "deliveries": p_d, "fuel": p_f, "steps": p_s},
    }
    with open(f"eval_results_p{phase}.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    if p_r > h_r:
        print(f"\n  ✅ PPO beats the baseline! ({p_r:.1f} vs {h_r:.1f} reward)")
        print(f"  ✅ Deliveries: PPO {p_d:.1f} vs Heuristic {h_d:.1f}")
    elif p_r == 0:
        print("\n  ⚠️  PPO model not found. Train first: python train_ppo.py --phase 1")
    else:
        print(f"\n  ⚠️  PPO ({p_r:.1f}) not yet beating baseline ({h_r:.1f}). Try more training.")

    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate PPO agent against heuristic baseline"
    )
    parser.add_argument("--phase", default="1",
                        help="Phase to evaluate (1/2/3 or 'all')")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    if args.phase == "all":
        all_results = {}
        for p in [1, 2, 3]:
            all_results[f"phase_{p}"] = run_evaluation(p, args.episodes)
        with open("eval_results_all.json", "w") as f:
            json.dump(all_results, f, indent=2)
    else:
        run_evaluation(int(args.phase), args.episodes)
