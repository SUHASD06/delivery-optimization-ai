"""
Phased PPO Training Pipeline
  Phase 1 → Fixed deterministic env  → delivery_ppo_p1.zip
  Phase 2 → Limited randomness       → delivery_ppo_p2.zip
  Phase 3 → Full stochasticity       → delivery_ppo_p3.zip

Run:  python train_ppo.py --phase 1
      python train_ppo.py --phase 2   (starts from p1 checkpoint)
      python train_ppo.py --phase all (runs all 3 sequentially)
"""
import os
import sys
import csv
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive so it works headless
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from env.environment import DeliveryEnv

# ──────────────────────────────────────────────────────────────
# Telemetry Wrapper — captures per-episode info dict → CSV
# ──────────────────────────────────────────────────────────────
class TelemetryWrapper(gym.Wrapper):
    def __init__(self, env, log_file):
        super().__init__(env)
        self.log_file  = log_file
        self.ep_reward = 0
        with open(self.log_file, "w", newline="") as f:
            csv.writer(f).writerow(
                ["episode", "deliveries_completed", "fuel_used", "steps_taken", "reward"]
            )
        self.episode = 0

    def reset(self, **kwargs):
        self.ep_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.ep_reward += reward
        if terminated or truncated:
            self.episode += 1
            with open(self.log_file, "a", newline="") as f:
                csv.writer(f).writerow([
                    self.episode,
                    info.get("deliveries_completed", 0),
                    round(info.get("fuel_used", 0), 3),
                    info.get("steps_taken", 0),
                    round(self.ep_reward, 3),
                ])
        return obs, reward, terminated, truncated, info


# ──────────────────────────────────────────────────────────────
# Progress callback — prints periodic stats
# ──────────────────────────────────────────────────────────────
class ProgressCallback(BaseCallback):
    def __init__(self, check_freq=10_000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Access the telemetry wrapper to get episode count
            print(f"  ... {self.n_calls:,} timesteps completed")
        return True


# ──────────────────────────────────────────────────────────────
# Plotting helper — publication-quality charts
# ──────────────────────────────────────────────────────────────
def plot_results(log_file, phase):
    data = pd.read_csv(log_file)
    if data.empty:
        print("No data to plot yet.")
        return

    window = max(1, len(data) // 20)     # 5% smoothing window
    smooth = data.rolling(window, min_periods=1).mean()

    # -- Style --
    plt.style.use('seaborn-v0_8-darkgrid')

    # -- Reward curve --
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(data["episode"], data["reward"], alpha=0.15, color="#4C72B0")
    ax.plot(smooth["episode"], smooth["reward"], color="#4C72B0", linewidth=2.5,
            label="Smoothed Reward")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.set_title(f"Phase {phase} — PPO Reward Convergence", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = f"performance_p{phase}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  📊 Saved {fname}")

    # -- Deliveries + Fuel twin-axis --
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(smooth["episode"], smooth["deliveries_completed"], color="#55A868",
             linewidth=2.5, label="Deliveries Completed")
    ax2.plot(smooth["episode"], smooth["fuel_used"], color="#DD8452",
             linewidth=2.5, label="Fuel Consumed")
    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Deliveries Completed", color="#55A868", fontsize=12)
    ax2.set_ylabel("Fuel Consumed", color="#DD8452", fontsize=12)
    ax1.set_title(f"Phase {phase} — Delivery Efficiency & Fuel Usage", fontsize=14, fontweight="bold")
    ax1.grid(alpha=0.3)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=11)

    fig.tight_layout()
    fname = f"telemetry_p{phase}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  📊 Saved {fname}")

    # -- Steps per episode --
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(data["episode"], data["steps_taken"], alpha=0.15, color="#C44E52")
    ax.plot(smooth["episode"], smooth["steps_taken"], color="#C44E52", linewidth=2.5,
            label="Steps per Episode")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Steps Taken", fontsize=12)
    ax.set_title(f"Phase {phase} — Episode Length Over Training", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = f"steps_p{phase}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  📊 Saved {fname}")


# ──────────────────────────────────────────────────────────────
# Single-phase training
# ──────────────────────────────────────────────────────────────
def train_phase(phase, timesteps=200_000, load_from=None):
    print(f"\n{'='*55}")
    print(f"  🚀 PHASE {phase} TRAINING  ({timesteps:,} timesteps)")
    print(f"{'='*55}")

    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/metrics_p{phase}.csv"

    raw_env  = DeliveryEnv(phase=phase)
    mon_env  = Monitor(raw_env, f"logs/monitor_p{phase}")
    env      = TelemetryWrapper(mon_env, log_file)

    # Network architecture — 2 hidden layers of 256 neurons each
    policy_kwargs = dict(net_arch=[256, 256])

    if load_from and os.path.exists(load_from + ".zip"):
        print(f"  📦 Loading checkpoint: {load_from}.zip")
        model = PPO.load(load_from, env=env)
        # Keep the same learning rate — the new distribution needs strong updates
        model.learning_rate = 3e-4
        model.ent_coef = 0.01  # boost exploration for new distribution
    else:
        model = PPO(
            "MlpPolicy", env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01 if phase >= 2 else 0.005,  # more exploration on random envs
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
        )

    callback = ProgressCallback(check_freq=20_000)
    model.learn(
        total_timesteps=timesteps,
        reset_num_timesteps=(load_from is None),
        callback=callback,
    )

    save_path = f"delivery_ppo_p{phase}"
    model.save(save_path)
    print(f"  ✅ Model saved → {save_path}.zip")

    plot_results(log_file, phase)
    return save_path


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO agent for delivery optimization"
    )
    parser.add_argument("--phase", default="1",
                        help="Phase to train: 1, 2, 3, or 'all'")
    parser.add_argument("--steps", type=int, default=200_000,
                        help="Timesteps per phase (default 200000)")
    args = parser.parse_args()

    if args.phase == "all":
        p1 = train_phase(1, args.steps)
        p2 = train_phase(2, args.steps, load_from=p1)
        _  = train_phase(3, args.steps, load_from=p2)
    else:
        phase = int(args.phase)
        prev  = f"delivery_ppo_p{phase - 1}" if phase > 1 else None
        train_phase(phase, args.steps, load_from=prev)
