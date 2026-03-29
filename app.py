"""
AI Delivery Optimization — Interactive Demo
Rich Gradio interface with visual grid, agent comparison, and metrics dashboard.

Run:  python app.py
"""
import os
import io
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import gradio as gr
from env.environment import DeliveryEnv, GRID_SIZE, FUEL_STATIONS
from agent.baseline import choose_best


# ──────────────────────────────────────────────────────────────
# Grid Visualization
# ──────────────────────────────────────────────────────────────
def render_grid(env, path_history, title="Delivery Simulation", step_idx=None):
    """Render the grid as a matplotlib figure."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    # Traffic heatmap background
    traffic = np.array(env.traffic_map, dtype=float).T  # transpose for x/y
    cmap = LinearSegmentedColormap.from_list("traffic", ["#1a1a2e", "#16213e", "#e94560"])
    ax.imshow(traffic, origin="lower", cmap=cmap, alpha=0.3,
              extent=(-0.5, GRID_SIZE - 0.5, -0.5, GRID_SIZE - 0.5))

    # Grid lines
    for i in range(GRID_SIZE + 1):
        ax.axhline(i - 0.5, color="#333", linewidth=0.5, alpha=0.3)
        ax.axvline(i - 0.5, color="#333", linewidth=0.5, alpha=0.3)

    # Fuel stations
    for fx, fy in env.fuel_stations:
        ax.plot(fx, fy, "s", color="#00d2ff", markersize=18, markeredgecolor="white",
                markeredgewidth=2, zorder=5)
        ax.text(fx, fy, "⛽", ha="center", va="center", fontsize=12, zorder=6)

    # Pending deliveries
    for i, (dx, dy) in enumerate(env.pending_deliveries):
        ax.plot(dx, dy, "D", color="#ff6b6b", markersize=16, markeredgecolor="white",
                markeredgewidth=2, zorder=5)
        ax.text(dx, dy, f"📦", ha="center", va="center", fontsize=11, zorder=6)

    # Path history
    if path_history and len(path_history) > 1:
        display_path = path_history[:step_idx + 1] if step_idx is not None else path_history
        px = [p[0] for p in display_path]
        py = [p[1] for p in display_path]
        ax.plot(px, py, "-", color="#ffd93d", linewidth=2.5, alpha=0.7, zorder=3)
        ax.plot(px, py, "o", color="#ffd93d", markersize=5, alpha=0.5, zorder=3)

    # Agent current position
    ax_pos, ay_pos = env.current_location
    ax.plot(ax_pos, ay_pos, "o", color="#6bcb77", markersize=22, markeredgecolor="white",
            markeredgewidth=3, zorder=7)
    ax.text(ax_pos, ay_pos, "🚛", ha="center", va="center", fontsize=14, zorder=8)

    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_facecolor("#0f0f23")
    fig.patch.set_facecolor("#0f0f23")
    ax.tick_params(colors="#aaa", labelsize=9)
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=12)

    # Info box
    info_text = (f"Fuel: {env.fuel:.1f}/{env.max_fuel:.0f}  |  "
                 f"Delivered: {env.deliveries_completed}/{env.deliveries_completed + len(env.pending_deliveries)}  |  "
                 f"Steps: {env.steps_taken}")
    ax.text(GRID_SIZE / 2, -1.2, info_text, ha="center", va="center",
            color="#ddd", fontsize=11, bbox=dict(boxstyle="round,pad=0.4",
            facecolor="#1a1a2e", edgecolor="#444", alpha=0.9))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#6bcb77", edgecolor="white", label="Agent 🚛"),
        mpatches.Patch(facecolor="#ff6b6b", edgecolor="white", label="Deliveries 📦"),
        mpatches.Patch(facecolor="#00d2ff", edgecolor="white", label="Fuel Stations ⛽"),
        mpatches.Patch(facecolor="#ffd93d", edgecolor="white", label="Path Taken"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# Run Agents
# ──────────────────────────────────────────────────────────────
def run_heuristic_agent(phase):
    """Run heuristic baseline and return path + metrics."""
    env = DeliveryEnv(phase=int(phase))
    env.reset(seed=42)
    path = [env.current_location]
    total_reward = 0
    done = False
    step = 0

    while not done and step < 200:
        step += 1
        action = choose_best(env)
        if action is None:
            break
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        path.append(env.current_location)

    return env, path, total_reward, info


def run_ppo_agent(phase):
    """Run PPO agent and return path + metrics."""
    from stable_baselines3 import PPO

    env = DeliveryEnv(phase=int(phase))
    model_path = f"delivery_ppo_p{int(phase)}"

    if not os.path.exists(model_path + ".zip"):
        return None, [], 0, {"error": f"Model {model_path}.zip not found"}

    model = PPO.load(model_path, env=env)
    obs, _ = env.reset(seed=42)
    path = [env.current_location]
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        path.append(env.current_location)

    return env, path, total_reward, info


# ──────────────────────────────────────────────────────────────
# Gradio Interface Functions
# ──────────────────────────────────────────────────────────────
def compare_agents(phase):
    """Run both agents and generate comparison visual."""
    phase = int(phase)

    # Run heuristic
    h_env, h_path, h_reward, h_info = run_heuristic_agent(phase)
    h_fig = render_grid(h_env, h_path, f"Heuristic Baseline (Phase {phase})")

    # Run PPO
    p_env, p_path, p_reward, p_info = run_ppo_agent(phase)
    if p_env is not None:
        p_fig = render_grid(p_env, p_path, f"PPO Agent (Phase {phase})")
    else:
        p_fig, ax = plt.subplots(figsize=(7, 7))
        ax.text(0.5, 0.5, f"PPO model not found\nTrain first:\npython train_ppo.py --phase {phase}",
                ha="center", va="center", fontsize=14, color="#ff6b6b",
                transform=ax.transAxes)
        ax.set_facecolor("#0f0f23")
        fig.patch.set_facecolor("#0f0f23")

    # Build metrics table
    h_del = h_info.get("deliveries_completed", 0)
    h_fuel = h_info.get("fuel_used", 0)
    h_steps = h_info.get("steps_taken", 0)

    if p_env is not None:
        p_del = p_info.get("deliveries_completed", 0)
        p_fuel = p_info.get("fuel_used", 0)
        p_steps = p_info.get("steps_taken", 0)
    else:
        p_del = p_fuel = p_steps = 0
        p_reward = 0

    comparison_md = f"""
## 📊 Phase {phase} — Agent Comparison

| Metric | 🧠 Heuristic Baseline | 🤖 PPO Agent | Winner |
|--------|----------------------|--------------|--------|
| **Deliveries Completed** | {h_del}/3 | {p_del}/3 | {'🤖 PPO' if p_del > h_del else ('🧠 Heuristic' if h_del > p_del else '🤝 Tie')} |
| **Total Reward** | {h_reward:.1f} | {p_reward:.1f} | {'🤖 PPO' if p_reward > h_reward else '🧠 Heuristic'} |
| **Fuel Efficiency** | {h_fuel:.1f} | {p_fuel:.1f} | {'🤖 PPO' if p_fuel < h_fuel and p_del >= h_del else '🧠 Heuristic'} |
| **Steps Taken** | {h_steps} | {p_steps} | {'🤖 PPO' if p_steps < h_steps and p_del >= h_del else '🧠 Heuristic'} |

"""
    if p_reward > h_reward:
        comparison_md += "### ✅ PPO Agent outperforms the Heuristic Baseline!"
    elif p_env is None:
        comparison_md += "### ⚠️ Train the PPO agent first to see comparison"
    else:
        comparison_md += "### 📈 PPO agent is learning — try more training steps"

    return h_fig, p_fig, comparison_md


def run_single_demo(phase, agent_type):
    """Run a single agent and show step-by-step."""
    phase = int(phase)

    if agent_type == "Heuristic Baseline":
        env, path, total_reward, info = run_heuristic_agent(phase)
    else:
        env, path, total_reward, info = run_ppo_agent(phase)
        if env is None:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.text(0.5, 0.5, "PPO model not found", ha="center", va="center",
                    fontsize=14, color="#ff6b6b", transform=ax.transAxes)
            ax.set_facecolor("#0f0f23")
            fig.patch.set_facecolor("#0f0f23")
            return fig, "PPO model not found. Train first."

    fig = render_grid(env, path, f"{agent_type} — Phase {phase}")

    report = f"""## 📋 Episode Report — {agent_type}

| Metric | Value |
|--------|-------|
| **Phase** | {phase} |
| **Deliveries Completed** | {info.get('deliveries_completed', 0)}/3 |
| **Total Reward** | {total_reward:.2f} |
| **Fuel Used** | {info.get('fuel_used', 0):.2f} |
| **Steps Taken** | {info.get('steps_taken', 0)} |
| **Path Length** | {len(path)} cells |

{'✅ All deliveries completed!' if info.get('deliveries_completed', 0) >= 3 else '⚠️ Some deliveries remaining'}
"""
    return fig, report


# ──────────────────────────────────────────────────────────────
# Build Gradio App
# ──────────────────────────────────────────────────────────────
def create_app():
    theme = gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(theme=theme, title="AI Delivery Optimization") as app:
        gr.Markdown("""
# 🚀 AI-Driven Delivery Optimization System
### Reinforcement Learning Agent for Last-Mile Logistics

This demo compares a **rule-based heuristic** baseline against a **PPO deep RL agent**
trained to optimize delivery routes under dynamic traffic and fuel constraints.
        """)

        with gr.Tabs():
            # ---- Tab 1: Agent Comparison ----
            with gr.TabItem("⚔️ Agent Comparison"):
                gr.Markdown("### Compare Heuristic Baseline vs PPO Agent side-by-side")
                with gr.Row():
                    phase_select = gr.Radio(
                        choices=["1", "2", "3"],
                        value="1",
                        label="Environment Phase",
                        info="Phase 1: Fixed | Phase 2: Random Layout | Phase 3: Full Stochastic"
                    )
                    compare_btn = gr.Button("⚡ Run Comparison", variant="primary", size="lg")

                with gr.Row():
                    heuristic_plot = gr.Plot(label="🧠 Heuristic Baseline")
                    ppo_plot = gr.Plot(label="🤖 PPO Agent")

                comparison_md = gr.Markdown()
                compare_btn.click(
                    fn=compare_agents,
                    inputs=[phase_select],
                    outputs=[heuristic_plot, ppo_plot, comparison_md]
                )

            # ---- Tab 2: Single Agent Demo ----
            with gr.TabItem("🎮 Single Agent Demo"):
                gr.Markdown("### Run a single agent and inspect its behavior")
                with gr.Row():
                    demo_phase = gr.Radio(
                        choices=["1", "2", "3"],
                        value="1",
                        label="Environment Phase"
                    )
                    demo_agent = gr.Radio(
                        choices=["Heuristic Baseline", "PPO Agent"],
                        value="Heuristic Baseline",
                        label="Agent Type"
                    )
                    demo_btn = gr.Button("▶️ Run Demo", variant="primary", size="lg")

                demo_plot = gr.Plot(label="Agent Trajectory")
                demo_report = gr.Markdown()
                demo_btn.click(
                    fn=run_single_demo,
                    inputs=[demo_phase, demo_agent],
                    outputs=[demo_plot, demo_report]
                )

            # ---- Tab 3: About ----
            with gr.TabItem("📖 About"):
                gr.Markdown("""
## Architecture

### Environment (`DeliveryEnv`)
- **Grid**: 10×10 with traffic hotspots and fuel stations
- **Observation**: 11-dimensional normalized vector `[x, y, fuel, pending_count, distances..., steps_left]`
- **Actions**: `Discrete(5)` — Up, Down, Left, Right, Refuel
- **Phases**: Curriculum from deterministic → stochastic

### Agent (PPO)
- **Algorithm**: Proximal Policy Optimization (Stable-Baselines3)
- **Network**: MLP with 2×256 hidden layers
- **Training**: Phased curriculum — Phase 1 → Phase 2 → Phase 3
- **Reward Design**: Dense proximity gradient + milestone bonuses + fuel efficiency

### Reward Components
| Component | Value | Purpose |
|-----------|-------|---------|
| Step cost | -0.05 | Encourages efficiency |
| Progress toward delivery | +2.0 × Δdistance | Dense learning signal |
| Delivery completed | +8.0 + speed bonus | Big milestone reward |
| All deliveries done | +25.0 | Completion bonus |
| Fuel depletion | -5.0 | Punish running out |
| Smart refueling | +0.5 | Reward fuel management |

### Tech Stack
`Python` · `Gymnasium` · `Stable-Baselines3` · `PyTorch` · `Gradio` · `Matplotlib`
                """)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
