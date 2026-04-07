"""
AI Delivery Optimization — Interactive Demo
Rich Gradio interface with visual grid, agent comparison, and metrics dashboard.
Also serves OpenEnv-compliant API endpoints (/reset, /step, /state) for
the Scaler Meta PyTorch Hackathon automated evaluator.

Run:  python app.py
"""
import os
import io
import json
import numpy as np
import folium
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Dict, Any
from pydantic import BaseModel
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from env.environment import DeliveryEnv, GRID_SIZE, FUEL_STATIONS
from agent.baseline import choose_best

# ──────────────────────────────────────────────────────────────
# Grid to Real-world GPS mapping
# ──────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 34.020, 34.090  # Culver City to Hollywood roughly
LON_MIN, LON_MAX = -118.420, -118.320 # Santa Monica to K-Town roughly

def grid_to_latlon(x, y):
    # Map (0..9) grid to realistic Lat/Lon bounding box
    lon = LON_MIN + (x / (GRID_SIZE - 1)) * (LON_MAX - LON_MIN)
    lat = LAT_MIN + (y / (GRID_SIZE - 1)) * (LAT_MAX - LAT_MIN)
    return lat, lon

# ──────────────────────────────────────────────────────────────
# Global environment instance for OpenEnv API
# ──────────────────────────────────────────────────────────────
_api_env: Optional[DeliveryEnv] = None


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    options: Optional[Dict[str, Any]] = None
    task: Optional[str] = None  # "easy", "medium", "hard"


class StepRequest(BaseModel):
    action: int  # 0=up, 1=down, 2=left, 3=right, 4=refuel


# ──────────────────────────────────────────────────────────────
# Grid Visualization
# ──────────────────────────────────────────────────────────────
def render_map(env, path_history, title="Delivery Simulation", total_reward=0, info=None, ppo_not_found=False):
    """Render a realistic leafleat map using Folium."""
    if ppo_not_found:
        return f"""
        <div style="background-color: #1a1a1e; padding: 40px; border-radius: 12px; text-align: center;">
            <h3 style="color: #ff4444; font-family: monospace;">PPO Model Not Found</h3>
            <p style="color: #888;">Train the PPO agent first to see results here.</p>
        </div>
        """

    info = info or {}
    del_count = info.get("deliveries_completed", 0)
    fuel_used = info.get("fuel_used", 0)

    center_lat = (LAT_MIN + LAT_MAX) / 2
    center_lon = (LON_MIN + LON_MAX) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB dark_matter")

    # Helper function to snap path to real roads using OSRM
    def get_road_route(points):
        if not points or len(points) < 2: return points
        import requests
        # OSRM expects lon,lat format
        coords = ";".join([f"{lon:.5f},{lat:.5f}" for lat, lon in points])
        url = f"http://router.project-osrm.org/route/v1/driving/{coords}?overview=full&geometries=geojson"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == "Ok":
                    # GeoJSON is [lon, lat], folium needs [lat, lon]
                    route = data["routes"][0]["geometry"]["coordinates"]
                    return [[lat, lon] for lon, lat in route]
        except Exception as e:
            print("OSRM Routing failed, using straight lines:", e)
        return points

    # Path history — plot agent steps and connections
    if path_history:
        raw_latlons = [grid_to_latlon(px, py) for px, py in path_history]
        
        # Split into chunks of 50 to avoid any OSRM URL length limits if the path is very long
        route_latlons = []
        for i in range(0, len(raw_latlons), 50):
            chunk = raw_latlons[i:i+51] # overlap by 1 to connect
            route_latlons.extend(get_road_route(chunk))
            
        # A* style yellow line snapped to roads
        folium.PolyLine(route_latlons, color="#ffd93d", weight=5, opacity=0.9).add_to(m)
        
        # Blue visited nodes (original logic points)
        for lat, lon in raw_latlons:
            folium.CircleMarker([lat, lon], radius=2, color="#0088ff", fill=True, fillOpacity=0.7).add_to(m)

    # Fuel stations (Red Pins)
    for fx, fy in env.fuel_stations:
        flat, flon = grid_to_latlon(fx, fy)
        folium.Marker([flat, flon], icon=folium.Icon(color="darkred", icon="tint", prefix="fa")).add_to(m)

    # Pending deliveries (Blue Pins)
    for dx, dy in env.pending_deliveries:
        dlat, dlon = grid_to_latlon(dx, dy)
        folium.Marker([dlat, dlon], icon=folium.Icon(color="darkblue", icon="gift", prefix="fa")).add_to(m)

    # Start location (Green pin)
    if path_history:
        sx, sy = path_history[0]
        slat, slon = grid_to_latlon(sx, sy)
        folium.Marker([slat, slon], icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(m)

    # Target/Current Location (Red/Black pin)
    ax, ay = env.current_location
    alat, alon = grid_to_latlon(ax, ay)
    folium.Marker([alat, alon], icon=folium.Icon(color="red", icon="truck", prefix="fa")).add_to(m)

    map_html = m._repr_html_()

    html = f"""
    <div style="background-color: #1a1a1e; padding: 20px; border-radius: 12px; font-family: 'Inter', sans-serif;">
        <h3 style="color: #ffffff; text-align: center; font-weight: bold; margin-bottom: 20px; font-family: monospace;">{title}</h3>
        
        <div style="display: flex; justify-content: space-around; background-color: #242429; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <div style="text-align: center;">
                <p style="color: #888; font-size: 11px; margin: 0 0 5px 0; font-weight: bold; letter-spacing: 1px;">DELIVERIES</p>
                <p style="color: #0088ff; font-size: 18px; margin: 0; font-weight: bold;">{del_count}/3</p>
            </div>
            <div style="text-align: center;">
                <p style="color: #888; font-size: 11px; margin: 0 0 5px 0; font-weight: bold; letter-spacing: 1px;">FUEL USED</p>
                <p style="color: #0088ff; font-size: 18px; margin: 0; font-weight: bold;">{fuel_used:.1f}</p>
            </div>
            <div style="text-align: center;">
                <p style="color: #888; font-size: 11px; margin: 0 0 5px 0; font-weight: bold; letter-spacing: 1px;">TOTAL REWARD</p>
                <p style="color: #0088ff; font-size: 18px; margin: 0; font-weight: bold;">{total_reward:.1f}</p>
            </div>
        </div>
        
        <div style="border-radius: 8px; overflow: hidden; border: 1px solid #333; height: 400px;">
            {{MAP_PLACEHOLDER}}
        </div>
    </div>
    """
    return html.replace("{MAP_PLACEHOLDER}", map_html)


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
    h_html = render_map(h_env, h_path, f"Heuristic Baseline (Phase {phase})", h_reward, h_info)

    # Run PPO
    p_env, p_path, p_reward, p_info = run_ppo_agent(phase)
    if p_env is not None:
        p_html = render_map(p_env, p_path, f"PPO Agent (Phase {phase})", p_reward, p_info)
    else:
        p_html = render_map(None, None, f"PPO Agent (Phase {phase})", ppo_not_found=True)

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

    return h_html, p_html, comparison_md


def run_single_demo(phase, agent_type):
    """Run a single agent and show step-by-step."""
    phase = int(phase)

    if agent_type == "Heuristic Baseline":
        env, path, total_reward, info = run_heuristic_agent(phase)
    else:
        env, path, total_reward, info = run_ppo_agent(phase)
        if env is None:
            return render_map(None, None, f"{agent_type} (Phase {phase})", ppo_not_found=True), "PPO model not found. Train first."

    html = render_map(env, path, f"{agent_type} — Phase {phase}", total_reward, info)

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
    return html, report


# ──────────────────────────────────────────────────────────────
# Build Gradio App
# ──────────────────────────────────────────────────────────────
def create_gradio_blocks():
    theme = gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(theme=theme, title="AI Delivery Optimization") as blocks:
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
                    heuristic_html = gr.HTML()
                    ppo_html = gr.HTML()

                comparison_md = gr.Markdown()
                compare_btn.click(
                    fn=compare_agents,
                    inputs=[phase_select],
                    outputs=[heuristic_html, ppo_html, comparison_md]
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

                demo_html = gr.HTML()
                demo_report = gr.Markdown()
                demo_btn.click(
                    fn=run_single_demo,
                    inputs=[demo_phase, demo_agent],
                    outputs=[demo_html, demo_report]
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

    return blocks


# ──────────────────────────────────────────────────────────────
# Create the MAIN FastAPI app with OpenEnv API routes
# Then mount Gradio onto it
# This runs at MODULE LEVEL so it works on HuggingFace Spaces
# ──────────────────────────────────────────────────────────────

# Step 1: Create FastAPI app with API routes
app = FastAPI(title="Delivery Optimization OpenEnv")


@app.get("/health")
def health():
    return {"status": "ok", "env": "delivery-optimization"}


@app.post("/reset")
def api_reset(request: ResetRequest = None):
    global _api_env
    phase = 1
    if request and request.task:
        phase = {"easy": 1, "medium": 2, "hard": 3}.get(request.task, 1)
    _api_env = DeliveryEnv(phase=phase)
    seed = request.seed if request else None
    obs, info = _api_env.reset(seed=seed)
    return {"observation": obs.tolist(), "info": info}


@app.post("/step")
def api_step(request: StepRequest):
    global _api_env
    if _api_env is None:
        _api_env = DeliveryEnv(phase=1)
        _api_env.reset()
    obs, reward, terminated, truncated, info = _api_env.step(request.action)
    return {
        "observation": obs.tolist(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "done": bool(terminated or truncated),
        "info": info,
    }


@app.get("/state")
def api_state():
    global _api_env
    if _api_env is None:
        _api_env = DeliveryEnv(phase=1)
        _api_env.reset()
    s = _api_env.state()
    return {
        "location": list(s["location"]),
        "fuel": float(s["fuel"]),
        "time_elapsed": float(s["time"]),
        "pending_deliveries": [list(d) for d in s["pending"]],
        "deadlines": list(s["deadlines"]),
        "steps_taken": int(_api_env.steps_taken),
    }


# Step 2: Create Gradio blocks and mount on the FastAPI app
gradio_blocks = create_gradio_blocks()
app = gr.mount_gradio_app(app, gradio_blocks, path="/")


# Step 3: Entry point — run with uvicorn
if __name__ == "__main__":
    print("=" * 60)
    print("  Delivery Optimization OpenEnv Server")
    print("  API: /reset, /step, /state, /health")
    print("  UI:  http://127.0.0.1:7860")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=7860)
