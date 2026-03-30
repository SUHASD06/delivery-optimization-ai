---
title: Delivery Optimization AI
emoji: 🚛
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# 🚀 AI-Driven Delivery Optimization System

> **Reinforcement Learning agent that learns to optimize last-mile delivery routes under dynamic traffic and fuel constraints.**

Built with `Gymnasium` · `Stable-Baselines3 (PPO)` · `PyTorch` · `Gradio`

<!-- 📸 DEVELOPER: Replace the link below with a screenshot or GIF of your Gradio UI -->
<!-- ![Interactive Demo UI](demo_ui_screenshot.png) -->

---

## 1. Problem Statement

Last-mile delivery represents **53% of total shipping costs** globally. Traditional greedy routing algorithms fail under real-world constraints — fluctuating traffic patterns, limited fuel, and hard delivery deadlines create brittle systems prone to cascading failures.

**This project trains an RL agent that autonomously learns to:**
- Navigate a dynamic 10×10 grid with traffic hotspots
- Manage fuel strategically (refuel vs. deliver trade-offs)
- Complete all deliveries in minimum steps under stochastic conditions
- Outperform a hand-crafted heuristic baseline

---

## 2. Environment Design (`DeliveryEnv`)

A custom Gymnasium environment modeling realistic logistics constraints:

### Observation Space (11-dimensional float32 vector)

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | `agent_x` | [0, 1] | Normalized x-coordinate |
| 1 | `agent_y` | [0, 1] | Normalized y-coordinate |
| 2 | `fuel_level` | [0, 1] | Current fuel / max fuel |
| 3 | `pending_count` | [0, 1] | Remaining deliveries / 5 |
| 4 | `nearest_delivery` | [0, 1] | Distance to closest delivery |
| 5 | `nearest_fuel` | [0, 1] | Distance to closest fuel station |
| 6 | `traffic` | [0, 1] | Traffic level at current cell |
| 7-9 | `sorted_dists` | [0, 1] | Sorted distances to top-3 deliveries |
| 10 | `steps_remaining` | [0, 1] | Remaining budget / 200 |

### Action Space (Discrete, 5 actions)

| Action | Movement |
|--------|----------|
| 0 | Up (+Y) |
| 1 | Down (-Y) |
| 2 | Left (-X) |
| 3 | Right (+X) |
| 4 | Refuel (at station) |

### Environment Phases (Curriculum Learning)

| Phase | Layout | Traffic | Fuel Noise | Purpose |
|-------|--------|---------|------------|---------|
| **1** | Fixed (deterministic) | Static | None | Learn basic navigation |
| **2** | Randomized | Dynamic | None | Generalize to new maps |
| **3** | Randomized | Dynamic + hotspots | Stochastic | Handle full uncertainty |

---

## 3. Reward Design

The reward function uses **dense shaping** to provide continuous learning signal, combined with **milestone bonuses** for key achievements:

| Component | Value | Purpose |
|-----------|-------|---------|
| Step cost | −0.05 | Mild efficiency pressure |
| Distance progress | +2.0 × Δ closest | Dense gradient toward deliveries |
| Fuel usage | −0.1 × fuel_used | Penalize wasteful routes |
| Delivery completed | +8.0 + speed bonus | Big milestone reward |
| Completion (all done) | +25.0 | Massive bonus for task completion |
| Fuel depletion | −5.0 | Terminal failure penalty |
| Smart refueling | +0.5 | Reward proactive fuel management |
| Low-fuel station approach | +0.2 | Encourage fuel-awareness |

**Design rationale:** The reward avoids sparse signals that stall learning. The distance-progress term creates a "gradient descent" toward deliveries, while escalating milestone bonuses ensure the agent prioritizes task completion over local optimization.

---

## 4. RL Approach — Proximal Policy Optimization (PPO)

### Architecture

```
Input (11-dim obs) → Linear(256) → ReLU → Linear(256) → ReLU → Policy Head (5 actions)
                                                               → Value Head (scalar)
```

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate` | 3e-4 | Standard for PPO |
| `n_steps` | 2048 | Sufficient trajectory length |
| `batch_size` | 128 | Balanced gradient estimates |
| `n_epochs` | 10 | Multiple passes per batch |
| `gamma` | 0.99 | Long-horizon credit assignment |
| `gae_lambda` | 0.95 | Advantage estimation smoothing |
| `clip_range` | 0.2 | Standard PPO clipping |
| `ent_coef` | 0.005 | Mild exploration bonus |
| `net_arch` | [256, 256] | Capacity for complex policies |

### Curriculum Training

```
Phase 1 (200K steps) → Phase 2 (200K steps, fine-tuned from P1) → Phase 3 (200K steps, fine-tuned from P2)
```

Transfer learning between phases enables the agent to build on previously learned behaviors rather than starting from scratch.

---

## 5. Results

### Training Convergence (Phase 1)

The PPO agent converges to near-optimal policy within 100K steps, achieving **3/3 delivery completion rate** and 90+ cumulative reward:

![Training Convergence](performance_p1.png)

### Evaluation: PPO vs Heuristic Baseline (100 episodes each)

#### Phase 1 — Deterministic Environment ✅

| Method | Avg Reward | Deliveries | Fuel Used | Steps |
|--------|-----------|------------|-----------|-------|
| **Heuristic Baseline** | 89.53 | **3.0/3** | 13.26 | 4.0 |
| **PPO Agent** | **90.46** | **3.0/3** | 17.0 | 16.0 |

> ✅ **PPO beats the heuristic** by +0.93 reward while matching 100% delivery rate.

![Phase 1 Result Comparison](comparison_p1.png)

#### Phase 2 — Randomized Layout

| Method | Avg Reward | Deliveries | Fuel Used | Steps |
|--------|-----------|------------|-----------|-------|
| **Heuristic Baseline** | **77.26** | **2.63/3** | 24.4 | 5.81 |
| **PPO Agent** | 5.72 | 0.39/3 | 16.09 | 12.05 |

#### Phase 3 — Full Stochastic

| Method | Avg Reward | Deliveries | Fuel Used | Steps |
|--------|-----------|------------|-----------|-------|
| **Heuristic Baseline** | **54.01** | **2.06/3** | 22.05 | 4.80 |
| **PPO Agent** | 4.97 | 0.34/3 | 15.91 | 9.60 |

#### Analysis

The heuristic baseline excels on Phases 2-3 because it uses **explicit TSP route planning** and **fuel-aware look-ahead** — capabilities that an MLP policy must discover purely through trial-and-error. This is a well-known challenge in RL: generalizing to new map configurations requires either (a) significantly more training data, (b) map-aware architectures (CNN/GNN), or (c) meta-RL techniques. The Phase 1 victory demonstrates that given a consistent environment, the PPO agent learns an effective policy that **surpasses hand-crafted heuristics**.

> *Reproduce: `python evaluate.py --phase all`*

---

## 6. Heuristic Baseline

The baseline agent (`agent/baseline.py`) uses a multi-factor scoring system:

- **Route optimization**: TSP-style permutation search over delivery order
- **Fuel awareness**: Proactive refueling when fuel drops below safety threshold
- **Future feasibility**: Checks if a delivery leads to stranded states
- **Cluster detection**: Prefers deliveries near other pending deliveries
- **Traffic avoidance**: Penalizes high-traffic routes

This provides a strong comparison point — the PPO agent must learn to match or exceed this hand-crafted intelligence.

---

## 7. How to Run

### Setup

```bash
pip install -r requirements.txt
```

### Train the Agent

```bash
# Train Phase 1 (deterministic)
python train_ppo.py --phase 1 --steps 200000

# Train Phase 2 (loads from Phase 1 checkpoint)
python train_ppo.py --phase 2 --steps 200000

# Train all phases sequentially
python train_ppo.py --phase all --steps 200000
```

### Evaluate

```bash
# Compare PPO vs Heuristic on Phase 1
python evaluate.py --phase 1

# Evaluate all phases
python evaluate.py --phase all
```

### Validate Environment

```bash
python test_env.py
```

### Launch Interactive Demo

```bash
python app.py
# Opens at http://localhost:7860
```

### Docker

```bash
docker build -t delivery-optimizer .
docker run -p 7860:7860 delivery-optimizer
```

---

## 8. Project Structure

```
delivery_openenv/
├── env/
│   ├── environment.py    # Gymnasium environment (DeliveryEnv)
│   ├── models.py         # Pydantic data models (Observation, Action, Reward)
│   ├── utils.py          # Helper functions (distance, traffic, clustering)
│   ├── grader.py         # Task grading logic
│   └── tasks.py          # Predefined task configurations
├── agent/
│   └── baseline.py       # Heuristic baseline agent
├── train_ppo.py          # PPO training pipeline with telemetry
├── evaluate.py           # Agent comparison and benchmarking
├── test_env.py           # Environment validation test suite
├── test_ppo.py           # PPO smoke test
├── app.py                # Gradio interactive demo
├── visualize.py          # Matplotlib animation helper
├── openenv.yaml          # OpenEnv configuration
├── Dockerfile            # Container deployment
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 9. Limitations & Future Work

| Limitation | Potential Solution |
|------------|-------------------|
| Fixed 10×10 grid | Parameterize grid size, test larger maps |
| Max 5 deliveries | Dynamic observation with attention mechanism |
| Single agent | Multi-agent coordination for fleet management |
| MLP policy | CNN/Transformer for spatial reasoning |
| Synthetic traffic | Integrate real-world traffic APIs |

---

## 10. Tech Stack

| Component | Technology |
|-----------|-----------|
| RL Framework | Gymnasium 0.29+ |
| Training Algorithm | PPO (Stable-Baselines3) |
| Neural Network | PyTorch |
| Data Models | Pydantic |
| Visualization | Matplotlib, Gradio |
| Containerization | Docker |
