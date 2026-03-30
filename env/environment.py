import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .models import Observation, Action
from .utils import distance, get_traffic, is_clustered
import random

GRID_SIZE = 10

# Fixed delivery positions for Phase 1 (deterministic)
FIXED_DELIVERIES = [(2, 3), (5, 5), (8, 2)]
FIXED_DEADLINES  = [30, 50, 70]
FIXED_FUEL_START = 20.0
FUEL_STATIONS    = [(0, 0), (5, 5)]
HOTSPOTS         = [(4, 4), (5, 5), (6, 6)]

# All grid cells available for random sampling in Phase 2/3
GRID_CELLS = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
              if (x, y) not in FUEL_STATIONS]


class DeliveryEnv(gym.Env):
    """
    Phased Delivery Environment for last-mile logistics optimization.

    The agent operates on a 10×10 grid and must complete all pending
    deliveries while managing fuel under dynamic traffic conditions.

    Phases:
      phase=1 → Fully deterministic (fixed map, fixed deliveries, no noise)
      phase=2 → Limited randomness  (randomized layout, no step noise)
      phase=3 → Full stochasticity  (traffic & fuel noise every step)

    Observation (11-dim float32 in [0, 1]):
      [0] agent_x / grid_size
      [1] agent_y / grid_size
      [2] fuel    / max_fuel
      [3] pending_deliveries_count / 5
      [4] nearest_delivery_dist   / max_dist
      [5] nearest_fuel_station    / max_dist
      [6] traffic_at_cell         / 2
      [7] sorted_delivery_dist_1  / max_dist
      [8] sorted_delivery_dist_2  / max_dist
      [9] sorted_delivery_dist_3  / max_dist
      [10] steps_remaining        / 200

    Action space (Discrete 5):
      0=up  1=down  2=left  3=right  4=refuel
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, phase=1, task=None, render_mode=None, **kwargs):
        super().__init__()
        self.phase = phase
        
        # Map OpenEnv standard task names to internal phases
        if task == "easy":
            self.phase = 1
        elif task == "medium":
            self.phase = 2
        elif task == "hard":
            self.phase = 3
            
        self.render_mode = render_mode

        # 11-dim normalized state [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(11,), dtype=np.float32
        )
        # 5 actions: 0=up, 1=down, 2=left, 3=right, 4=refuel
        self.action_space = spaces.Discrete(5)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.fuel_stations = list(FUEL_STATIONS)
        self.hotspots      = list(HOTSPOTS)
        self.max_fuel      = 20.0

        # --- Phase-based setup ---
        if self.phase == 1:
            # Fully deterministic
            self.current_location    = (0, 0)
            self.pending_deliveries  = list(FIXED_DELIVERIES)
            self.deadlines           = list(FIXED_DEADLINES)
            self.fuel                = FIXED_FUEL_START
            self.traffic_map         = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

        elif self.phase == 2:
            # Randomised layout, no stochastic noise during steps
            self.current_location   = (0, 0)
            self.pending_deliveries = random.sample(GRID_CELLS, k=3)
            self.deadlines          = sorted(random.sample(range(20, 80), 3))
            self.fuel               = random.uniform(0.7, 1.0) * self.max_fuel
            self.traffic_map        = [[random.randint(0, 1) for _ in range(GRID_SIZE)]
                                       for _ in range(GRID_SIZE)]

        else:  # phase == 3
            # Full stochasticity (original hard mode)
            self.current_location   = (0, 0)
            self.pending_deliveries = random.sample(GRID_CELLS, k=3)
            self.deadlines          = sorted(random.sample(range(20, 80), 3))
            self.fuel               = random.uniform(0.7, 1.0) * self.max_fuel
            self.traffic_map        = [[random.randint(0, 2) for _ in range(GRID_SIZE)]
                                       for _ in range(GRID_SIZE)]
            for (x, y) in self.hotspots:
                self.traffic_map[x][y] = 2

        self.done                 = False
        self.steps_taken          = 0
        self.total_fuel_used      = 0.0
        self.deliveries_completed = 0
        self.time_elapsed         = 0
        self._initial_delivery_count = len(self.pending_deliveries)
        self._prev_min_dist       = self._min_delivery_dist()

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def _min_delivery_dist(self):
        """Minimum Manhattan-distance to any pending delivery."""
        if not self.pending_deliveries:
            return 0.0
        return min(distance(self.current_location, d)
                   for d in self.pending_deliveries)

    # ------------------------------------------------------------------
    def _get_obs(self):
        x, y = self.current_location
        grid_size = GRID_SIZE
        max_dist  = 2 * grid_size  # normalization denominator

        delivery_dists = sorted(
            distance(self.current_location, d) for d in self.pending_deliveries
        )[:3]
        while len(delivery_dists) < 3:
            delivery_dists.append(0.0)

        d1, d2, d3 = delivery_dists

        nearest_delivery = d1
        nearest_fuel = min(
            distance(self.current_location, f) for f in self.fuel_stations
        )
        traffic = get_traffic(self.traffic_map, self.current_location)
        steps_left = max(0, (200 - self.steps_taken)) / 200

        obs = np.array([
            x / grid_size,
            y / grid_size,
            self.fuel / self.max_fuel,
            len(self.pending_deliveries) / 5,
            nearest_delivery / max_dist,
            nearest_fuel     / max_dist,
            traffic          / 2,
            d1               / max_dist,
            d2               / max_dist,
            d3               / max_dist,
            steps_left,
        ], dtype=np.float32)

        return np.clip(obs, 0.0, 1.0)

    # ------------------------------------------------------------------
    def state(self):
        return {
            "location":  self.current_location,
            "fuel":      self.fuel,
            "time":      self.time_elapsed,
            "pending":   self.pending_deliveries,
            "deadlines": self.deadlines,
        }

    def update_traffic(self):
        if self.phase == 1:
            return  # static map in phase 1
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                change = random.choice([-1, 0, 1])
                self.traffic_map[i][j] = min(2, max(0, self.traffic_map[i][j] + change))
        for (x, y) in self.hotspots:
            self.traffic_map[x][y] = 2

    # ------------------------------------------------------------------
    def _decode_action(self, action):
        """Map discrete int → (target_cell, is_refuel)."""
        av = int(action)
        x, y = self.current_location
        moves = {
            0: (x,     min(y + 1, GRID_SIZE - 1)),   # up
            1: (x,     max(y - 1, 0)),                # down
            2: (max(x - 1, 0), y),                    # left
            3: (min(x + 1, GRID_SIZE - 1), y),        # right
        }
        if av == 4:
            return self.current_location, True
        return moves[av], False

    # ------------------------------------------------------------------
    def step(self, action):
        if self.done:
            info = {
                "deliveries_completed": self.deliveries_completed,
                "fuel_used":            self.total_fuel_used,
                "steps_taken":          self.steps_taken,
            }
            return self._get_obs(), 0.0, True, False, info

        self.steps_taken += 1
        reward = -0.05  # small step cost — encourages efficiency without panic

        # Decode -------------------------------------------------------
        if isinstance(action, (int, np.integer, np.ndarray)):
            step_target, step_refuel = self._decode_action(action)
        else:                         # Action object from baseline.py
            step_refuel = action.refuel
            step_target = action.next_location

        # --- REFUEL ---------------------------------------------------
        if step_refuel:
            if self.current_location in self.fuel_stations:
                if self.fuel < self.max_fuel * 0.8:
                    # Only reward refueling when fuel was actually low
                    self.fuel = self.max_fuel
                    reward += 0.5
                    info = {"refueled": True}
                else:
                    # Already nearly full — wasting a step
                    self.fuel = self.max_fuel
                    reward -= 0.5
                    info = {"refueled": True, "warning": "unnecessary_refuel"}
            else:
                reward -= 1.0
                info = {"error": "not_at_station"}

            info.update({
                "deliveries_completed": self.deliveries_completed,
                "fuel_used":            self.total_fuel_used,
                "steps_taken":          self.steps_taken,
            })
            truncated = self.steps_taken >= 200
            if truncated:
                self.done = True
            return self._get_obs(), np.clip(reward, -10.0, 10.0), False, truncated, info

        # --- INVALID MOVE (wall bump) --------------------------------
        if step_target == self.current_location:
            reward -= 1.0
            info = {
                "error":                "invalid_move",
                "deliveries_completed": self.deliveries_completed,
                "fuel_used":            self.total_fuel_used,
                "steps_taken":          self.steps_taken,
            }
            truncated = self.steps_taken >= 200
            if truncated:
                self.done = True
            return self._get_obs(), np.clip(reward, -10.0, 10.0), False, truncated, info

        # --- MOVE -----------------------------------------------------
        traffic = get_traffic(self.traffic_map, step_target)
        if step_target in self.hotspots:
            traffic += 1
        if self.phase == 3:
            traffic += np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])

        dist      = distance(self.current_location, step_target)
        fuel_used = dist * (1 + 0.5 * traffic)
        if self.phase == 3:
            fuel_used *= np.random.uniform(0.9, 1.2)
        time_taken = dist * (1 + traffic)

        # --- OUT OF FUEL ----------------------------------------------
        if fuel_used > self.fuel:
            reward -= 5.0
            info = {
                "error":                "insufficient_fuel",
                "deliveries_completed": self.deliveries_completed,
                "fuel_used":            self.total_fuel_used,
                "steps_taken":          self.steps_taken,
            }
            self.done = True
            return self._get_obs(), np.clip(reward, -10.0, 10.0), True, False, info

        self.fuel            -= fuel_used
        self.total_fuel_used += fuel_used
        self.time_elapsed    += time_taken
        self.current_location = step_target

        # ---- DENSE SHAPING REWARDS -----------------------------------

        # 1. Distance-based progress reward (approach signal)
        curr_min_dist = self._min_delivery_dist()
        progress = self._prev_min_dist - curr_min_dist
        reward += progress * 2.0          # strong gradient toward deliveries
        self._prev_min_dist = curr_min_dist

        # 2. Fuel efficiency penalty (mild — don't over-punish movement)
        reward -= 0.1 * fuel_used

        # 3. Delivery completion reward (BIG — not clipped)
        if step_target in self.pending_deliveries:
            self.deliveries_completed += 1
            idx = self.pending_deliveries.index(step_target)
            self.pending_deliveries.pop(idx)
            self.deadlines.pop(idx)

            # Escalating reward: later deliveries are worth more
            reward += 8.0
            # Speed bonus: faster delivery = more reward
            reward += max(0, 3.0 - self.steps_taken * 0.05)
            # Progress bonus based on fraction completed
            frac = self.deliveries_completed / self._initial_delivery_count
            reward += frac * 5.0

            # Update prev_min_dist after removal
            self._prev_min_dist = self._min_delivery_dist()

        # 4. Cluster bonus (reward being near multiple deliveries)
        if is_clustered(step_target, self.pending_deliveries):
            reward += 0.3

        # 5. Low-fuel warning — encourage heading to station
        if self.fuel < 5.0 and self.pending_deliveries:
            nearest_station_dist = min(
                distance(self.current_location, s) for s in self.fuel_stations
            )
            if nearest_station_dist < self._prev_min_dist:
                reward += 0.2  # heading toward station is good when low

        # 6. COMPLETION BONUS (not clipped — full +25 flows through)
        terminated = False
        if len(self.pending_deliveries) == 0:
            reward += 25.0
            self.done = True
            terminated = True

        self.update_traffic()

        info = {
            "deliveries_completed": self.deliveries_completed,
            "fuel_used":            self.total_fuel_used,
            "steps_taken":          self.steps_taken,
        }

        truncated = self.steps_taken >= 200
        if truncated:
            # Penalize unfinished deliveries at timeout
            remaining = len(self.pending_deliveries)
            if remaining > 0:
                reward -= remaining * 3.0
            self.done = True

        # Clip per-step reward but allow large positive for completion
        reward = np.clip(reward, -10.0, 50.0)

        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "ansi":
            grid = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
            for s in self.fuel_stations:
                grid[s[1]][s[0]] = "F"
            for d in self.pending_deliveries:
                grid[d[1]][d[0]] = "D"
            ax, ay = self.current_location
            grid[ay][ax] = "A"
            header = f"Step: {self.steps_taken}  Fuel: {self.fuel:.1f}  Delivered: {self.deliveries_completed}/{self._initial_delivery_count}"
            rows = [header, "+" + "-" * GRID_SIZE + "+"]
            for row in reversed(grid):
                rows.append("|" + "".join(row) + "|")
            rows.append("+" + "-" * GRID_SIZE + "+")
            return "\n".join(rows)
        return None