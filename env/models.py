from pydantic import BaseModel
from typing import List, Tuple

class Observation(BaseModel):
    current_location: Tuple[int, int]
    pending_deliveries: List[Tuple[int, int]]

    fuel: float
    max_fuel: float

    time_elapsed: float

    deadlines: List[float]

    traffic_map: List[List[int]]  # 0 low, 1 medium, 2 high

class Action(BaseModel):
    next_location: Tuple[int, int]
    refuel: bool = False

class Reward(BaseModel):
    value: float