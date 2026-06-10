"""Data classes for AGV agents and tasks."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AGV:
    """Represents an AGV (Automated Guided Vehicle)."""
    agent_id: int
    x: float
    y: float
    agent_type: int = 0          # 0 = car01, 1 = car02
    nom_velocity: float = 1.0    # m/s
    battery: float = 100.0       # percentage (0–100)
    max_tasks: int = 4           # upper limit UL
    availability: float = 0.0   # earliest time agent is free
    max_payload: float = 1500.0  # kg (high-rack forklift capacity)

    # battery drain per unit distance
    bpm: float = 0.01
    battery_threshold: float = 30.0  # return to charge below this %

    def battery_ok(self, extra_distance: float) -> bool:
        return self.battery - self.bpm * extra_distance >= self.battery_threshold


@dataclass
class Task:
    """Represents a cargo-retrieval task."""
    task_id: int
    x: float              # shelf / pickup location
    y: float
    task_type: int        # 0 = type-1 (heavy, value 100), 1 = type-2 (light, value 80)
    start_time: float = 0.0
    end_time: float = 900.0
    duration: float = 55.0    # service time at shelf (seconds)
    task_value: float = 100.0
    discount: float = 0.0002  # spatial discount factor (per metre)
    weight: float = 500.0     # cargo weight (kg)

    # destination: unloading dock
    dest_x: float = 1677.0
    dest_y: float = 425.0


# Task-type presets
TASK_TYPES = {
    0: {"duration": 55.0, "task_value": 100.0},
    1: {"duration": 30.0, "task_value": 80.0},
}
