"""
Load and preprocess logistics-park data from the survey Excel files.

Data source:
    物流园手动坐标【22-11-10】.xlsx   ← facility polygons
    物流园调查数据【22-11-07】.xlsx   ← park survey (14 forklifts, etc.)

The park coordinates are in an AutoCAD coordinate system; we normalise them
to a 3000 × 3000 grid (as used in the paper's simulation).
"""
import math
import random
from pathlib import Path

import openpyxl

from models import AGV, Task, TASK_TYPES

# ── Paths ─────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent.parent / "课程大作业-智慧园区" / "数据"
_COORD_FILE = _DATA_DIR / "物流园手动坐标【22-11-10】.xlsx"


# ── Coordinate normalisation ───────────────────────────────────────────
# The hand-entered coordinates span roughly x∈[0, 4400], y∈[0, 3824].
# We scale them to the simulation space [0, 3000] × [0, 3000].
_WORLD = 3000.0
_RAW_X_MAX = 4400.0
_RAW_Y_MAX = 3824.0


def _norm(x: float, y: float) -> tuple[float, float]:
    return x / _RAW_X_MAX * _WORLD, y / _RAW_Y_MAX * _WORLD


def load_facility_coords() -> dict[str, list[tuple[float, float]]]:
    """Return {facility_name: [(x, y), ...]} in normalised simulation coordinates."""
    wb = openpyxl.load_workbook(_COORD_FILE)
    ws = wb["Sheet1"]

    facilities: dict[str, list[tuple[float, float]]] = {}
    current: str = ""

    for row in ws.iter_rows(min_row=2, values_only=True):
        name, rx, ry = row[0], row[1], row[2]
        if name and isinstance(name, str) and name not in ("设施",):
            current = name
            facilities.setdefault(current, [])
        if rx is not None and ry is not None and isinstance(rx, (int, float)):
            if current:
                facilities[current].append(_norm(float(rx), float(ry)))

    return facilities


def facility_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return sum(xs) / len(xs), sum(ys) / len(ys)


# ── Public: build simulation inputs ────────────────────────────────────

def build_agents(
    n: int = 14,
    start: tuple[float, float] = (1677.0, 425.0),
    velocity: float = 1.0,
    battery: float = 100.0,
    max_tasks: int = 4,
    seed: int = 42,
) -> list[AGV]:
    """Create n AGVs starting at the charging zone.

    The paper places all AGVs at the centroid of the charging area (1677, 425).
    Two agent types (car01 and car02) are used, alternating.
    """
    rng = random.Random(seed)
    agents = []
    for i in range(n):
        # Slight position jitter so AGVs don't stack exactly
        jitter_x = rng.uniform(-10, 10)
        jitter_y = rng.uniform(-10, 10)
        agents.append(AGV(
            agent_id=i,
            x=start[0] + jitter_x,
            y=start[1] + jitter_y,
            agent_type=i % 2,
            nom_velocity=velocity,
            battery=battery,
            max_tasks=max_tasks,
        ))
    return agents


def build_tasks(
    num_tasks: int = 30,
    facilities: dict | None = None,
    dest: tuple[float, float] = (1677.0, 425.0),
    time_window: tuple[float, float] = (0.0, 900.0),
    seed: int = 42,
) -> list[Task]:
    """Generate tasks whose pickup locations lie on known shelf facilities.

    Task type assignment:
        type 0 (heavy goods, value 100, duration 55 s) → 50 %
        type 1 (light goods, value 80,  duration 30 s) → 50 %
    """
    if facilities is None:
        try:
            facilities = load_facility_coords()
        except Exception:
            facilities = {}

    # Collect all known shelf/warehouse polygon vertices as candidate locations
    shelf_points: list[tuple[float, float]] = []
    for name, pts in facilities.items():
        if pts:
            shelf_points.extend(pts)

    rng = random.Random(seed)
    if not shelf_points:
        # Fallback: random positions in the warehouse space
        shelf_points = [(rng.uniform(200, 2800), rng.uniform(200, 2800)) for _ in range(200)]

    tasks = []
    for i in range(num_tasks):
        px, py = rng.choice(shelf_points)
        t_type = i % 2
        preset = TASK_TYPES[t_type]
        tasks.append(Task(
            task_id=i,
            x=px,
            y=py,
            task_type=t_type,
            start_time=time_window[0],
            end_time=time_window[1],
            duration=preset["duration"],
            task_value=preset["task_value"],
            dest_x=dest[0],
            dest_y=dest[1],
        ))
    return tasks


def build_obstacle_grid(
    size: int = 300,
    facilities: dict | None = None,
) -> list[list[int]]:
    """Build a coarse occupancy grid (0=free, 1=obstacle) for JPS.

    We mark the interiors of warehouse buildings as obstacles so AGVs
    navigate *around* them, only approaching via aisle entry points.

    The grid is size×size; each cell = (3000/size) real metres.
    """
    grid = [[0] * size for _ in range(size)]
    cell = _WORLD / size

    if facilities is None:
        try:
            facilities = load_facility_coords()
        except Exception:
            facilities = {}

    # Mark building footprints as obstacles
    obstacle_names = [
        "德邦、中通库房",
        "顺丰库房",
        "云仓E",
    ]
    for name in obstacle_names:
        pts = facilities.get(name, [])
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        c_x0 = max(0, int(x_min / cell))
        c_x1 = min(size - 1, int(x_max / cell))
        c_y0 = max(0, int(y_min / cell))
        c_y1 = min(size - 1, int(y_max / cell))
        for cy in range(c_y0, c_y1 + 1):
            for cx in range(c_x0, c_x1 + 1):
                grid[cy][cx] = 1

    return grid


def world_to_grid(
    wx: float, wy: float, size: int = 300
) -> tuple[int, int]:
    """Convert real-world coordinates to grid cell (col, row)."""
    cell = _WORLD / size
    return int(wx / cell), int(wy / cell)


def grid_to_world(
    cx: int, cy: int, size: int = 300
) -> tuple[float, float]:
    cell = _WORLD / size
    return cx * cell + cell / 2, cy * cell + cell / 2
