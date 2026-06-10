"""
Jump Point Search (JPS) path planning on a 2-D grid.

JPS prunes the A* search space using forced-neighbour and jump-point rules,
giving the same optimal path length as A* but visiting far fewer nodes.
"""
import heapq
import math
from typing import Optional


class JPSPlanner:
    """JPS on an occupancy grid.

    Args:
        grid: 2-D list[list[int]] where 0 = free, 1 = obstacle.
              grid[row][col] → row=y, col=x.
    """

    def __init__(self, grid: list[list[int]]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def find_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Return the shortest path as a list of (col, row) waypoints, or []."""
        sx, sy = start
        gx, gy = goal
        if not self._walkable(sx, sy) or not self._walkable(gx, gy):
            return []
        if start == goal:
            return [start]

        open_heap: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, start))

        came_from: dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
        g_score: dict[tuple[int, int], float] = {start: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current == goal:
                return self._reconstruct(came_from, current)

            cx, cy = current
            parent = came_from[current]

            for neighbour in self._successors(cx, cy, parent, gx, gy):
                nx, ny = neighbour
                new_g = g_score[current] + self._dist(cx, cy, nx, ny)
                if neighbour not in g_score or new_g < g_score[neighbour]:
                    g_score[neighbour] = new_g
                    f = new_g + self._heuristic(nx, ny, gx, gy)
                    heapq.heappush(open_heap, (f, neighbour))
                    came_from[neighbour] = current
        return []

    # ------------------------------------------------------------------
    # JPS internals
    # ------------------------------------------------------------------

    def _successors(
        self,
        x: int, y: int,
        parent: Optional[tuple[int, int]],
        gx: int, gy: int,
    ) -> list[tuple[int, int]]:
        successors = []
        neighbours = self._natural_neighbours(x, y, parent)
        for nx, ny in neighbours:
            dx = self._sign(nx - x)
            dy = self._sign(ny - y)
            jp = self._jump(x, y, dx, dy, gx, gy)
            if jp is not None:
                successors.append(jp)
        return successors

    def _natural_neighbours(
        self,
        x: int, y: int,
        parent: Optional[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if parent is None:
            # No parent → expand all 8 directions
            return [
                (x + dx, y + dy)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                if (dx, dy) != (0, 0) and self._walkable(x + dx, y + dy)
            ]
        px, py = parent
        dx = self._sign(x - px)
        dy = self._sign(y - py)
        neighbours = []

        if dx != 0 and dy != 0:
            # Diagonal move
            if self._walkable(x, y + dy):
                neighbours.append((x, y + dy))
            if self._walkable(x + dx, y):
                neighbours.append((x + dx, y))
            if self._walkable(x + dx, y + dy):
                neighbours.append((x + dx, y + dy))
            # Forced neighbours
            if not self._walkable(x - dx, y) and self._walkable(x - dx, y + dy):
                neighbours.append((x - dx, y + dy))
            if not self._walkable(x, y - dy) and self._walkable(x + dx, y - dy):
                neighbours.append((x + dx, y - dy))
        elif dx != 0:
            # Horizontal move
            if self._walkable(x + dx, y):
                neighbours.append((x + dx, y))
            if not self._walkable(x, y + 1) and self._walkable(x + dx, y + 1):
                neighbours.append((x + dx, y + 1))
            if not self._walkable(x, y - 1) and self._walkable(x + dx, y - 1):
                neighbours.append((x + dx, y - 1))
        else:
            # Vertical move
            if self._walkable(x, y + dy):
                neighbours.append((x, y + dy))
            if not self._walkable(x + 1, y) and self._walkable(x + 1, y + dy):
                neighbours.append((x + 1, y + dy))
            if not self._walkable(x - 1, y) and self._walkable(x - 1, y + dy):
                neighbours.append((x - 1, y + dy))
        return neighbours

    def _jump(
        self,
        x: int, y: int,
        dx: int, dy: int,
        gx: int, gy: int,
        depth: int = 0,
    ) -> Optional[tuple[int, int]]:
        nx, ny = x + dx, y + dy
        if not self._walkable(nx, ny):
            return None
        if nx == gx and ny == gy:
            return (nx, ny)

        if dx != 0 and dy != 0:
            # Diagonal: check for forced neighbours
            if (self._walkable(nx - dx, ny) and not self._walkable(nx - dx, ny - dy)) or \
               (self._walkable(nx, ny - dy) and not self._walkable(nx - dx, ny - dy)):
                return (nx, ny)
            # Recurse horizontally and vertically
            if self._jump(nx, ny, dx, 0, gx, gy) or self._jump(nx, ny, 0, dy, gx, gy):
                return (nx, ny)
        elif dx != 0:
            # Horizontal
            if (not self._walkable(nx, ny + 1) and self._walkable(nx + dx, ny + 1)) or \
               (not self._walkable(nx, ny - 1) and self._walkable(nx + dx, ny - 1)):
                return (nx, ny)
        else:
            # Vertical
            if (not self._walkable(nx + 1, ny) and self._walkable(nx + 1, ny + dy)) or \
               (not self._walkable(nx - 1, ny) and self._walkable(nx - 1, ny + dy)):
                return (nx, ny)

        if depth > 3000:
            return None
        return self._jump(nx, ny, dx, dy, gx, gy, depth + 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _walkable(self, x: int, y: int) -> bool:
        return 0 <= x < self.cols and 0 <= y < self.rows and self.grid[y][x] == 0

    @staticmethod
    def _sign(v: int) -> int:
        return (v > 0) - (v < 0)

    @staticmethod
    def _dist(x1: int, y1: int, x2: int, y2: int) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def _heuristic(x: int, y: int, gx: int, gy: int) -> float:
        # Octile distance (admissible for 8-directional movement)
        dx, dy = abs(gx - x), abs(gy - y)
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    @staticmethod
    def _reconstruct(
        came_from: dict,
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path = []
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path


def path_length(path: list[tuple[int, int]]) -> float:
    """Euclidean length of a waypoint path."""
    if len(path) < 2:
        return 0.0
    return sum(
        math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    )
