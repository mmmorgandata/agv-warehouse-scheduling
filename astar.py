"""A* path planning on an eight-connected occupancy grid."""

import heapq
import math


class AStarPlanner:
    def __init__(self, grid: list[list[int]]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows else 0

    def find_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        if not self._walkable(*start) or not self._walkable(*goal):
            return []
        if start == goal:
            return [start]

        queue = [(self._heuristic(*start, *goal), 0.0, start)]
        came_from = {start: None}
        best_cost = {start: 0.0}

        while queue:
            _, cost, current = heapq.heappop(queue)
            if cost > best_cost[current]:
                continue
            if current == goal:
                return self._reconstruct(came_from, current)

            for neighbour, step_cost in self._neighbours(*current):
                candidate = cost + step_cost
                if candidate >= best_cost.get(neighbour, math.inf):
                    continue
                best_cost[neighbour] = candidate
                came_from[neighbour] = current
                priority = candidate + self._heuristic(*neighbour, *goal)
                heapq.heappush(queue, (priority, candidate, neighbour))

        return []

    def _neighbours(self, x: int, y: int):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if not self._walkable(nx, ny):
                    continue
                yield (nx, ny), math.sqrt(2) if dx and dy else 1.0

    def _walkable(self, x: int, y: int) -> bool:
        return 0 <= x < self.cols and 0 <= y < self.rows and self.grid[y][x] == 0

    @staticmethod
    def _heuristic(x: int, y: int, gx: int, gy: int) -> float:
        dx, dy = abs(gx - x), abs(gy - y)
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    @staticmethod
    def _reconstruct(came_from, current):
        path = []
        while current is not None:
            path.append(current)
            current = came_from[current]
        return list(reversed(path))
