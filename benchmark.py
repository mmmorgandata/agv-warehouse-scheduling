"""Run assignment and path-planning baselines on one public scenario."""

import argparse
import csv
import json
import math
import time
from pathlib import Path

from astar import AStarPlanner
from cbba import CBBA
from data_loader import build_agents, build_tasks, world_to_grid
from jps import JPSPlanner, path_length


WORLD_SIZE = 3000.0
GRID_SIZE = 120
CELL_SIZE_M = WORLD_SIZE / GRID_SIZE
START_POS = (1677.0, 425.0)
MAX_TASKS = 4


def greedy_nearest_assignment(agents, tasks, max_tasks=MAX_TASKS):
    """Assign the nearest remaining task to the cheapest available agent."""
    assignments = {agent.agent_id: [] for agent in agents}
    positions = {agent.agent_id: (agent.x, agent.y) for agent in agents}
    remaining = set(range(len(tasks)))

    while remaining:
        choices = []
        for agent in agents:
            if len(assignments[agent.agent_id]) >= max_tasks:
                continue
            ax, ay = positions[agent.agent_id]
            for task_id in remaining:
                task = tasks[task_id]
                distance = math.hypot(task.x - ax, task.y - ay)
                distance += math.hypot(task.dest_x - task.x, task.dest_y - task.y)
                choices.append((distance, agent.agent_id, task_id))
        if not choices:
            break
        _, agent_id, task_id = min(choices)
        assignments[agent_id].append(task_id)
        task = tasks[task_id]
        positions[agent_id] = (task.dest_x, task.dest_y)
        remaining.remove(task_id)

    return assignments


def cbba_style_assignment(agents, tasks):
    model = CBBA(agents, tasks, max_depth=MAX_TASKS, use_time_window=False)
    model.run(max_iter=300)
    return model.assignment_summary()


def public_obstacle_grid(tasks):
    """Create a deterministic public grid without relying on private survey files."""
    grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    rectangles = [
        (25, 26, 8, 102),
        (55, 56, 18, 119),
        (85, 86, 0, 96),
    ]
    for x0, x1, y0, y1 in rectangles:
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                grid[y][x] = 1

    # Aisle openings keep the public scenario connected.
    for x, y0, y1 in ((25, 45, 52), (55, 74, 81), (85, 32, 39)):
        for y in range(y0, y1 + 1):
            grid[y][x] = 0
            grid[y][x + 1] = 0

    required = [world_to_grid(*START_POS, GRID_SIZE)]
    required.extend(world_to_grid(task.x, task.y, GRID_SIZE) for task in tasks)
    for x, y in required:
        for yy in range(max(0, y - 1), min(GRID_SIZE, y + 2)):
            for xx in range(max(0, x - 1), min(GRID_SIZE, x + 2)):
                grid[yy][xx] = 0
    return grid


def evaluate(assignments, planner_class, agents, tasks, grid):
    planner = planner_class(grid)
    started = time.perf_counter()
    total_distance_m = 0.0
    mission_times = []
    routes = 0

    for agent in agents:
        current = world_to_grid(agent.x, agent.y, GRID_SIZE)
        mission_time = 0.0
        for task_id in assignments[agent.agent_id]:
            task = tasks[task_id]
            pickup = world_to_grid(task.x, task.y, GRID_SIZE)
            dock = world_to_grid(task.dest_x, task.dest_y, GRID_SIZE)
            to_pickup = planner.find_path(current, pickup)
            to_dock = planner.find_path(pickup, dock)
            if not to_pickup or not to_dock:
                raise RuntimeError(
                    f"{planner_class.__name__} could not route task {task_id}"
                )
            distance_m = (path_length(to_pickup) + path_length(to_dock)) * CELL_SIZE_M
            total_distance_m += distance_m
            mission_time += distance_m / agent.nom_velocity + task.duration
            current = dock
            routes += 2
        mission_times.append(mission_time)

    return {
        "tasks_assigned": sum(map(len, assignments.values())),
        "active_agvs": sum(bool(task_ids) for task_ids in assignments.values()),
        "route_distance_m": round(total_distance_m, 1),
        "makespan_s": round(max(mission_times, default=0.0), 1),
        "planning_runtime_ms": round((time.perf_counter() - started) * 1000, 1),
        "routes_planned": routes,
    }


def run_benchmark():
    agents = build_agents(n=14, start=START_POS, max_tasks=MAX_TASKS, seed=42)
    tasks = build_tasks(num_tasks=30, facilities={}, dest=START_POS, seed=42)
    grid = public_obstacle_grid(tasks)
    assignments = {
        "Greedy nearest": greedy_nearest_assignment(agents, tasks),
        "CBBA-style": cbba_style_assignment(agents, tasks),
    }
    planners = {"A*": AStarPlanner, "JPS": JPSPlanner}

    results = []
    for assignment_name, assignment in assignments.items():
        for planner_name, planner_class in planners.items():
            metrics = evaluate(assignment, planner_class, agents, tasks, grid)
            results.append({
                "assignment": assignment_name,
                "planner": planner_name,
                **metrics,
            })
    return results


def save_results(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(results[0])
    with (output_dir / "benchmark_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    with (output_dir / "benchmark_results.json").open("w") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")


def print_results(results):
    columns = ("Assignment", "Planner", "Tasks", "Active", "Distance (m)", "Makespan (s)", "Plan (ms)")
    print(f"{columns[0]:<16} {columns[1]:<7} {columns[2]:>5} {columns[3]:>6} {columns[4]:>13} {columns[5]:>12} {columns[6]:>10}")
    for row in results:
        print(
            f"{row['assignment']:<16} {row['planner']:<7} "
            f"{row['tasks_assigned']:>5} {row['active_agvs']:>6} "
            f"{row['route_distance_m']:>13.1f} {row['makespan_s']:>12.1f} "
            f"{row['planning_runtime_ms']:>10.1f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    results = run_benchmark()
    save_results(results, args.output_dir)
    print_results(results)


if __name__ == "__main__":
    main()
