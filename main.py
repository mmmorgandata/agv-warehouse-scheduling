"""
Smart-Warehouse AGV Two-Stage Scheduling Simulation
====================================================
Stage 1 – Multi-objective task assignment via improved CBBA
Stage 2 – Obstacle-aware path planning via JPS

Usage:
    python main.py [--no-plots]

Outputs:
    assignment_result.txt   – per-AGV task lists + KPIs
    path_result.txt         – per-AGV JPS waypoints
    task_assignment.png     – 2-D scatter of agents and tasks
    agv_paths.png           – all AGV routes on the warehouse grid
    jps_vs_astar_bar.png    – algorithmic comparison (replicated from paper)
"""
import argparse
import math
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Local modules ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from models import AGV, Task
from cbba import CBBA
from jps import JPSPlanner, path_length
from data_loader import (
    build_agents, build_tasks, build_obstacle_grid,
    load_facility_coords, facility_centroid,
    world_to_grid, grid_to_world,
)

# ── Simulation parameters (from the paper §5.2) ────────────────────────
WORLD_SIZE   = 3000          # simulated warehouse side length (m)
NUM_AGENTS   = 14
NUM_TASKS    = 30
START_POS    = (1677.0, 425.0)   # charging-zone centroid
TIME_WINDOW  = (0.0, 900.0)     # global task time window (s)
VELOCITY     = 1.0              # m/s
BATTERY      = 100.0
BPM          = 0.01            # battery drain per metre
MAX_TASKS    = 4               # upper limit UL per AGV
GRID_SIZE    = 300             # JPS grid resolution


def run_stage1(agents, tasks):
    """Phase 1: CBBA task assignment."""
    print("\n── Stage 1: CBBA task assignment ─────────────────────────")
    t0 = time.time()
    cbba = CBBA(agents, tasks, max_depth=MAX_TASKS, time_window=TIME_WINDOW,
                use_time_window=False)
    cbba.run(max_iter=300)
    elapsed = time.time() - t0

    summary = cbba.assignment_summary()
    f1 = cbba.total_distance()
    f2 = cbba.total_time()
    f3 = cbba.total_reward()

    print(f"  Elapsed:          {elapsed:.2f} s")
    print(f"  f1 total distance: {f1:.1f} m")
    print(f"  f2 total time:     {f2:.1f} s")
    print(f"  f3 total reward:   {f3:.1f}")
    print()
    for ag_id, task_ids in summary.items():
        names = [f"T{t}" for t in task_ids]
        print(f"  AGV {ag_id:2d}: {names}")

    return cbba, summary


def run_stage2(agents, tasks, summary, grid):
    """Phase 2: JPS path planning for each AGV according to its assignment."""
    print("\n── Stage 2: JPS path planning ────────────────────────────")
    planner = JPSPlanner(grid)
    all_paths: dict[int, list[list[tuple[int, int]]]] = {}

    start_grid = world_to_grid(*START_POS, GRID_SIZE)
    total_jps_time = 0.0

    for ag_id, task_ids in summary.items():
        agent = agents[ag_id]
        prev_grid = world_to_grid(agent.x, agent.y, GRID_SIZE)
        ag_paths = []

        for t_id in task_ids:
            task = tasks[t_id]
            shelf_grid = world_to_grid(task.x, task.y, GRID_SIZE)
            dock_grid  = world_to_grid(task.dest_x, task.dest_y, GRID_SIZE)

            t0 = time.time()
            path_to_shelf = planner.find_path(prev_grid, shelf_grid)
            path_to_dock  = planner.find_path(shelf_grid, dock_grid)
            total_jps_time += time.time() - t0

            ag_paths.append((t_id, path_to_shelf, path_to_dock))
            prev_grid = dock_grid

        # Return to charging station
        path_home = planner.find_path(prev_grid, start_grid)
        ag_paths.append((-1, path_home, []))
        all_paths[ag_id] = ag_paths

    print(f"  Total JPS time: {total_jps_time:.3f} s")
    return all_paths, total_jps_time


def save_results(summary, all_paths, agents, tasks, f1, f2, f3):
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    with open(out / "assignment_result.txt", "w") as f:
        f.write("=== AGV Task Assignment (CBBA) ===\n")
        f.write(f"f1 total distance : {f1:.1f} m\n")
        f.write(f"f2 total time     : {f2:.1f} s\n")
        f.write(f"f3 total reward   : {f3:.1f}\n\n")
        for ag_id, task_ids in summary.items():
            f.write(f"AGV {ag_id}: tasks {task_ids}\n")
            for t_id in task_ids:
                t = tasks[t_id]
                f.write(f"  Task {t_id}: type={t.task_type} "
                        f"shelf=({t.x:.0f},{t.y:.0f}) "
                        f"value={t.task_value} dur={t.duration}s\n")

    with open(out / "path_result.txt", "w") as f:
        f.write("=== AGV Paths (JPS) ===\n")
        for ag_id, ag_paths in all_paths.items():
            f.write(f"\nAGV {ag_id}:\n")
            for t_id, path1, path2 in ag_paths:
                label = f"T{t_id}" if t_id >= 0 else "HOME"
                f.write(f"  {label} → shelf: {len(path1)} waypoints,"
                        f" length={path_length(path1):.1f}\n")
                if path2:
                    f.write(f"  {label} → dock : {len(path2)} waypoints,"
                            f" length={path_length(path2):.1f}\n")

    print(f"\n  Results saved to {out}/")


# ── Plotting ───────────────────────────────────────────────────────────

def plot_task_assignment(agents, tasks, summary, facilities, out_dir):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, WORLD_SIZE)
    ax.set_ylim(0, WORLD_SIZE)
    ax.set_title("AGV Task Assignment (CBBA)", fontsize=14)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Draw building footprints
    colors_bld = {"德邦、中通库房": "#c8e6c9", "顺丰库房": "#bbdefb", "云仓E": "#ffe0b2"}
    for name, pts in facilities.items():
        if len(pts) < 3:
            continue
        color = colors_bld.get(name, "#e0e0e0")
        poly = plt.Polygon(pts, closed=True, facecolor=color, edgecolor="grey",
                           linewidth=0.8, alpha=0.6)
        ax.add_patch(poly)
        cx, cy = facility_centroid(pts)
        ax.text(cx, cy, name, fontsize=6, ha="center", va="center", color="#333")

    # Charging station
    ax.scatter(*START_POS, s=200, marker="*", c="gold", zorder=5, label="Charging station")

    # Tasks
    colors_task = {0: "red", 1: "blue"}
    for t in tasks:
        c = colors_task.get(t.task_type, "grey")
        ax.scatter(t.x, t.y, s=60, c=c, marker="x", zorder=4)
        ax.text(t.x + 15, t.y + 15, f"T{t.task_id}", fontsize=6)

    # Agents and assignment arrows
    cmap = plt.cm.tab20
    for ag_id, task_ids in summary.items():
        agent = agents[ag_id]
        color = cmap(ag_id / NUM_AGENTS)
        ax.scatter(agent.x, agent.y, s=80, c=[color], marker="o", zorder=5)
        ax.text(agent.x + 15, agent.y + 15, f"A{ag_id}", fontsize=7, color=color)
        prev_x, prev_y = agent.x, agent.y
        for t_id in task_ids:
            t = tasks[t_id]
            ax.annotate("", xy=(t.x, t.y), xytext=(prev_x, prev_y),
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
            prev_x, prev_y = t.x, t.y

    legend_elems = [
        mpatches.Patch(facecolor="red",  label="Task type 1 (value=100)"),
        mpatches.Patch(facecolor="blue", label="Task type 2 (value=80)"),
        plt.Line2D([0], [0], marker="*", color="gold", ms=12, label="Charging station"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "task_assignment.png", dpi=150)
    plt.close(fig)
    print("  Saved task_assignment.png")


def plot_agv_paths(agents, tasks, all_paths, grid, out_dir):
    grid_arr = np.array(grid)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid_arr, cmap="Greys", origin="lower",
              extent=[0, GRID_SIZE, 0, GRID_SIZE], alpha=0.4)
    ax.set_title("AGV Planned Routes (JPS)", fontsize=14)
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")

    cmap = plt.cm.tab20
    for ag_id, ag_paths in all_paths.items():
        color = cmap(ag_id / NUM_AGENTS)
        for t_id, path1, path2 in ag_paths:
            for path in (path1, path2):
                if len(path) >= 2:
                    xs = [p[0] for p in path]
                    ys = [p[1] for p in path]
                    ax.plot(xs, ys, color=color, linewidth=0.7, alpha=0.8)

    # Mark start
    sx, sy = world_to_grid(*START_POS, GRID_SIZE)
    ax.scatter(sx, sy, s=200, marker="*", c="gold", zorder=5)

    # Mark task shelf positions
    for t in tasks:
        gx, gy = world_to_grid(t.x, t.y, GRID_SIZE)
        c = "red" if t.task_type == 0 else "blue"
        ax.scatter(gx, gy, s=30, c=c, marker="x", zorder=4)

    fig.tight_layout()
    fig.savefig(out_dir / "agv_paths.png", dpi=150)
    plt.close(fig)
    print("  Saved agv_paths.png")


def plot_algorithm_comparison(jps_time: float, out_dir):
    """Reproduce the A* vs JPS comparison table from the paper as a bar chart."""
    labels = ["A*", "JPS"]
    memory_gb = [13.24, 6.42]
    runtime_s = [6532, 1458]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("A* vs JPS Algorithm Comparison (single task)", fontsize=13)

    ax1.bar(labels, memory_gb, color=["#ef9a9a", "#90caf9"])
    ax1.set_title("Memory Usage (GB)")
    ax1.set_ylabel("GB")
    for i, v in enumerate(memory_gb):
        ax1.text(i, v + 0.2, f"{v}", ha="center")

    ax2.bar(labels, runtime_s, color=["#ef9a9a", "#90caf9"])
    ax2.set_title("Runtime (s)")
    ax2.set_ylabel("Seconds")
    for i, v in enumerate(runtime_s):
        ax2.text(i, v + 50, f"{v}", ha="center")

    fig.tight_layout()
    fig.savefig(out_dir / "jps_vs_astar_bar.png", dpi=150)
    plt.close(fig)
    print("  Saved jps_vs_astar_bar.png")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    print("=" * 60)
    print("Smart-Warehouse AGV Two-Stage Scheduling Simulation")
    print("=" * 60)

    # Load real data
    print("\nLoading facility coordinate data …")
    try:
        facilities = load_facility_coords()
        print(f"  Loaded {len(facilities)} facilities.")
    except Exception as e:
        print(f"  Warning: could not load facility data ({e}). Using random layout.")
        facilities = {}

    # Build inputs
    agents = build_agents(
        n=NUM_AGENTS, start=START_POS, velocity=VELOCITY,
        battery=BATTERY, max_tasks=MAX_TASKS
    )
    tasks = build_tasks(
        num_tasks=NUM_TASKS, facilities=facilities,
        dest=START_POS, time_window=TIME_WINDOW
    )
    grid = build_obstacle_grid(size=GRID_SIZE, facilities=facilities)

    print(f"\n  Agents: {len(agents)}")
    print(f"  Tasks:  {len(tasks)}")
    print(f"  Grid:   {GRID_SIZE}×{GRID_SIZE}")

    # Stage 1 – CBBA
    cbba, summary = run_stage1(agents, tasks)
    f1 = cbba.total_distance()
    f2 = cbba.total_time()
    f3 = cbba.total_reward()

    # Stage 2 – JPS
    all_paths, jps_time = run_stage2(agents, tasks, summary, grid)

    # Save text results
    save_results(summary, all_paths, agents, tasks, f1, f2, f3)

    # Plots
    if not args.no_plots:
        out_dir = Path(__file__).parent / "results"
        out_dir.mkdir(exist_ok=True)
        print("\n── Generating plots ──────────────────────────────────────")
        plot_task_assignment(agents, tasks, summary, facilities, out_dir)
        plot_agv_paths(agents, tasks, all_paths, grid, out_dir)
        plot_algorithm_comparison(jps_time, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
