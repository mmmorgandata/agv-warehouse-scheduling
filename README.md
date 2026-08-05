# Smart Warehouse AGV Scheduling

This repository contains a two-stage AGV scheduling and evaluation pipeline for warehouse retrieval:

1. assign tasks with either a greedy nearest-task baseline or a centralized CBBA-style auction;
2. plan static-obstacle-aware routes with A* or Jump Point Search (JPS).

The reproducible benchmark models 14 AGVs and 30 pickup-and-delivery tasks. Across the four tested configurations, **greedy nearest-task assignment with A*** achieves the shortest makespan and the lowest planning time.

[Open the interactive dashboard](https://mmmorgandata.github.io/agv-warehouse-scheduling/)

## What I built

- a two-stage pipeline connecting task assignment, grid routing, evaluation, and visualization;
- greedy nearest-task and centralized CBBA-style assignment with per-agent task limits;
- eight-directional A* and JPS routing around static obstacles;
- a deterministic public scenario that can run without the private survey spreadsheets;
- end-to-end metrics for task completion, route distance, makespan, active fleet size, and planning time.

CBBA, A*, and JPS are established algorithms rather than algorithms introduced by this project. The implementation adapts them into one comparable planning workflow. Its CBBA-style assignment uses bundle bidding and a shared winner table in a centralized simulation; it does not model peer-to-peer communication. Battery usage and task windows are represented in the data model but are not enforced by the public benchmark.

## Reproducible comparison

Run the 2×2 experiment:

```bash
python benchmark.py
```

It evaluates:

| Assignment | Path planner |
|---|---|
| Greedy nearest-task | A* |
| Greedy nearest-task | JPS |
| CBBA-style bidding | A* |
| CBBA-style bidding | JPS |

The public scenario uses seed 42, a 120×120 occupancy grid representing a 3,000×3,000 m area, 14 AGVs, 30 tasks, and three fixed obstacle walls with aisle openings. Results are written to `results/benchmark_results.csv` and `.json`.

### Current results

| Assignment | Planner | Tasks | Active AGVs | Route distance | Makespan | Planning runtime* |
|---|---:|---:|---:|---:|---:|---:|
| Greedy nearest | A* | 30/30 | 10 | 95,656 m | 16,184 s | ≈0.16 s |
| Greedy nearest | JPS | 30/30 | 10 | 95,656 m | 16,184 s | ≈0.22 s |
| CBBA-style | A* | 30/30 | 9 | 95,714 m | 17,286 s | ≈0.16 s |
| CBBA-style | JPS | 30/30 | 9 | 95,714 m | 17,286 s | ≈0.22 s |

\*The timing column comes from one Apple Silicon / Python 3.12 run. Route and makespan results are deterministic for this seed; wall-clock time will vary by machine.

For this scenario, **greedy nearest-task assignment with A*** is the strongest default: it completes all 30 tasks with the shortest makespan and the lowest planning time. CBBA-style assignment becomes interesting only when reducing the active fleet matters more than finishing early—it uses one fewer AGV, but increases makespan by 6.8%. JPS produces the same routes as A* and is slightly slower at this scale, so A* remains the practical choice unless larger or more open maps reveal a clearer pruning advantage.

## What the benchmark revealed

The current CBBA-style bid is based on distance from an AGV's starting position, so it can underestimate the cost of inserting another task into an existing route. A route-aware marginal bid would address the weakness exposed by the benchmark.

Collision avoidance is a separate problem. The current planners find spatial paths independently; a deployable multi-AGV system would also need to coordinate when each vehicle enters a shared aisle.

## Original scenario and private data

The original university-industry study used logistics-park survey spreadsheets that are not distributed in this repository. `main.py` attempts to load those files and otherwise falls back to a seeded synthetic layout. Historical figures in the dashboard describe the original study and should not be interpreted as regenerated public-benchmark results.

Run the original-style simulation with:

```bash
python main.py
```

## Project structure

```text
├── astar.py             # A* baseline
├── benchmark.py         # deterministic 2×2 comparison
├── cbba.py              # centralized CBBA-style assignment
├── data_loader.py       # scenario construction and optional survey loader
├── jps.py               # Jump Point Search planner
├── main.py              # original-style simulation and plots
├── models.py            # AGV and task data classes
├── tests/               # benchmark checks
└── results/
    ├── benchmark_results.csv
    ├── benchmark_results.json
    └── dashboard.html
```

## Setup and verification

```bash
python -m pip install matplotlib numpy openpyxl
python benchmark.py
python -m unittest discover -s tests -v
```

## References

1. Choi, H.-L., Brunet, L., & How, J. P. (2009). Consensus-Based Decentralized Auctions for Robust Task Allocation. *IEEE Transactions on Robotics*, 25(4), 912–926.
2. Harabor, D., & Grastien, A. (2011). Online Graph Pruning for Pathfinding on Grid Maps. *AAAI Conference on Artificial Intelligence*.
