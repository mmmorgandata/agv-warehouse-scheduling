# Smart Warehouse AGV Planning Prototype

This repository explores a two-stage planning workflow for warehouse retrieval:

1. assign tasks with either a greedy nearest-task baseline or a centralized CBBA-style auction;
2. plan static-obstacle-aware routes with A* or Jump Point Search (JPS).

The project models 14 AGVs and 30 pickup-and-delivery tasks. It is a planning simulation, not a production fleet controller: routes are planned independently, so the code does not yet resolve AGV-to-AGV conflicts in space and time.

[Open the interactive dashboard](https://mmmorgandata.github.io/agv-warehouse-scheduling/)

## What the implementation covers

- per-agent task-count limits;
- reward-and-distance bidding in a shared winner table;
- eight-directional A* and JPS routing around static obstacles;
- route distance, makespan, fleet utilization, and planning-runtime metrics;
- a deterministic public benchmark that does not depend on private survey files.

The CBBA-style implementation is centralized. It borrows bundle bidding and winner resolution from CBBA, but it does not simulate peer-to-peer messages, fault tolerance, or a decentralized network. Battery usage and task windows are represented in the data model but are not enforced by the public benchmark.

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

\*One local run on Apple Silicon with Python 3.12; runtime should be remeasured on the target environment.

The comparison does not show a universal winner. Both planners return the same route length for a given assignment, as expected under the shared movement model. At this grid size, this Python JPS implementation is slower than A*. The CBBA-style assignment activates one fewer AGV but has a 6.8% longer makespan than the greedy baseline. These results are useful boundaries, not claims of production improvement.

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

## Limitations and next steps

- add a time-expanded reservation table or conflict-based search for AGV-to-AGV collision avoidance;
- enforce battery, payload, and task-window feasibility during assignment;
- replace the shared winner table with explicit peer-to-peer consensus if distributed behavior is required;
- benchmark multiple seeds, grid sizes, obstacle densities, and task loads;
- compare route-aware marginal bids instead of scoring each task only from the agent's initial position.

## References

1. Choi, H.-L., Brunet, L., & How, J. P. (2009). Consensus-Based Decentralized Auctions for Robust Task Allocation. *IEEE Transactions on Robotics*, 25(4), 912–926.
2. Harabor, D., & Grastien, A. (2011). Online Graph Pruning for Pathfinding on Grid Maps. *AAAI Conference on Artificial Intelligence*.
