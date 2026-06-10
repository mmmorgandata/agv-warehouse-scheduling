# Smart Warehouse AGV Scheduling

A two-stage optimization framework for automated cargo retrieval in a smart warehouse, combining multi-objective task assignment with obstacle-aware path planning.

---

## Problem Background

Modern smart warehouses deploy fleets of **AGVs (Automated Guided Vehicles)** to replace manual labor in cargo retrieval tasks. Each outbound order requires an AGV to travel to a target shelf, carry the goods to an unloading dock, and return to its charging station — all while coordinating with other AGVs to avoid conflicts.

This creates a classic **multi-agent, multi-task scheduling problem** with three competing objectives:

| Objective | Description |
|-----------|-------------|
| **f₁** | Minimise total travel distance across all AGVs |
| **f₂** | Minimise total mission time across all AGVs |
| **f₃** | Maximise total task reward (weighted by cargo type and priority) |

Constraints include per-AGV task-count limits, battery capacity, payload weight, and time windows.

The problem is solved in **two decoupled stages**:

```
Orders received
      │
      ▼
┌─────────────────────────┐
│  Stage 1 – CBBA         │  Who does what?
│  Task Assignment        │  → Each AGV bids on tasks via
│                         │    a distributed auction
└────────────┬────────────┘
             │  Assignment + time windows
             ▼
┌─────────────────────────┐
│  Stage 2 – JPS          │  How to get there?
│  Path Planning          │  → Jump Point Search finds the
│                         │    shortest obstacle-free route
└────────────┬────────────┘
             │
             ▼
     AGV routes executed
```

---

## Methods

### Stage 1 — CBBA (Consensus-Based Bundle Algorithm)

CBBA is a **distributed auction algorithm** in which each AGV independently bids on tasks and resolves conflicts through peer-to-peer communication, converging on a conflict-free assignment without a central controller.

**Bid score for task *j* assigned to agent *i*:**

$$\text{score}_{ij} = v_j \cdot e^{-\lambda \cdot d_{ij}} - \mu \cdot d_{ij}$$

where $v_j$ is task value, $d_{ij}$ is the distance from agent $i$ to task $j$, $\lambda$ is a spatial discount factor, and $\mu$ is a battery cost coefficient.

Each agent greedily extends its task bundle by inserting the highest-marginal-value unassigned task at its optimal position in the sequence, until the bundle is full or no profitable task remains. A consensus round then elects the globally highest bidder for each task, after which losing agents release and re-bid.

### Stage 2 — JPS (Jump Point Search)

JPS is an **optimised grid search algorithm** that extends A\* by pruning symmetric paths using *forced-neighbour* and *jump-point* rules, dramatically reducing the number of nodes expanded.

| Algorithm | Memory | Runtime (single task) |
|-----------|--------|-----------------------|
| A\*       | 13.24 GB | 6,532 s |
| **JPS**   | **6.42 GB** | **1,458 s** |

JPS achieves **4.5× faster** planning with **51% less memory** while returning the same optimal path length.

---

## Simulation Setup

| Parameter | Value |
|-----------|-------|
| Warehouse grid | 3,000 × 3,000 m |
| AGV fleet size | 14 |
| Task count | 30 (15 heavy · 15 light) |
| AGV speed | 1 m/s |
| Max tasks per AGV | 4 |
| Battery threshold | 30% |
| Task types | Type 1: 55 s · 100 pts &nbsp;&nbsp; Type 2: 30 s · 80 pts |

Task pickup locations are drawn from the real polygon coordinates of warehouse buildings, extracted from field survey data. The charging station (AGV start and end point) is fixed at the centroid of the charging zone.

---

## Results

**CBBA assignment** — all 30 tasks assigned, 8 of 14 AGVs active:

| AGV | Tasks |
|-----|-------|
| AGV 2 | T25, T9, T27, T19 |
| AGV 3 | T17, T15, T29 |
| AGV 5 | T23, T11, T1, T12 |
| AGV 6 | T5, T7, T26, T18 |
| AGV 7 | T8, T24, T28 |
| AGV 8 | T10, T16, T6, T0 |
| AGV 10 | T4, T14, T2, T22 |
| AGV 12 | T13, T21, T3, T20 |

**KPIs:**

| Metric | Value |
|--------|-------|
| f₁ Total travel distance | 63,139 m |
| f₂ Total mission time | 64,414 s |
| f₃ Total reward | 2,700 pts |
| Tasks assigned | 30 / 30 |

---

## Interactive Dashboard

Open `results/dashboard.html` in any browser (no server needed):

- **Animated warehouse map** — AGVs move in real time, pulse when servicing a shelf, hover for tooltips
- **Gantt chart** — color-coded Travel / Service / Return blocks per agent with a live time cursor
- **A\* vs JPS comparison** — memory and runtime bar chart
- **KPI cards** — animated on load
- **AGV fleet panel** — click any row to highlight that agent's route

---

## Project Structure

```
├── models.py        # AGV and Task dataclasses
├── cbba.py          # Consensus-Based Bundle Algorithm
├── jps.py           # Jump Point Search path planner
├── data_loader.py   # Load survey data, build simulation inputs
├── main.py          # Main script (run to reproduce results)
└── results/
    ├── dashboard.html        # Interactive web dashboard
    ├── task_assignment.png   # 2-D assignment map
    ├── agv_paths.png         # AGV routes on obstacle grid
    └── jps_vs_astar_bar.png  # Algorithm comparison chart
```

## Usage

```bash
pip install matplotlib numpy openpyxl
python main.py
```

Results and plots are written to `results/`.

---

## References

1. Choi, H.-L., Brunet, L., & How, J. P. (2009). Consensus-Based Decentralized Auctions for Robust Task Allocation. *IEEE Transactions on Robotics*, 25(4), 912–926.
2. Harabor, D., & Grastien, A. (2011). Online Graph Pruning for Pathfinding on Grid Maps. *AAAI Conference on Artificial Intelligence*.
