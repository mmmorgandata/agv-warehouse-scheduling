"""
Centralized simulation of CBBA-style bidding for multi-AGV task assignment.

Two-phase auction:
  Phase 1 – Task selection: each AGV greedily adds tasks to its bundle,
             inserting them at the position that maximises marginal gain.
  Phase 2 – Conflict resolution: agents share bids; highest bid wins.

The outer loop repeats until bundles stop changing.

The implementation borrows bundle construction and winner resolution from
CBBA, but uses a shared winner table rather than peer-to-peer communication.
"""
import copy
import math
from typing import Optional

from models import AGV, Task, TASK_TYPES


class CBBA:
    """Multi-AGV task assignment using CBBA-style bundle bidding.

    Args:
        agents:          list of AGV objects.
        tasks:           list of Task objects.
        max_depth:       bundle size upper limit (UL).
        time_window:     (t_start, t_end) global window in seconds.
        use_time_window: enforce per-task time-window feasibility.
    """

    def __init__(
        self,
        agents: list[AGV],
        tasks: list[Task],
        max_depth: int = 4,
        time_window: tuple[float, float] = (0.0, 900.0),
        use_time_window: bool = False,
    ):
        self.agents = agents
        self.tasks = tasks
        self.n = len(agents)
        self.m = len(tasks)
        self.max_depth = max_depth
        self.time_window = time_window
        self.use_time_window = use_time_window

        # path_list[i]  = ordered task IDs for agent i
        # bundle_list[i] = same set (tracks what agent committed to)
        self.bundle: list[list[int]] = [[] for _ in range(self.n)]
        self.path:   list[list[int]] = [[] for _ in range(self.n)]
        self.times:  list[list[float]] = [[] for _ in range(self.n)]

        # Shared winner table used by this centralized simulation.
        # winning_agent[j]  = agent index that won task j  (-1 = unclaimed)
        # winning_bid[j]    = winning bid for task j
        self.winning_agent: list[int]   = [-1] * self.m
        self.winning_bid:   list[float] = [0.0] * self.m

        # Each agent's local bid for each task (-1 = no bid placed)
        self.local_bid: list[list[float]] = [[-1.0] * self.m for _ in range(self.n)]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, max_iter: int = 300) -> list[list[int]]:
        """Run CBBA; return path[agent] – list of task IDs in assigned order."""
        for _ in range(max_iter):
            old_paths = [list(p) for p in self.path]

            # Phase 1: every agent greedily extends its bundle
            for i in range(self.n):
                self._build_bundle(i)

            # Phase 2: elect a winner from the shared bid table
            self._consensus()

            # Each agent drops tasks it did not win
            for i in range(self.n):
                self._release_lost_tasks(i)

            if all(self.path[i] == old_paths[i] for i in range(self.n)):
                break

        return copy.deepcopy(self.path)

    # ------------------------------------------------------------------
    # Phase 1 – greedy bundle construction for one agent
    # ------------------------------------------------------------------

    def _build_bundle(self, idx: int):
        """Extend agent idx's bundle one task at a time until full or no gain."""
        while len(self.bundle[idx]) < self.max_depth:
            best_task = -1
            best_score = -1.0      # must beat 0 to qualify
            best_pos = 0
            best_time = 0.0

            for j in range(self.m):
                # Skip tasks already claimed by this agent
                if j in self.bundle[idx]:
                    continue
                # Skip tasks where another agent is already the confirmed winner
                # and this agent cannot outbid them
                if self.winning_agent[j] != idx and self.winning_agent[j] != -1:
                    # Could we still outbid? Only if our bid exceeds the current winner.
                    # We'll compute the bid and check below.
                    pass

                bid, pos, t_start = self._marginal_bid(idx, j)
                if bid <= 0:
                    continue

                # Must exceed the globally known winning bid to take this task
                if bid <= self.winning_bid[j] and self.winning_agent[j] != idx:
                    continue

                if bid > best_score:
                    best_score = bid
                    best_task = j
                    best_pos = pos
                    best_time = t_start

            if best_task == -1:
                break  # nothing profitable or biddable

            # Record local bid
            self.local_bid[idx][best_task] = best_score

            # Tentatively claim this task if we outbid current winner
            if best_score > self.winning_bid[best_task]:
                self.winning_agent[best_task] = idx
                self.winning_bid[best_task] = best_score

            # Insert into path
            self.path[idx].insert(best_pos, best_task)
            self.times[idx].insert(best_pos, best_time)
            self.bundle[idx].append(best_task)

    def _marginal_bid(
        self, idx: int, task_id: int
    ) -> tuple[float, int, float]:
        """
        Find the best insertion position for task_id in agent idx's path.
        Returns (bid_score, insert_position, start_time); bid=0 if infeasible.
        """
        agent = self.agents[idx]
        task  = self.tasks[task_id]
        path  = self.path[idx]
        times = self.times[idx]

        best_bid = 0.0
        best_pos = 0
        best_time = 0.0

        for pos in range(len(path) + 1):
            # --- Previous task / agent start ---
            if pos == 0:
                prev_x, prev_y = agent.x, agent.y
                prev_done = agent.availability
            else:
                prev_t = self.tasks[path[pos - 1]]
                prev_x, prev_y = prev_t.x, prev_t.y
                prev_done = times[pos - 1] + prev_t.duration

            dt_to = math.hypot(task.x - prev_x, task.y - prev_y) / agent.nom_velocity
            min_start = max(task.start_time, prev_done + dt_to)

            # --- Next task / end ---
            if pos < len(path):
                next_t = self.tasks[path[pos]]
                dt_from = math.hypot(next_t.x - task.x, next_t.y - task.y) / agent.nom_velocity
                max_start = min(task.end_time, times[pos] - task.duration - dt_from)
            else:
                max_start = task.end_time

            # Feasibility
            if self.use_time_window and min_start > max_start:
                continue

            # --- Score ---
            dist_to_task = math.hypot(task.x - agent.x, task.y - agent.y)
            if self.use_time_window:
                elapsed = min_start - task.start_time
                reward = task.task_value * math.exp(-task.discount * elapsed)
            else:
                reward = task.task_value * math.exp(-task.discount * dist_to_task)

            penalty = agent.bpm * dist_to_task
            score = reward - penalty

            if score > best_bid:
                best_bid = score
                best_pos = pos
                best_time = min_start if self.use_time_window else max(task.start_time, prev_done + dt_to)

        return best_bid, best_pos, best_time

    # ------------------------------------------------------------------
    # Phase 2 – global consensus
    # ------------------------------------------------------------------

    def _consensus(self):
        """Elect the globally highest bidder for each task."""
        # Aggregate: for each task find the agent with the highest local bid
        for j in range(self.m):
            best_agent = -1
            best_bid = 0.0
            for i in range(self.n):
                if self.local_bid[i][j] > best_bid:
                    best_bid = self.local_bid[i][j]
                    best_agent = i
            self.winning_agent[j] = best_agent
            self.winning_bid[j] = best_bid

    # ------------------------------------------------------------------
    # Drop tasks not won by this agent
    # ------------------------------------------------------------------

    def _release_lost_tasks(self, idx: int):
        lost = [j for j in self.bundle[idx] if self.winning_agent[j] != idx]
        for j in lost:
            if j in self.path[idx]:
                k = self.path[idx].index(j)
                self.path[idx].pop(k)
                self.times[idx].pop(k)
            self.bundle[idx].remove(j)
            self.local_bid[idx][j] = -1.0
            # DO NOT reset winning_bid/winning_agent — that's global state

    # ------------------------------------------------------------------
    # Summary / KPI helpers
    # ------------------------------------------------------------------

    def assignment_summary(self) -> dict[int, list[int]]:
        return {self.agents[i].agent_id: list(self.path[i]) for i in range(self.n)}

    def total_distance(self) -> float:
        total = 0.0
        for i, agent in enumerate(self.agents):
            prev_x, prev_y = agent.x, agent.y
            for t_id in self.path[i]:
                t = self.tasks[t_id]
                total += math.hypot(t.x - prev_x, t.y - prev_y)
                total += math.hypot(t.dest_x - t.x, t.dest_y - t.y)
                prev_x, prev_y = t.dest_x, t.dest_y
        return total

    def total_time(self) -> float:
        total = 0.0
        for i, agent in enumerate(self.agents):
            t_acc, prev_x, prev_y = 0.0, agent.x, agent.y
            for t_id in self.path[i]:
                t = self.tasks[t_id]
                t_acc += math.hypot(t.x - prev_x, t.y - prev_y) / agent.nom_velocity
                t_acc += t.duration
                t_acc += math.hypot(t.dest_x - t.x, t.dest_y - t.y) / agent.nom_velocity
                prev_x, prev_y = t.dest_x, t.dest_y
            total += t_acc
        return total

    def total_reward(self) -> float:
        return sum(
            self.tasks[t_id].task_value
            for i in range(self.n)
            for t_id in self.path[i]
        )

    def tasks_assigned(self) -> set[int]:
        return {t_id for i in range(self.n) for t_id in self.path[i]}
