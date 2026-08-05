import unittest

from benchmark import run_benchmark


class BenchmarkTest(unittest.TestCase):
    def test_all_four_combinations_complete_the_public_scenario(self):
        results = run_benchmark()
        self.assertEqual(len(results), 4)
        self.assertTrue(all(row["tasks_assigned"] == 30 for row in results))

    def test_astar_and_jps_return_the_same_route_distance(self):
        results = run_benchmark()
        by_assignment = {}
        for row in results:
            by_assignment.setdefault(row["assignment"], {})[row["planner"]] = row
        for planners in by_assignment.values():
            self.assertEqual(
                planners["A*"]["route_distance_m"],
                planners["JPS"]["route_distance_m"],
            )


if __name__ == "__main__":
    unittest.main()
