import unittest

from aw_ai.planner import Config
from aw_ai.scenarios import tactical
from aw_ai.tree import SampledPlanner


class SampledTreeTests(unittest.TestCase):
    def test_sampled_tree_returns_legal_actions_without_mutation(self):
        p = SampledPlanner(config=Config(seconds=.2, nodes=180), seed=4)
        s = tactical("screen")
        key = s.key()
        analysis = p.analyze(s)
        p.execute(s, analysis).validate()
        self.assertEqual(s.key(), key)
        self.assertLessEqual(analysis.metrics.counts["nodes"], 180)
        self.assertGreater(analysis.metrics.counts["tree_iterations"], 0)

    def test_zero_budget_tree_returns_legal_turn(self):
        p = SampledPlanner(config=Config(seconds=0, nodes=0))
        s = tactical("screen")
        p.execute(s, p.analyze(s)).validate()


if __name__ == "__main__":
    unittest.main()
