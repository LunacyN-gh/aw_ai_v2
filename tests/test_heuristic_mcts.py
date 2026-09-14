import unittest
from unittest.mock import patch
from aw_ai.bundle_mcts import MCTSPlanner
from aw_ai.planner import Config
from aw_ai.scenarios import load,digest
from aw_ai.notation import GameRecord,loads


class HeuristicMCTSTests(unittest.TestCase):
    def test_heuristic_never_calls_neural_proposer_and_replay_is_legal(self):
        state=load('maps/6x6_ita/predeploy_6x6_ita.json')
        planner=MCTSPlanner(config=Config(seconds=.5,nodes=500))
        with patch('aw_ai.bundle_mcts.policy_turn',side_effect=AssertionError('Neural call')):
            analysis=planner.analyze(state)
        self.assertGreater(analysis.metrics.counts['mcts_completed_simulations'],0)
        self.assertLessEqual(analysis.metrics.counts['nodes'],500)
        self.assertAlmostEqual(sum(analysis.visit_weights.values()),1.)
        record=GameRecord(state)
        for action in analysis.actions:record.append(action)
        self.assertEqual(digest(loads(record.dumps()).states[-1]),digest(planner.execute(state,analysis)))

    def test_zero_budget_heuristic_has_complete_baseline(self):
        state=load('maps/6x6_ita/test_cities_6x6_ita.json')
        planner=MCTSPlanner(config=Config(seconds=0,nodes=0))
        analysis=planner.analyze(state)
        self.assertEqual(analysis.metrics.counts['mcts_fallback'],1)
        self.assertEqual(planner.execute(state,analysis).player,1-state.player)
