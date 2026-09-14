import unittest
from aw_ai.order_search import OrderPlanner, Prefix
from aw_ai.planner import Config
from aw_ai.model import Board, State, Unit, Action
from aw_ai.scenarios import load, digest
from aw_ai.budget import Budget
from aw_ai.notation import GameRecord,loads


class OrderTests(unittest.TestCase):
    def test_zero_budget_has_complete_legal_baseline_and_replay(self):
        state=load('maps/6x6_ita/predeploy_6x6_ita.json');key=state.key()
        planner=OrderPlanner(config=Config(seconds=0))
        analysis=planner.analyze(state);record=GameRecord(state)
        for action in analysis.actions:record.append(action)
        self.assertEqual(state.key(),key)
        self.assertEqual(digest(loads(record.dumps()).states[-1]),digest(planner.execute(state,analysis)))
        self.assertTrue(record.states[-1].player!=state.player or planner.rules.outcome(record.states[-1]) is not None)

    def test_factory_vacated_then_production_and_budget(self):
        state=load('maps/6x6_ita/predeploy_6x6_ita.json')
        planner=OrderPlanner(config=Config(seconds=2,nodes=60))
        analysis=planner.analyze(state)
        self.assertLessEqual(analysis.metrics.counts['nodes'],60)
        for c in analysis.alternatives:
            replay=state
            for action in c.actions:replay=planner.rules.apply(replay,action)
            self.assertEqual(replay.key(),c.state.key())

    def test_actor_order_can_open_blocked_destination(self):
        state=State(Board.plain(3,1),{'a':Unit('a',0,'infantry',0),
                    'b':Unit('b',0,'infantry',1),'e':Unit('e',1,'infantry',2)})
        planner=OrderPlanner();blocked=Action('wait','a',1)
        self.assertFalse(planner.rules.is_legal(state,blocked))
        # Moving b to the vacated a tile needs a to move first; occupancy is
        # recomputed after every real transition, not from a cached root list.
        state=State(Board.plain(4,2),{'a':Unit('a',0,'infantry',0),
                    'b':Unit('b',0,'infantry',1),'e':Unit('e',1,'infantry',7)})
        after=planner._step(Prefix(state),(Action('wait','b',2),None))
        self.assertTrue(planner.rules.is_legal(after.state,blocked))

    def test_commuting_orders_have_same_transposition_key(self):
        state=State(Board.plain(4,2),{'a':Unit('a',0,'infantry',0),
                    'b':Unit('b',0,'infantry',1),'e':Unit('e',1,'infantry',7)})
        planner=OrderPlanner();a=(Action('wait','a',4),None);b=(Action('wait','b',5),None)
        ab=planner._step(planner._step(Prefix(state),a),b)
        ba=planner._step(planner._step(Prefix(state),b),a)
        self.assertEqual(ab.state.key(),ba.state.key())

    def test_heuristic_breakthroughs(self):
        from aw_ai.tactical_suite import cases,Oracle
        for case in cases():
            if not case.name.startswith('breakthrough'):continue
            planner=OrderPlanner(config=Config(seconds=2))
            analysis=planner.analyze(case.state)
            self.assertTrue(Oracle().success(planner.execute(case.state,analysis),case),case.name)
