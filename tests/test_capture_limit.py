import unittest
from aw_ai.model import Board, State, Unit, Action, INCOME_TILES
from aw_ai.rules import Rules
from aw_ai.scenarios import to_data, from_data, digest, league_scale
from aw_ai.arena import play, replay, terminal_samples
from aw_ai.planner import Planner, Config


class CaptureLimitTests(unittest.TestCase):
    def position(self):
        b=Board.plain(7,2,{(0,0):'city',(1,0):'factory',(2,0):'airport',
                           (3,0):'city',(4,0):'hq',(5,0):'city',(6,0):'tower'})
        return State(b,{'cap':Unit('cap',0,'infantry',3),'enemy':Unit('enemy',1,'tank',12)},
                     owners={0:0,1:0,2:0,4:1,6:0},captures={3:('cap',10)},income_capture_limit=4)

    def test_counts_all_income_properties_but_not_towers(self):
        s=self.position(); r=Rules()
        self.assertIsNone(r.outcome(s))
        result=r.apply(s,Action('capture','cap',3))
        self.assertEqual(result.winner,0)
        self.assertEqual(r.outcome(result),0)
        self.assertIn('enemy',result.units)
        self.assertEqual(result.owners[4],1)

    def test_disabled_limit_does_not_end_game(self):
        s=self.position(); s.income_capture_limit=None
        self.assertIsNone(Rules().outcome(Rules().apply(s,Action('capture','cap',3))))

    def test_key_clone_serialization_and_legacy_defaults(self):
        s=self.position(); self.assertEqual(s.clone().income_capture_limit,4)
        data=to_data(s); self.assertEqual(digest(from_data(data)),digest(s))
        old=s.clone(); old.income_capture_limit=None
        self.assertNotEqual(s.key(),old.key())
        self.assertNotIn('income_capture_limit',to_data(old))
        self.assertIsNone(from_data(to_data(old)).income_capture_limit)

    def test_real_terminal_labels_and_replay(self):
        s=self.position(); planners=[Planner(config=Config(seconds=5)),Planner()]
        record,final=play(s,planners,max_turns=1)
        self.assertEqual(record['termination'],'victory')
        self.assertEqual(record['result'],0)
        self.assertTrue(terminal_samples([record]))
        self.assertEqual(digest(replay(record)),digest(final))

    def test_limit_validation_and_training_default(self):
        from aw_ai.model import INCOME_TILES
        s=self.position()
        for value in (0,3,7,4.5,True):
            s.income_capture_limit=value
            with self.assertRaises(ValueError): s.validate()
        # Pure rules fixture; neural dependency is optional for the core tests.
        b=league_scale()
        count=sum(t in INCOME_TILES for t in b.board.tiles)
        self.assertEqual(count,42)
        self.assertEqual((count*2+2)//3,28)
