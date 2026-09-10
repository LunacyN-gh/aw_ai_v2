import unittest
from pathlib import Path
from aw_ai.model import Action, END, SPECS
from aw_ai.scenarios import load, to_data, from_data, digest
from aw_ai.rules import Rules
from aw_ai.planner import Planner, Config

MAP = Path(__file__).resolve().parents[1]/'maps/test_cities_6x6_ita.json'
ITA = ('infantry','tank','artillery')


class MapRosterTests(unittest.TestCase):
    def test_both_sides_rules_and_planner(self):
        rules = Rules()
        for side,base in ((0,0),(1,35)):
            state = load(MAP)
            state.player = side
            state.funds = [30000,30000]
            state.remove(state.occupancy[base])
            for kind in SPECS:
                action = Action('build',destination=base,build=kind)
                self.assertEqual(rules.is_legal(state,action),kind in ITA)
                if kind not in ITA:
                    with self.assertRaises(ValueError): rules.apply(state,action)
            self.assertEqual({a.build for a in rules.legal_actions(state) if a.kind=='build'},set(ITA))
            planner = Planner(config=Config(seconds=5,nodes=200))
            analysis = planner.analyze(state)
            for candidate in analysis.alternatives:
                self.assertTrue(all(a.build in ITA for a in candidate.actions if a.kind=='build'))
            self.assertEqual(planner.execute(state,analysis).allowed_builds,ITA)

    def test_roundtrip_clone_and_original_map(self):
        state = load(MAP)
        self.assertEqual(state.allowed_builds,ITA)
        self.assertEqual(state.funds,[1000,2000])
        self.assertEqual(Rules().apply(state,END).funds,[1000,3000])
        self.assertEqual(state.income_capture_limit,7)
        restored = from_data(to_data(state.clone()))
        self.assertEqual(restored.allowed_builds,ITA)
        self.assertEqual(digest(state),digest(restored))
        unrestricted = load(MAP.parent/'v1/test_cities_6x6.json')
        self.assertIsNone(unrestricted.allowed_builds)
        self.assertNotEqual(state.key(),unrestricted.key())
        state.allowed_builds=None
        self.assertEqual(digest(state),digest(unrestricted))

    def test_reject_invalid_roster_and_excluded_units(self):
        for roster in ([],['tank','tank'],['unknown'],['tank']):
            data=to_data(load(MAP)); data['allowed_builds']=roster
            with self.assertRaises(ValueError): from_data(data)

    def test_neural_choices_intersect_map_and_checkpoint(self):
        try:
            from aw_ai.neural_training import build_choices
            from aw_ai.neural import KINDS
        except ImportError:
            self.skipTest('optional torch unavailable')
        state=load(MAP); state.remove(state.occupancy[0]); state.funds[0]=30000
        self.assertEqual({KINDS[i] if i<len(KINDS) else 'save' for i in build_choices(state,Rules(),0,KINDS)},set(ITA)|{'save'})


if __name__ == '__main__': unittest.main()
