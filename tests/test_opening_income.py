import unittest
from pathlib import Path
from aw_ai.scenarios import load, from_data, to_data, league_scale
from aw_ai.model import END
from aw_ai.rules import Rules

class OpeningIncomeTests(unittest.TestCase):
    def test_first_turns_and_snapshot_roundtrip(self):
        path=Path(__file__).resolve().parents[1]/"maps/v1/test_cities_6x6.json"
        state=load(path)
        self.assertEqual(state.funds,[1000,2000])
        self.assertEqual(state.income_capture_limit,7)
        saved=from_data(to_data(state))
        self.assertEqual(saved.funds,[1000,2000])
        red=Rules().apply(saved,END)
        self.assertEqual(red.funds,[1000,3000])
        blue=Rules().apply(red,END)
        self.assertEqual(blue.funds,[2000,3000])
        self.assertEqual(load(path).funds,[1000,2000])

    def test_legacy_already_started_opt_out(self):
        import json
        path=Path(__file__).resolve().parents[1]/"maps/v1/test_cities_6x6.json"
        data=json.loads(path.read_text())
        data["turn_started"]=True
        self.assertEqual(from_data(data).funds,[0,2000])

    def test_generated_map_credits_only_current_player(self):
        state=league_scale()
        self.assertEqual(state.funds[0],12000+Rules().income(state,0))
        self.assertEqual(state.funds[1],12000)
