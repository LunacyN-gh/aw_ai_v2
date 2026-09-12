import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch
from aw_ai.gui import App
from aw_ai.scenarios import load
from aw_ai.model import Action,END


class FlipSidesTests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk();self.root.withdraw();self.app=App(self.root)
        self.app.reset(load(Path(__file__).resolve().parents[1]/'maps/test_cities_6x6_ita.json'))

    def tearDown(self):self.root.destroy()

    def test_opening_swap_and_ai_first_without_income_or_turn_advance(self):
        app=self.app;before=app.state.clone();initial=app.initial.key()
        app.selected=next(iter(before.units));app.destination=0;app.auto=True
        app.planners['stale']=object();app.alternatives.insert('end','stale')
        with patch.object(app,'start') as start:
            app.flip_sides();start.assert_called_once_with(True)
        self.assertEqual(app.human_side,1)
        self.assertEqual(app.state.key(),before.key())
        self.assertIn("Human Red",app.human_label.get())
        self.assertEqual(app.state.turn,before.turn)
        self.assertEqual(app.state.funds,before.funds)
        self.assertEqual(app.state.owners,before.owners)
        for uid,u in before.units.items():
            new=app.state.units[uid]
            self.assertEqual(new.owner,u.owner)
            self.assertEqual((new.pos,new.hp,new.acted),(u.pos,u.hp,u.acted))
        self.assertEqual(app.state.allowed_builds,before.allowed_builds)
        self.assertEqual(app.state.income_capture_limit,before.income_capture_limit)
        self.assertFalse(app.auto);self.assertTrue(app.human.get())
        self.assertIsNone(app.selected);self.assertIsNone(app.destination)
        self.assertFalse(app.planners);self.assertEqual(app.alternatives.size(),0)
        self.assertEqual(app.initial.key(),initial)
        after=app.rules.apply(app.state,END)
        self.assertEqual(after.player,1) # Human is next, with normal turn-start processing.

    def test_capture_progress_and_acted_flag_follow_units(self):
        app=self.app
        app.state=app.rules.apply(app.state,Action('capture','human_infantry_1',7))
        before=app.state.clone()
        with patch.object(app,'start'):app.flip_sides()
        self.assertEqual(app.state.captures,before.captures)
        self.assertTrue(app.state.units['human_infantry_1'].acted)
        self.assertEqual(app.state.units['human_infantry_1'].owner,0)

    def test_busy_terminal_and_bad_budget_do_not_mutate(self):
        app=self.app
        for kind in ('busy','terminal','bad_budget'):
            app.state=app.initial.clone();app.busy=False;app.seconds.set('1')
            if kind=='busy':app.busy=True
            if kind=='terminal':app.state.winner=0
            if kind=='bad_budget':app.seconds.set('invalid')
            before=app.state.key()
            with patch.object(app,'start') as start:
                app.flip_sides();start.assert_not_called()
            self.assertEqual(app.state.key(),before)

    def test_switch_to_active_red_and_automatic_blue_after_end(self):
        app=self.app
        app.state=app.rules.apply(app.state,END)
        before=app.state.key()
        with patch.object(app,'start') as start:
            app.flip_sides()
            start.assert_not_called()
            self.assertEqual(app.state.key(),before)
            self.assertEqual(app.human_side,1)
            app.end_turn()
            self.assertEqual(app.state.player,0)
            start.assert_called_once_with(True)

    def test_switch_back_to_blue(self):
        app=self.app
        with patch.object(app,'start'):
            app.flip_sides()
            app.flip_sides()
        self.assertEqual(app.human_side,0)
        self.assertIn('Human Blue',app.human_label.get())
