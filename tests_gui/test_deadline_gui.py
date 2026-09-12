import tkinter as tk
import unittest
from pathlib import Path
from aw_ai.gui import App
from aw_ai.model import DRAW, END
from aw_ai.scenarios import load, to_data, from_data


class DeadlineGuiTests(unittest.TestCase):
    def test_deadline_display_draw_and_restart(self):
        root=tk.Tk(); root.withdraw()
        try:
            app=App(root); app.human.set(False)
            path=Path(__file__).resolve().parents[1]/'maps/6x6_ita/test_cities_6x6_ita.json'
            state=load(path); state.turn_limit=2
            app.reset(state)
            self.assertIn('2 remaining',app.summary.get())
            app.end_turn(); app.end_turn()
            self.assertEqual(app.rules.outcome(app.state),DRAW)
            self.assertIn('Draw',app.game_over_message())
            self.assertIn('0 remaining',app.summary.get())
            self.assertEqual(from_data(to_data(app.state)).winner,DRAW)
            app.restart()
            self.assertEqual((app.state.turn,app.state.turn_limit),(0,2))
            self.assertIsNone(app.rules.outcome(app.state))
        finally: root.destroy()

    def test_deadline_win_message(self):
        root=tk.Tk(); root.withdraw()
        try:
            app=App(root); app.human.set(False)
            state=load(Path(__file__).resolve().parents[1]/'maps/6x6_ita/test_cities_6x6_ita.json')
            city=next(p for p in state.board.properties if state.board.tiles[p]=='city')
            state.owners[city]=1; state.turn_limit=1
            app.reset(state); app.end_turn()
            self.assertIn('Red wins',app.game_over_message())
            self.assertIn('higher income',app.game_over_message())
        finally: root.destroy()
