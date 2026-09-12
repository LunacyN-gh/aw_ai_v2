import tkinter as tk
import unittest
from aw_ai.gui import App
from aw_ai.review_ui import ReviewWindow
from aw_ai.notation import GameRecord,loads
from aw_ai.model import Board,State,Unit,END,Action
from aw_ai.scenarios import digest

class ReviewGuiTests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk();self.root.withdraw()
        self.state=State(Board.plain(4,2,{(0,0):'factory',(3,1):'factory'}),
                         {'a':Unit('a',0,'tank',0,hp=80)},owners={0:0,7:1})
    def tearDown(self):
        for job in self.root.tk.call('after','info'):self.root.after_cancel(job)
        self.root.destroy()

    def test_navigation_and_army_values(self):
        r=GameRecord(self.state)
        for a in [Action('wait','a',1),END,END]:r.append(a)
        ui=ReviewWindow(self.root,r);ui.window.withdraw()
        try:
            self.assertIn('$5,600',ui.economy.get())
            ui.navigate('next');self.assertEqual(ui.index,1)
            ui.navigate('next_turn');self.assertEqual(ui.index,2)
            ui.navigate('end');self.assertEqual(ui.index,3)
            ui.navigate('prev_turn');self.assertEqual(ui.index,2)
            ui.navigate('prev');self.assertEqual(ui.index,1)
            ui.navigate('start');self.assertEqual(ui.index,0)
            self.assertEqual(digest(ui.record.initial),digest(self.state))
            ui.results.put((ui.version-1,'STALE'));ui.poll()
            self.assertNotIn('STALE',ui.ranking.get('1.0','end'))
        finally:ui.close()

    def test_live_gui_records_and_prints_terminal_game_once(self):
        app=App(self.root);app.human.set(False);self.state.turn_limit=2
        app.reset(self.state)
        self.assertIn('$5,600',app.summary.get())
        app.apply_action(END);app.apply_action(END)
        text=app.log.get('1.0','end')
        self.assertEqual(text.count('Game notation:'),1)
        restored=loads(app.game_text)
        self.assertEqual(digest(restored.states[-1]),digest(app.state))
        app.apply_action(END)
        self.assertEqual(app.log.get('1.0','end').count('Game notation:'),1)
