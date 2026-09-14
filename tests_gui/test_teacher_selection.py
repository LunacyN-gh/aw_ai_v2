import tkinter as tk
import unittest
from unittest.mock import patch
from aw_ai.gui import App
from aw_ai.planner import Planner
from aw_ai.order_search import OrderPlanner
from aw_ai.scenarios import load


class TeacherSelectionTests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk();self.root.withdraw();self.app=App(self.root)
        self.app.reset(load('maps/6x6_ita/test_cities_6x6_ita.json'))
    def tearDown(self):
        for job in self.root.tk.call('after','info'):self.root.after_cancel(job)
        self.root.destroy()
    def selected_backend(self):
        with patch('aw_ai.gui.threading.Thread'):
            self.app.start(False)
        self.app.busy=False
        return next(iter(self.app.planners.values()))
    def test_default_and_explicit_old(self):
        self.assertIsInstance(self.selected_backend(),OrderPlanner)
        self.app.clear_neural('old')
        self.assertIs(type(self.selected_backend()),Planner)
        self.app.clear_neural()
        self.assertIsInstance(self.selected_backend(),OrderPlanner)
    def test_switch_from_mcts_clears_stale_analysis(self):
        app=self.app;app.search.set('mcts');app.neural_agent=object()
        app.analysis=object();app.alternatives.insert('end','stale')
        app.clear_neural()
        self.assertIsNone(app.neural_agent);self.assertIsNone(app.analysis)
        self.assertEqual(app.search.get(),'beam');self.assertEqual(app.alternatives.size(),0)
        self.assertIsInstance(self.selected_backend(),OrderPlanner)
    def test_busy_switch_leaves_active_agent_unchanged(self):
        app=self.app;app.busy=True;app.neural_agent=object();agent=app.neural_agent
        app.clear_neural('old')
        self.assertIs(app.neural_agent,agent);self.assertEqual(app.heuristic_backend,'ordered')
