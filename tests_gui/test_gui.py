"""Optional Tk integration checks. All windows are withdrawn immediately."""
import time
import tkinter as tk
import unittest

from aw_ai.gui import App
from aw_ai.model import Action
from aw_ai.scenarios import tactical


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()
        cls.app = App(cls.root)
        cls.root.update_idletasks()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.app.animation_speed.set("Instant")
        self.app.reset(tactical("blocker"))
        self.app.human.set(False)

    def test_guided_checkpoint_loads_deployment_settings(self):
        try:
            from aw_ai.neural import Network,NetworkConfig
            from aw_ai.guidance import GuidedAgent
        except ImportError:
            self.skipTest('optional torch unavailable')
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        network=Network(NetworkConfig(16,1,32,1,4))
        network.search_settings={'mode':'guided','neural_value_weight':0.}
        try:
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'guided.pt'; network.save(path)
                with patch('aw_ai.gui.filedialog.askopenfilename',return_value=str(path)):
                    self.app.load_neural()
                self.assertIsInstance(self.app.neural_agent,GuidedAgent)
                self.assertEqual(self.app.neural_agent.value_weight,0.)
                self.assertIn('Guided model',self.app.status.get())
        finally:
            self.app.clear_neural()

    def test_ita_map_menu_and_forbidden_purchase(self):
        from pathlib import Path
        from aw_ai.scenarios import load
        app=self.app
        app.reset(load(Path(__file__).resolve().parents[1]/'maps/test_cities_6x6_ita.json'))
        self.assertEqual(tuple(app.build_menu['values']),('infantry','tank','artillery'))
        app.state.remove(app.state.occupancy[0]); app.state.funds[0]=30000
        app.destination=0; app.build_kind.set('mech'); app.build()
        self.assertNotIn(0,app.state.occupancy)
        app.build_kind.set('tank'); app.build()
        self.assertEqual(app.state.units[app.state.occupancy[0]].kind,'tank')
        app.restart()
        self.assertEqual(app.state.allowed_builds,('infantry','tank','artillery'))
        app.reset(tactical('blocker'))
        self.assertIn('mech',app.build_menu['values'])

    def click_square(self, pos):
        from types import SimpleNamespace
        x,y=self.app.state.board.xy(pos)
        from unittest.mock import patch
        with patch.object(self.app.action_menu, "tk_popup"):
            self.app.click(SimpleNamespace(x=int(self.app.offset+(x+.5)*self.app.cell),
                                           y=int(self.app.offset+(y+.5)*self.app.cell)))

    def test_new_infantry_waits_then_moves_next_turn(self):
        from aw_ai.scenarios import load
        from aw_ai.model import END
        from pathlib import Path
        app=self.app
        app.reset(load(Path(__file__).resolve().parents[1]/"maps/v1/test_cities_6x6.json"))
        self.click_square(0)
        self.assertEqual(app.selected,"human_infantry_1")
        self.assertIsNone(app.destination)
        self.click_square(1)
        app.commit("wait")
        app.destination=0
        app.build_kind.set("infantry")
        app.build()
        uid=app.state.occupancy[0]
        self.click_square(0)
        self.assertIsNone(app.selected)
        self.assertIn("built this turn",app.status.get())
        app.apply_action(END)
        app.apply_action(END)
        self.click_square(0)
        self.assertEqual(app.selected,uid)
        self.click_square(6)
        self.assertEqual(app.destination,6)
        app.commit("wait")
        self.assertEqual(app.state.units[uid].pos,6)

    def test_base_infantry_selectable_after_two_neural_turns(self):
        try:
            from aw_ai.neural import Network,NeuralAgent,NetworkConfig
        except ImportError:
            self.skipTest("optional torch dependency unavailable")
        from aw_ai.scenarios import load
        from pathlib import Path
        app=self.app
        app.reset(load(Path(__file__).resolve().parents[1]/"maps/v1/test_cities_6x6.json"))
        checkpoint=Path(__file__).resolve().parents[1]/"models/cities_6x6.pt"
        app.neural_agent=NeuralAgent(Network.load(checkpoint) if checkpoint.exists() else Network(NetworkConfig(16,1,32,1,4)))
        app.human.set(True)
        app.seconds.set(".15")
        try:
            for _ in range(2):
                self.click_square(0)
                app.commit("wait")
                app.end_turn()
                deadline=time.monotonic()+10
                while app.busy and time.monotonic()<deadline:
                    self.root.update()
                    time.sleep(.01)
                self.assertFalse(app.busy)
                self.assertEqual(app.state.player,0)
            self.click_square(0)
            self.assertEqual(app.selected,"human_infantry_1")
            self.assertFalse(app.state.units[app.selected].acted)
            self.click_square(1)
            self.assertEqual(app.destination,1)
            app.commit("wait")
            self.assertEqual(app.state.units["human_infantry_1"].pos,1)
        finally:
            app.clear_neural()
            app.human.set(False)

    def test_saved_base_infantry_destination_menu_executes_move(self):
        from pathlib import Path
        from aw_ai.scenarios import load
        path=Path(__file__).resolve().parent/"fixtures/base_infantry_turn3.json"
        for destination in (1,6,2,12,3,8,13,18):
            with self.subTest(destination=destination):
                self.app.reset(load(path))
                self.click_square(0)
                self.assertEqual(self.app.selected,"p0_2")
                self.click_square(destination)
                self.assertEqual(self.app.state.units["p0_2"].pos,0)
                self.assertEqual(self.app.action_menu.entrycget(0,"label"),"Move / wait here")
                self.app.action_menu.invoke(0)
                self.assertEqual(self.app.state.units["p0_2"].pos,destination)
                self.assertTrue(self.app.state.units["p0_2"].acted)
                self.assertEqual(self.app.state.player,0)

    def test_destination_menu_preserves_move_and_attack(self):
        app=self.app
        app.apply_action(Action("wait","friend",7))
        self.click_square(0)
        self.click_square(2)
        index=next(i for i in range(app.action_menu.index("end")+1)
                   if app.action_menu.type(i)=="command" and app.action_menu.entrycget(i,"label").startswith("Attack "))
        app.action_menu.invoke(index)
        self.assertNotIn("enemy",app.state.units)

    def test_capture_limit_win_explains_rejected_purchase(self):
        from pathlib import Path
        from aw_ai.scenarios import load
        app=self.app
        app.reset(load(Path(__file__).resolve().parent/"fixtures/capture_limit_won.json"))
        self.assertIn("GAME OVER",app.summary.get())
        self.assertIn("7/7",app.status.get())
        key=app.state.key()
        self.click_square(0)
        self.assertIn("Blue wins",app.status.get())
        app.destination=0
        app.build_kind.set("tank")
        app.build()
        self.assertIn("game is over",app.status.get())
        self.assertNotIn("illegal action",app.status.get())
        self.assertEqual(app.state.key(),key)
        # The same site and funds support a tank when the win rule is disabled.
        changed=app.state.clone(); changed.winner=None; changed.income_capture_limit=None
        self.assertTrue(app.rules.is_legal(changed,Action("build",destination=0,build="tank")))

    def test_board_and_sidebar_construct(self):
        self.assertGreater(len(self.app.canvas.find_all()), 10)
        self.assertIn("Blue", self.app.summary.get())

    def test_human_move_then_attack(self):
        self.app.selected, self.app.destination = "friend", 7
        self.app.commit("wait")
        self.assertEqual(self.app.state.units["friend"].pos, 7)
        self.assertTrue(self.app.apply_action(Action("attack", "tank", 2, "enemy")))
        self.assertIn("wins", self.app.status.get())

    def test_generate_and_edit(self):
        self.app.size.set("20")
        self.app.count.set("12")
        self.app.generate()
        self.assertEqual(len(self.app.state.units), 24)
        self.app.paint_tile.set("city")
        self.app.paint_owner.set("blue")
        self.app.paint(200)
        self.assertEqual(self.app.state.board.tiles[200], "city")
        self.assertEqual(self.app.state.owners[200], 0)

    def test_async_analysis_does_not_mutate_board(self):
        key = self.app.state.key()
        self.app.seconds.set(".1")
        self.app.start(False)
        deadline = time.monotonic()+3
        while self.app.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.01)
        self.assertFalse(self.app.busy)
        self.assertIsNotNone(self.app.analysis)
        self.assertEqual(self.app.state.key(), key)
        self.assertGreater(self.app.alternatives.size(), 0)


if __name__ == "__main__":
    unittest.main()
