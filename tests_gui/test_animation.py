import time
import tkinter as tk
import unittest
from types import SimpleNamespace
from aw_ai.gui import App
from aw_ai.animation import movement_path
from aw_ai.model import State, Board, Unit, Action, END, SPECS, DEFENSE
from aw_ai.rules import Rules, movement_cost


class PathTests(unittest.TestCase):
    def test_detours_around_enemy_and_impassable_terrain(self):
        b=Board.plain(5,3,{(2,1):'water'})
        s=State(b,{'a':Unit('a',0,'tank',5),'e':Unit('e',1,'infantry',6)})
        path=movement_path(s,'a',9,Rules())
        self.assertEqual((path[0],path[-1]),(5,9))
        self.assertNotIn(6,path); self.assertNotIn(7,path)
        self.assertTrue(all(q in b.neighbors[p] for p,q in zip(path,path[1:])))
        self.assertLessEqual(sum(movement_cost('tread',b.tiles[p]) for p in path[1:]),6)

    def test_friendly_transit_but_no_occupied_destination(self):
        s=State(Board.plain(4,1),{'a':Unit('a',0,'infantry',0),'f':Unit('f',0,'infantry',1)})
        self.assertEqual(movement_path(s,'a',2,Rules()),[0,1,2])
        with self.assertRaises(ValueError): movement_path(s,'a',1,Rules())
        self.assertEqual(movement_path(s,'a',0,Rules()),[0])


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk(); self.root.withdraw(); self.app=App(self.root)
        self.app.human.set(False)
        self.state=State(Board.plain(5,2),{'a':Unit('a',0,'tank',0),'e':Unit('e',1,'infantry',3,hp=10)})
        self.app.reset(self.state)
    def tearDown(self):
        for job in self.root.tk.call('after','info'): self.root.after_cancel(job)
        self.root.destroy()
    def pump(self,condition,seconds=3):
        end=time.monotonic()+seconds
        while not condition() and time.monotonic()<end:
            self.root.update(); time.sleep(.005)
        self.assertTrue(condition())

    def test_move_then_both_flash_before_destroyed_unit_disappears(self):
        self.app.animation_speed.set('1×')
        self.app.play_action(Action('attack','a',2,'e'))
        self.assertTrue(self.app.busy)
        self.assertEqual(self.app.state.units['a'].pos,0)
        self.pump(lambda: bool(self.app.motion) and 0<self.app.motion['a'][0]<1)
        self.pump(lambda: bool(self.app.flashing))
        flash_started=time.monotonic()
        self.assertEqual(self.app.flashing,{'a','e'})
        self.assertIn('e',self.app.state.units)
        self.assertEqual(self.app.motion['a'],(2,0))
        self.pump(lambda: not self.app.busy)
        self.assertGreater(time.monotonic()-flash_started,.43)
        self.assertEqual(self.app.state.units['a'].pos,2)
        self.assertNotIn('e',self.app.state.units)
        self.assertFalse(self.app.motion or self.app.flashing)

    def test_reset_cancels_old_animation(self):
        self.app.play_action(Action('wait','a',2))
        replacement=self.state.clone(); self.app.reset(replacement)
        self.pump(lambda: True)
        end=time.monotonic()+.5
        while time.monotonic()<end: self.root.update(); time.sleep(.01)
        self.assertEqual(self.app.state.units['a'].pos,0)
        self.assertFalse(self.app.busy or self.app.motion)

    def test_instant_and_ai_sequence_match_rules_execution(self):
        self.app.animation_speed.set('Instant')
        actions=(Action('wait','a',1),END)
        expected=self.state
        for a in actions: expected=self.app.rules.apply(expected,a)
        self.app.animate(SimpleNamespace(root_key=self.state.key(),actions=actions))
        self.pump(lambda: not self.app.busy)
        self.assertEqual(self.app.state.key(),expected.key())

    def test_all_sprite_types_draw_at_small_and_large_zoom(self):
        from aw_ai import gui_art
        for size in (12,65):
            for tile in DEFENSE:
                gui_art.terrain(self.app.canvas,tile,0,0,size,1)
            for kind in SPECS:
                for side in (0,1):
                    gui_art.unit(self.app.canvas,kind,0,0,size,side)
                    gui_art.unit(self.app.canvas,kind,0,0,size,side,True,True)
        self.assertGreater(len(self.app.canvas.find_all()),100)
