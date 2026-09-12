import unittest
from aw_ai.gui_art import unit
from aw_ai.model import SPECS

class RecordingCanvas:
    def __init__(self): self.calls=[]
    def __getattr__(self,name):
        return lambda *xy,**style:self.calls.append((name,xy))

class FacingTests(unittest.TestCase):
    def test_every_sprite_is_reflected_for_player_two(self):
        for kind in SPECS:
            blue,red=RecordingCanvas(),RecordingCanvas()
            unit(blue,kind,10,20,64,0)
            unit(red,kind,10,20,64,1)
            self.assertEqual(len(blue.calls),len(red.calls))
            for (name,a),(other,b) in zip(blue.calls,red.calls):
                expected=list(a); expected[::2]=[84-x for x in a[::2]]
                if name in ('create_rectangle','create_oval'):
                    expected[0],expected[2]=min(expected[0],expected[2]),max(expected[0],expected[2])
                self.assertEqual(name,other)
                for x,y in zip(expected,b):self.assertAlmostEqual(x,y)
