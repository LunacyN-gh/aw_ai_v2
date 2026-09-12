import unittest
from aw_ai.model import Board,State,Unit,Action,END,DRAW,displayed_army_value
from aw_ai.rules import Rules
from aw_ai.notation import GameRecord,loads,action_text,parse_action,unit_name,square,position
from aw_ai.scenarios import digest,to_data

def initial():
    b=Board.plain(6,6,{(0,0):'factory',(5,5):'factory',(1,1):'city'})
    return State(b,{'inf':Unit('inf',0,'infantry',0),'tank':Unit('tank',0,'tank',30),
                    'enemy':Unit('enemy',1,'infantry',28)},funds=[30000,1000],owners={0:0,35:1})

class NotationTests(unittest.TestCase):
    def test_examples_and_every_prefix_roundtrip(self):
        s=initial();r=GameRecord(s)
        actions=[Action('capture','inf',7),Action('attack','tank',27,'enemy'),
                 Action('build',destination=0,build='tank'),END,END,Action('capture','inf',7)]
        self.assertEqual(action_text(s,actions[0]),'i: a6-b5x')
        self.assertEqual(unit_name(s,'enemy'),'i:e2')
        for a in actions:
            r.append(a)
            restored=loads(r.dumps())
            self.assertEqual(digest(restored.states[-1]),digest(r.states[-1]))
            self.assertEqual(restored.actions,r.actions)
        self.assertIn('i: b5x',r.dumps())
        self.assertIn('a6+t',r.dumps())
        no_prefix=r.dumps().replace('i: ','').replace('t: ','')
        self.assertEqual(digest(loads(no_prefix).states[-1]),digest(r.states[-1]))

    def test_deadline_results_and_empty_turns(self):
        for owner,outcome,ending in [(None,DRAW,'1/2-1/2'),(0,0,'1-0'),(1,1,'0-1')]:
            s=initial();s.turn_limit=2
            if owner is not None:s.owners[7]=owner
            r=GameRecord(s);r.append(END);r.append(END)
            self.assertEqual(Rules().outcome(r.states[-1]),outcome)
            self.assertTrue(r.dumps().endswith(ending))
            self.assertEqual(digest(loads(r.dumps()).states[-1]),digest(r.states[-1]))
            self.assertEqual(r.turn_boundaries(),[0,1,2])

    def test_invalid_text_does_not_change_initial(self):
        s=initial();before=digest(s)
        for text in ['1. a6-a1\n*','1. t:a6-b5\n*','2. --\n*','1. --\n1-0','1. --, --\n*']:
            with self.assertRaises(ValueError):loads(text,s)
            self.assertEqual(digest(s),before)

    def test_wide_board_coordinates_and_value_uses_rounded_hp(self):
        b=Board.plain(28,2)
        self.assertEqual(square(b,27),'ab2');self.assertEqual(position(b,'ab2'),27)
        s=State(b,{'a':Unit('a',0,'tank',0,hp=71),'b':Unit('b',0,'infantry',1,hp=81)})
        self.assertEqual(displayed_army_value(s,0),6500)
        self.assertEqual(displayed_army_value(s,1),0)

    def test_midgame_initial_snapshot(self):
        s=initial();s.turn=7;s.player=1;s.units={};s.__post_init__()
        r=GameRecord(s);r.append(END)
        self.assertIn('8. --',r.dumps())
        self.assertEqual(digest(loads(r.dumps()).states[-1]),digest(r.states[-1]))

class ReviewTests(unittest.TestCase):
    def test_successor_value_changes_perspective_at_end(self):
        import torch
        from aw_ai.review import rank_actions
        class Fake:
            allowed_builds=('infantry','tank','artillery')
            def encode_batch(self,states,rules):return [(torch.ones(2),None,None) for s in states]
            def value(self,globals):return torch.full((len(globals),1),.25)
        s=State(Board.plain(3,1),{'a':Unit('a',0,'infantry',0),'b':Unit('b',1,'infantry',2)})
        rows=rank_actions(Fake(),s,5)
        end=next(r for r in rows if r['action']==END)
        self.assertEqual(end['value'],-.25)
        self.assertEqual(rows[0]['value'],.25)
        self.assertLessEqual(len(rows),5)

    def test_terminal_value_is_exact(self):
        from aw_ai.review import rank_actions
        class Fake:
            allowed_builds=('infantry',)
            def encode_batch(self,states,rules):raise AssertionError('terminal states must not use NN')
        s=initial();s.units={};s.__post_init__();s.funds=[0,0];s.turn_limit=1;s.owners[7]=0
        rows=rank_actions(Fake(),s,1)
        self.assertEqual(rows[0]['value'],1.)
        self.assertTrue(rows[0]['terminal'])
