import unittest
from dataclasses import replace
from aw_ai.model import Board, State, Unit, Action, END
from aw_ai.rules import Rules
from aw_ai.production import utility_scores, facing_front
from aw_ai.policy import production_plans
from aw_ai.budget import Budget
from aw_ai.tactical_suite import cases, Oracle, hq_wall, breakthrough
from aw_ai.planner import Planner, Config


class ProductionTests(unittest.TestCase):
    def position(self,kind='b_copter'):
        b=Board.plain(20,8,{(1,4):'factory',(18,4):'factory'})
        return State(b,{'enemy':Unit('enemy',1,kind,b.pos(5,4))},funds=[20000,0],
                     owners={b.pos(1,4):0,b.pos(18,4):0})

    def test_counter_changes_with_enemy_composition(self):
        s=self.position(); r=Rules(); p=s.board.pos(1,4)
        scores=utility_scores(s,r,p)
        self.assertEqual(max(scores,key=scores.get),'anti_air')
        s.units['enemy']=replace(s.units['enemy'],kind='anti_air')
        scores=utility_scores(s,r,p)
        self.assertEqual(max(scores,key=scores.get),'tank')

    def test_final_planner_build_changes_with_composition(self):
        b=Board.plain(12,5,{(1,2):'factory'})
        s=State(b,{'e':Unit('e',1,'b_copter',b.pos(5,2))},funds=[8000,0],owners={b.pos(1,2):0})
        for enemy,counter in (('b_copter','anti_air'),('anti_air','tank')):
            s.units['e']=replace(s.units['e'],kind=enemy)
            a=Planner(config=Config(seconds=5)).analyze(s)
            self.assertEqual([a.build for a in a.actions if a.kind=='build'],[counter])

    def test_opposite_factories_face_different_armies(self):
        s=self.position(); b=s.board; r=Rules()
        s.units['other']=Unit('other',1,'anti_air',b.pos(15,4))
        left=facing_front(s,r,b.pos(1,4)); right=facing_front(s,r,b.pos(18,4))
        self.assertGreater(left.enemy['b_copter'],right.enemy['b_copter'])
        self.assertGreater(right.enemy['anti_air'],left.enemy['anti_air'])
        self.assertEqual(max(utility_scores(s,r,b.pos(1,4)),key=utility_scores(s,r,b.pos(1,4)).get),'anti_air')
        self.assertEqual(max(utility_scores(s,r,b.pos(18,4)),key=utility_scores(s,r,b.pos(18,4)).get),'tank')

    def test_reinforcements_reduce_duplicate_counter_demand(self):
        s=self.position(); r=Rules(); p=s.board.pos(1,4)
        before=utility_scores(s,r,p)['anti_air']
        s.units['counter']=Unit('counter',0,'anti_air',s.board.pos(2,4))
        self.assertLess(utility_scores(s,r,p)['anti_air'],before)

    def test_shared_budget_and_save(self):
        s=self.position(); s.funds[0]=8000; r=Rules()
        plans=production_plans(s,r,Budget(5,1000),8)
        self.assertTrue(any(not actions for _,actions,_ in plans))
        for branch,actions,_ in plans:
            checked=s
            for a in actions: checked=r.apply(checked,a)
            self.assertEqual(checked.key(),branch.key())
            self.assertGreaterEqual(branch.funds[0],0)
        self.assertTrue(any(a.build=='anti_air' for _,aa,_ in plans for a in aa))


class TacticalTests(unittest.TestCase):
    def test_unique_solutions_all_orientations(self):
        for p in cases():
            with self.subTest(p.name):
                self.assertEqual(len(Oracle().solve(p)),1)

    def test_hq_requires_both_gates(self):
        p=hq_wall(); r=Rules(); solution=Oracle().solve(p)[0]
        s=p.state
        for a in solution: s=r.apply(s,a)
        self.assertTrue(Oracle().success(s,p))
        for uid in ('north','west'):
            broken=s.clone(); broken.remove(uid)
            self.assertFalse(Oracle().success(broken,p))

    def test_small_oracle_budget_does_not_claim_proof(self):
        with self.assertRaises(RuntimeError): Oracle(max_states=1).solve(hq_wall())

    def test_search_finds_breakthrough_even_when_value_prefers_other_plan(self):
        for kind in ('artillery','anti_air','b_copter'):
            p=breakthrough(kind); planner=Planner(config=Config(seconds=5))
            a=planner.analyze(p.state)
            self.assertTrue(any(Oracle().success(c.state,p) for c in a.alternatives))

    def test_selects_complete_hq_screen(self):
        p=hq_wall(); planner=Planner(config=Config(seconds=5))
        self.assertTrue(Oracle().success(planner.execute(p.state,planner.analyze(p.state)),p))
