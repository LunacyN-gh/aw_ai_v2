import copy
import random
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aw_ai.model import Action, END, SPECS
from aw_ai.rules import Rules
from aw_ai.planner import Planner, Config
from aw_ai.policy import Proposal, HeuristicPolicy
from aw_ai.guidance import GuidedAgent
from aw_ai.opening_suite import fixtures
from aw_ai.promotion import promotion_decision


class HostilePrior:
    network = SimpleNamespace(allowed_builds=('infantry','tank','artillery'))
    def propose(self,state,context,rules,actors=None,limit=12,quiet=False):
        actions = [a for a in rules.legal_actions(state) if a.kind=='wait'
                   and (actors is None or a.actor in actors)]
        return [Proposal(a,1e20,'bad prior') for a in actions[:limit]]
    def build_scores(self,state,rules,site):
        return {k:1e20 if k=='save' else -1e20 for k in (*SPECS,'save')}
    def evaluate(self,*args):
        raise AssertionError('zero-weight neural value must not run')


class GuidedSearchTests(unittest.TestCase):
    def test_openings_survive_overconfident_bad_policy_and_value(self):
        for name,state,base,city in fixtures():
            rules=Rules(); player=state.player
            agent=GuidedAgent(HostilePrior())
            planner=Planner(rules=rules,policy=agent,value=agent,config=Config(seconds=10,nodes=1000))
            result=planner.analyze(state)
            after=planner.execute(state,result)
            self.assertGreater(result.metrics.counts['heuristic_fallback_turns'],0)
            if city is not None:
                self.assertEqual(after.owners.get(city),player,name)
            else:
                self.assertIn(base,after.occupancy,name)
                self.assertEqual(after.units[after.occupancy[base]].kind,'infantry')
                self.assertTrue(after.captures,name)

    def test_candidates_include_both_sources_and_respect_roster(self):
        _,state,base,_=next(fixtures()); rules=Rules()
        agent=GuidedAgent(HostilePrior())
        from aw_ai.strategy import Strategist
        from aw_ai.budget import Budget
        context=Strategist().plan(state,rules,Budget(10,1000))
        proposals=agent.propose(state,context,rules,limit=8)
        self.assertTrue(any(p.action.kind=='capture' for p in proposals))
        self.assertTrue(any(p.action.kind=='wait' for p in proposals))
        self.assertEqual(len({p.action for p in proposals}),len(proposals))
        state.remove(state.occupancy[base]); state.funds[0]=30000
        scores=agent.build_scores(state,rules,base)
        self.assertEqual(scores['mech'],float('-inf'))
        self.assertGreater(scores['infantry'],scores['save'])


class PromotionTests(unittest.TestCase):
    def reports(self):
        reference={'summary':{m:{'win':2,'loss':2,'censored':0} for m in ('policy','beam')},
                   'openings':{m:{'a':True,'b':True} for m in ('policy','beam')}}
        candidate=copy.deepcopy(reference)
        candidate['summary']['gate']={'win':3,'loss':1,'censored':0}
        return reference,candidate

    def test_gate_accepts_only_without_regressions(self):
        reference,candidate=self.reports()
        self.assertTrue(promotion_decision(candidate,reference)[0])
        for field in ('head_to_head','policy','beam','opening','raw_opening'):
            _,bad=self.reports()
            if field=='head_to_head': bad['summary']['gate']={'win':1,'loss':0,'censored':3}
            elif field in ('policy','beam'): bad['summary'][field]['loss']=3
            elif field=='opening': bad['openings']['beam']['a']=False
            else: bad['openings']['policy']['a']=False
            self.assertFalse(promotion_decision(bad,reference)[0],field)


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'optional torch dependency unavailable')
class ExpertReplayTests(unittest.TestCase):
    def test_timeout_does_not_teach_unsearched_save(self):
        from aw_ai.search_training import search_turn,teacher_planner
        _,state,base,_=next(fixtures())
        state.remove(state.occupancy[base])
        planner=teacher_planner(Rules(),('infantry','tank','artillery'),Config(seconds=0,nodes=0))
        _,examples,metrics=search_turn(planner,state,random.Random(0))
        self.assertEqual(examples,[])
        self.assertEqual(metrics['partial'],1)

    def test_partial_production_keeps_purchases_without_inventing_saves(self):
        from aw_ai.model import Board,State,Unit
        from aw_ai.planner import Candidate
        from aw_ai.search_training import search_turn
        rules=Rules()
        state=State(Board.plain(2,2,{(0,0):'factory',(1,0):'factory'}),
                    {'e':Unit('e',1,'infantry',3)},funds=[10000,0],owners={0:0,1:0})
        sequence=(Action('build',destination=0,build='infantry'),END)
        after=state
        for a in sequence: after=rules.apply(after,a)
        candidate=Candidate(sequence,after,0.,'interrupted',verified=True,production_complete=False)
        planner=SimpleNamespace(rules=rules,analyze=lambda _:SimpleNamespace(actions=sequence,alternatives=[candidate]))
        _,examples,metrics=search_turn(planner,state,random.Random(0))
        self.assertEqual([e.site for e in examples],[0])
        self.assertEqual(examples[0].target,{0:1.})
        self.assertEqual(metrics['partial'],1)

    def test_expert_targets_teach_the_recommended_action(self):
        from aw_ai.search_training import search_turn,teacher_planner
        _,state,_,_=next(fixtures())
        planner=teacher_planner(Rules(),('infantry','tank','artillery'),Config(seconds=10,nodes=1000))
        result,examples,_=search_turn(planner,state,random.Random(0))
        self.assertTrue(examples)
        for e in examples: self.assertEqual(list(e.target.values()),[1.])
        self.assertIn(0,result.occupancy)  # Teacher actually bought opening infantry.

    def test_anchors_survive_and_batch_reserves_teacher_and_production(self):
        from aw_ai.search_training import Example
        from aw_ai.teacher_replay import TeacherReplay, training_batch
        _,state,base,_=next(fixtures())
        bank=TeacherReplay(16,0)
        bank.extend([Example(state,None,{},source='bootstrap'),Example(state,base,{},source='bootstrap')]*8)
        original=list(bank.anchors)
        bank.extend([Example(state,None,{},source='correction'),Example(state,base,{},source='teacher')]*100)
        self.assertEqual(original,bank.anchors)
        replay=[Example(state,None,{}),Example(state,base,{})]*100
        batch=training_batch(replay,bank,.5,64,random.Random(0))
        self.assertEqual(len(batch),64)
        self.assertEqual(sum(e.source!='search' for e in batch),32)
        self.assertEqual(sum(e.site is not None for e in batch),32)
        self.assertEqual(sum(e.source=='bootstrap' for e in batch),16)

    def test_teacher_counterfactuals_never_get_actual_outcome(self):
        from aw_ai.search_training import play_job, Example, model_payload
        from aw_ai.neural import Network,NetworkConfig
        torch.set_num_threads(1)
        network=Network(NetworkConfig(16,1,32,1,4))
        network.search_settings={'mode':'guided','neural_value_weight':0.}
        _,state,_,_=next(fixtures()); action=next(a for a in Rules().legal_actions(state) if a.kind=='wait')
        def turn(planner,state,rng,explore=False):
            after=state.clone()
            after.winner=1 if isinstance(planner.policy,HeuristicPolicy) else 0
            return after,[Example(state,None,{action:1.})],{'verified':1,'partial':0}
        job=dict(state=state,seed=0,boot=False,opponent='teacher',side=0,allowed=network.allowed_builds,
                 search=Config(seconds=5,nodes=1000),turns=2,current=model_payload(network),other=None,teacher_every=1)
        with patch('aw_ai.search_training.search_turn',side_effect=turn):
            actual,winner,_,_,_,stats=play_job(job)
        self.assertEqual(winner,0)
        self.assertEqual(actual[0].result,1.)
        self.assertTrue(stats['corrections'])
        self.assertTrue(all(e.result is None and e.source=='correction' for e in stats['corrections']))
        self.assertEqual(stats['played_turns'],1)

    def test_teacher_opponent_moves_are_retained_and_side_pairs_balance(self):
        from aw_ai.search_training import play_job,Example,opponent_kind,model_payload
        from aw_ai.neural import Network,NetworkConfig
        for baseline,n in ((False,4),(True,5)):
            for i in range(n):
                self.assertEqual(opponent_kind(i*2,baseline),opponent_kind(i*2+1,baseline))
        _,state,_,_=next(fixtures()); action=next(a for a in Rules().legal_actions(state) if a.kind=='wait')
        def turn(planner,state,rng,explore=False):
            after=state.clone(); after.winner=0
            return after,[Example(state,None,{action:1.})],{'verified':1,'partial':0}
        network=Network(NetworkConfig(16,1,32,1,4))
        job=dict(state=state,seed=0,boot=False,opponent='teacher',side=1,allowed=network.allowed_builds,
                 search=Config(),turns=1,current=model_payload(network),other=None)
        with patch('aw_ai.search_training.search_turn',side_effect=turn):
            actual,*_=play_job(job)
        self.assertEqual(actual[0].source,'teacher')
        self.assertEqual(actual[0].result,1.)

    def test_self_play_queries_teacher_for_both_sides(self):
        from aw_ai.search_training import play_job,model_payload
        from aw_ai.neural import Network,NetworkConfig
        network=Network(NetworkConfig(16,1,32,1,4))
        network.search_settings={'mode':'guided','neural_value_weight':0.}
        _,state,_,_=next(fixtures()); queried=[]
        def turn(planner,state,rng,explore=False):
            if isinstance(planner.policy,HeuristicPolicy): queried.append(state.player)
            return planner.rules.apply(state,END),[],{'verified':1,'partial':0}
        job=dict(state=state,seed=0,boot=False,opponent='self',side=0,allowed=network.allowed_builds,
                 search=Config(),turns=4,current=model_payload(network),other=None,teacher_every=2)
        with patch('aw_ai.search_training.search_turn',side_effect=turn): play_job(job)
        self.assertEqual(queried,[0,1])

    def test_checkpoint_roundtrip_preserves_guided_deployment(self):
        from aw_ai.neural import Network,NetworkConfig
        from aw_ai.guidance import deployment_agent
        from aw_ai.search_training import model_payload,restore_model
        network=Network(NetworkConfig(16,1,32,1,4))
        network.search_settings={'mode':'guided','neural_value_weight':.2}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'guided.pt'; network.save(path)
            loaded=Network.load(path)
        for restored in (loaded,restore_model(model_payload(network))):
            self.assertEqual(restored.search_settings,network.search_settings)
            self.assertIsInstance(deployment_agent(restored),GuidedAgent)
            self.assertEqual(deployment_agent(restored).value_weight,.2)

    def test_runner_keeps_reference_and_rejects_unpromoted_weights(self):
        from aw_ai.search_training import run,Example
        from aw_ai.neural import Network
        torch.set_num_threads(1)
        measured=[]
        def turn(planner,state,rng,explore=False):
            action=next(a for a in planner.rules.legal_actions(state) if a.kind=='wait')
            after=state.clone(); after.winner=state.player
            return after,[Example(state,None,{action:1.})],{'verified':1,'partial':0}
        def evaluate(network,initial,config,games):
            measured.append({k:v.detach().cpu().clone() for k,v in network.state_dict().items()})
            result={'summary':{m:{'win':2,'loss':2,'censored':0} for m in ('policy','beam')},
                    'games':[],'openings':{m:{'opening':True} for m in ('policy','beam')}}
            if len(measured)>1:
                wins=3 if len(measured)==2 else 0
                result['summary']['gate']={'win':wins,'loss':4-wins,'censored':0}
            return result
        with tempfile.TemporaryDirectory() as d, patch('aw_ai.search_training.search_turn',side_effect=turn), patch('aw_ai.search_training.evaluate',side_effect=evaluate):
            output=Path(d)/'test.pt'
            report=run(output,games=4,bootstrap=1,turns=1,tiny=True,device='cpu',
                       map_path=Path(__file__).resolve().parents[1]/'maps/v1/test_cities_6x6.json',
                       eval_every=2,eval_games=4,updates_per_game=1,batch_size=8)
            self.assertEqual([p['accepted'] for p in report['promotions']],[True,True,False])
            self.assertEqual(report['best_measured_game'],3)
            self.assertEqual(report['reference_game'],1)
            for file,index in (('test.reference.pt',0),('test.best.pt',1),('test.pt',2)):
                actual=Network.load(Path(d)/file).state_dict()
                for key,expected in measured[index].items():
                    torch.testing.assert_close(actual[key],expected,rtol=0,atol=0)


if __name__ == '__main__': unittest.main()
