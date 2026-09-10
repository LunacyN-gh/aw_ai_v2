import unittest
import tempfile
import random
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
try:
    import torch
except ImportError:
    torch=None

from aw_ai.model import Board,State,Unit,Action,END
from aw_ai.rules import Rules
from aw_ai.planner import Planner,Config
from aw_ai.policy import production_plans,HeuristicPolicy
from aw_ai.budget import Budget


class TeacherMechTests(unittest.TestCase):
    def test_never_proposes_mech_without_ground_vehicles(self):
        b=Board.plain(8,4,{(0,0):'factory',(3,0):'city'})
        for enemy in ('infantry','mech','b_copter'):
            s=State(b,{'e':Unit('e',1,enemy,31)},funds=[30000,0],owners={0:0})
            plans=production_plans(s,Rules(),Budget(5,10000),width=100)
            self.assertFalse(any(a.build=='mech' for _,actions,_ in plans for a in actions))
            a=Planner(config=Config(seconds=5)).analyze(s)
            self.assertFalse(any(x.build=='mech' for c in a.alternatives for x in c.actions))
            self.assertTrue(Rules().is_legal(s,Action('build',destination=0,build='mech')))

    def test_vehicle_allows_mech_again(self):
        b=Board.plain(8,4,{(0,0):'factory'})
        s=State(b,{'e':Unit('e',1,'tank',31)},funds=[30000,0],owners={0:0})
        scores=HeuristicPolicy().build_scores(s,Rules(),0)
        self.assertNotEqual(scores['mech'],float('-inf'))
        plans=production_plans(s,Rules(),Budget(5,10000),width=100)
        self.assertTrue(any(a.build=='mech' for _,actions,_ in plans for a in actions))


@unittest.skipIf(torch is None,'optional torch dependency unavailable')
class SearchTrainingTests(unittest.TestCase):
    def setUp(self):
        from aw_ai.neural import Network,NetworkConfig
        torch.set_num_threads(1); torch.manual_seed(1)
        self.network=Network(NetworkConfig(16,1,32,1,4))
        self.rules=Rules()

    def test_search_targets_normalized_and_legally_executable(self):
        from aw_ai.search_training import search_turn
        from aw_ai.neural_training import training_map,decision_logits
        s=training_map(0)
        planner=Planner(config=Config(seconds=5,nodes=1000))
        result,examples,_=search_turn(planner,s,random.Random(1),True)
        result.validate()
        self.assertTrue(examples)
        for e in examples:
            choices,_,_=decision_logits(self.network,e.state,self.rules,e.site)
            self.assertAlmostEqual(sum(e.target.values()),1.)
            self.assertTrue(set(e.target).issubset(choices))

    def test_soft_candidate_targets_not_only_selected_action(self):
        from aw_ai.search_training import search_turn
        from aw_ai.planner import Candidate
        from types import SimpleNamespace
        s=State(Board.plain(5,2),{'a':Unit('a',0,'infantry',0),'e':Unit('e',1,'infantry',9)})
        aa=[(Action('wait','a',p),END) for p in (0,1)]
        cc=[]
        for sequence in aa:
            child=s
            for action in sequence: child=self.rules.apply(child,action)
            cc.append(Candidate(sequence,child,0.,'test',verified=True))
        planner=SimpleNamespace(rules=self.rules,analyze=lambda _:SimpleNamespace(actions=aa[0],alternatives=cc))
        _,examples,_=search_turn(planner,s,random.Random(0))
        self.assertEqual(examples[0].target,{aa[0][0]:.5,aa[1][0]:.5})

    def test_bootstrap_updates_value_and_censored_examples_do_not(self):
        from aw_ai.search_training import Example,label_result,fit
        s=State(Board.plain(5,2),{'a':Unit('a',0,'infantry',0),'e':Unit('e',1,'infantry',9)})
        examples=[Example(s,None,{Action('wait','a',1):1.})]
        optimizer=torch.optim.AdamW(self.network.parameters(),lr=.001)
        before=self.network.value[-2].weight.detach().clone()
        metrics=fit(self.network,optimizer,label_result(examples,0),self.rules,random.Random(0),1,8)
        self.assertEqual(metrics['value_samples'],1)
        self.assertFalse(torch.equal(before,self.network.value[-2].weight))
        before=self.network.value[-2].weight.detach().clone()
        metrics=fit(self.network,optimizer,label_result(examples,None),self.rules,random.Random(0),1,8)
        self.assertEqual(metrics['value_samples'],0)
        self.assertIsNone(metrics['value_mse'])
        self.assertTrue(torch.equal(before,self.network.value[-2].weight))

    def test_roster_masks_survive_checkpoint_and_limit_teacher(self):
        from aw_ai.neural import Network,NeuralAgent,KINDS
        from aw_ai.search_training import ITA
        from aw_ai.neural_training import build_choices,decision_logits
        self.network.allowed_builds=ITA
        b=Board.plain(8,4,{(0,0):'factory'})
        s=State(b,{'e':Unit('e',1,'tank',31)},funds=[30000,0],owners={0:0})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test.pt'; self.network.save(path)
            restored=Network.load(path)
            self.assertEqual(restored.allowed_builds,ITA)
            choices,_,_=decision_logits(restored,s,self.rules,0)
            self.assertEqual(set(choices),{KINDS.index(k) for k in ITA}|{len(KINDS)})
            for policy in (HeuristicPolicy(ITA),NeuralAgent(restored)):
                plans=production_plans(s,self.rules,Budget(5,10000),100,scorer=policy.build_scores)
                self.assertTrue(all(a.build in ITA for _,actions,_ in plans for a in actions))

    def test_fixed_map_run_periodically_evaluates_and_saves(self):
        from aw_ai.search_training import run,Example
        from aw_ai.scenarios import load,digest
        path=Path(__file__).resolve().parents[1]/'maps/v1/test_cities_6x6.json'
        expected=load(path)
        observed=[]
        def turn(planner,state,rng,explore=False):
            if not observed: self.assertEqual(digest(state),digest(expected))
            observed.append(state)
            action=next(a for a in self.rules.legal_actions(state) if a.kind=='wait')
            example=Example(state.clone(),None,{action:1.})
            result=state.clone(); result.winner=state.player
            return result,[example],{'verified':1,'partial':0}
        evaluations=[]
        def evaluate(network,initial,config,games):
            evaluations.append(games)
            self.assertEqual(digest(initial(1000000)),digest(expected))
            return {'summary':{m:{'win':1,'loss':1,'censored':0} for m in ('policy','beam')},'games':[]}
        with tempfile.TemporaryDirectory() as d, patch('aw_ai.search_training.search_turn',side_effect=turn), patch('aw_ai.search_training.evaluate',side_effect=evaluate):
            target=Path(d)/'test.pt'
            report=run(target,games=2,bootstrap=1,turns=1,map_path=path,tiny=True,roster='ita',
                       updates_per_game=1,batch_size=8,eval_every=1,eval_games=2,checkpoint_every=1)
            self.assertTrue(target.exists()); self.assertTrue(target.with_name('test.best.pt').exists())
            self.assertEqual(len(evaluations),3)
            self.assertTrue(report['complete'])
            for row in report['runs']:
                for key in ('elapsed_seconds','play_seconds','training_seconds','run_elapsed_seconds'):
                    self.assertGreaterEqual(row[key],0.)
                self.assertGreaterEqual(row['elapsed_seconds']+.003,row['play_seconds']+row['training_seconds'])
            for evaluation in report['evaluations']:
                self.assertGreaterEqual(evaluation['elapsed_seconds'],0.)
                self.assertGreaterEqual(evaluation['run_elapsed_seconds'],evaluation['elapsed_seconds'])
            self.assertGreaterEqual(report['total_elapsed_seconds'],report['runs'][-1]['run_elapsed_seconds'])
            import json
            persisted=json.loads(target.with_suffix('.json').read_text())
            self.assertEqual(persisted['total_elapsed_seconds'],report['total_elapsed_seconds'])
            self.assertEqual(persisted['runs'][0]['elapsed_seconds'],report['runs'][0]['elapsed_seconds'])
            self.assertTrue(report['runs'][0]['new_value_labels'])
            self.assertTrue(report['runs'][1]['new_value_labels'])
            self.assertTrue(report['runs'][2]['new_value_labels'])  # Paired self-play records either side's moves.
            self.assertGreater(report['runs'][0]['losses']['value_samples'],0)
