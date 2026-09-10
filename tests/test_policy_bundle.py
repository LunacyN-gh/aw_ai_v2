import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from aw_ai.neural import Network,NetworkConfig,NeuralAgent
from aw_ai.neural_training import learned_turn
from aw_ai.policy_bundle import policy_turn
from aw_ai.guidance import deployment_agent
from aw_ai.planner import Planner,Config
from aw_ai.rules import Rules
from aw_ai.opening_suite import fixtures


class PolicyBundleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(4)
        self.n=Network(NetworkConfig(16,1,32,1,4)).eval()
        self.n.allowed_builds=('infantry','tank','artillery')
        self.n.search_settings=dict(mode='guided',planner='bundle',bundle_version=2,neural_value_weight=.5,bundle_candidates=4)

    def test_greedy_matches_raw_including_builds_and_capture(self):
        for _,state,_,_ in fixtures():
            r=Rules();expected,_=learned_turn(self.n,state.clone(),r,random.Random(0),False,True)
            c=policy_turn(NeuralAgent(self.n),state,r,random.Random(0))
            self.assertEqual(expected.key(),c.state.key())
            replay=state.clone()
            for a in c.actions:replay=r.apply(replay,a)
            self.assertEqual(replay.key(),expected.key())

    def test_no_reply_budget_preserves_raw_turn_and_target(self):
        from aw_ai.search_training import search_turn
        _,state,_,_=next(fixtures());r=Rules();agent=deployment_agent(self.n)
        p=Planner(r,agent,agent,Config(seconds=0,nodes=0))
        expected,_=learned_turn(self.n,state.clone(),r,random.Random(0),False,True)
        a=p.analyze(state)
        self.assertEqual(len(a.alternatives),1)
        self.assertEqual(p.execute(state,a).key(),expected.key())
        self.assertEqual(a.metrics.counts['common_reply_rounds'],0)
        after,examples,_=search_turn(p,state.clone(),random.Random(5),True)
        self.assertEqual(after.key(),expected.key())
        self.assertTrue(all(len(e.target)==1 for e in examples))

    def test_only_complete_reply_rounds_are_committed(self):
        _,state,_,_=next(fixtures());agent=deployment_agent(self.n)
        p=Planner(Rules(),agent,agent,Config(seconds=3,nodes=1000))
        original=policy_turn
        calls=[]
        def interrupted(neural,s,r,rng,temperature=0.,budget=None,until=None):
            if budget is not None and until is None:
                calls.append(s.key())
                if len(calls)==2:return None
            return original(neural,s,r,rng,temperature,budget,until)
        with patch('aw_ai.policy_bundle.policy_turn',side_effect=interrupted):a=p.analyze(state)
        self.assertGreaterEqual(len(calls),2)
        self.assertEqual(a.metrics.counts['common_reply_rounds'],0)
        self.assertEqual(a.metrics.counts['raw_baseline_selected'],1)
        self.assertTrue(all(d['reply_rounds']==0 for d in a.bundle_diagnostics))

    def test_sampled_bundles_are_reproducible_and_legal(self):
        _,s,_,_=next(fixtures());r=Rules()
        a=policy_turn(NeuralAgent(self.n),s,r,random.Random(14),.7)
        b=policy_turn(NeuralAgent(self.n),s,r,random.Random(14),.7)
        self.assertEqual(a.actions,b.actions)
        state=s.clone()
        for x in a.actions:
            self.assertTrue(r.is_legal(state,x));state=r.apply(state,x)
        self.assertEqual(state.key(),a.state.key())

    def test_two_evaluation_maps_and_checkpoint_settings(self):
        import tempfile
        from pathlib import Path
        from aw_ai.scenarios import save,digest
        from aw_ai.search_training import run
        _,first,_,_=next(fixtures());second=first.clone();second.funds[0]+=1000
        observed=[]
        def evaluate(network,initial,config,games):
            observed.append([digest(initial(1000000+i)) for i in range(2)])
            return {'summary':{m:dict(win=2,loss=2,censored=0) for m in ('policy','beam')},'games':[]}
        with tempfile.TemporaryDirectory() as d:
            paths=[Path(d)/f'map{i}.json' for i in range(2)]
            save(first,paths[0]);save(second,paths[1]);target=Path(d)/'model.pt'
            with patch('aw_ai.search_training.evaluate',side_effect=evaluate):
                report=run(target,games=1,bootstrap=0,turns=1,map_path=paths[0],eval_maps=paths,
                           eval_games=4,tiny=True,roster='ita',updates_per_game=1,batch_size=4,train_seconds=.05,device='cpu')
            self.assertTrue(observed)
            self.assertTrue(all(a!=b for a,b in observed))
            loaded=Network.load(target)
            self.assertEqual(loaded.search_settings['bundle_version'],2)
            self.assertEqual(len(report['evaluation_maps']),2)
