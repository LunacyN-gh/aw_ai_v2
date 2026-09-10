import copy
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from aw_ai.model import Board,State,Unit,Action,END
from aw_ai.rules import Rules
from aw_ai.planner import Planner,Config,Candidate
from aw_ai.guidance import GuidedAgent,deployment_agent
from aw_ai.neural import Network,NetworkConfig
from aw_ai.search_training import Example,fit,candidate_weights,search_turn


class BundleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.rules=Rules()
        self.state=State(Board.plain(5,2),{'a':Unit('a',0,'infantry',0),'e':Unit('e',1,'infantry',9)})

    def network(self):
        n=Network(NetworkConfig(16,1,32,1,4))
        n.allowed_builds=('infantry','tank','artillery')
        n.search_settings=dict(mode='guided',planner='bundle',neural_value_weight=.5,bundle_candidates=6)
        return n

    def test_value_prediction_changes_selected_bundle(self):
        candidates=[]
        for p in (0,1):
            actions=(Action('wait','a',p),END); s=self.state
            for a in actions:s=self.rules.apply(s,a)
            candidates.append(Candidate(actions,s,0,'fixture'))
        candidates[0].variants=[candidates[1]]
        class Fake:
            network=SimpleNamespace(allowed_builds=('infantry',),search_settings=dict(planner='bundle',neural_value_weight=1.))
            def evaluate(self,s,p,r): return (1 if s.units['a'].pos==1 else -1)*10000*self.sign
            sign=1
        fake=Fake(); agent=GuidedAgent(fake,1)
        planner=Planner(self.rules,agent,agent,Config(seconds=0,nodes=0))
        with patch.object(Planner,'_finish',return_value=candidates[0]):
            self.assertEqual(planner.analyze(self.state).actions[0].destination,1)
            fake.sign=-1
            self.assertEqual(planner.analyze(self.state).actions[0].destination,0)

    def test_bounded_score_temperature_is_not_heuristic_temperature(self):
        cs=[SimpleNamespace(score=x) for x in (0,.5)]
        self.assertGreater(candidate_weights(cs,.15)[1],.96)
        self.assertLess(candidate_weights(cs)[1],.501)

    def test_value_gradient_not_diluted_by_unlabeled_corrections(self):
        torch.manual_seed(4); n=self.network(); other=copy.deepcopy(n)
        e=Example(self.state,None,{Action('wait','a',0):1.},1.)
        correction=copy.copy(e);correction.result=None
        for net,batch in ((n,[e]),(other,[e,correction,correction,correction])):
            with patch('aw_ai.search_training.training_batch',return_value=batch):
                fit(net,torch.optim.SGD(net.parameters(),lr=0),batch,self.rules,random.Random(0),1,4)
        for p,q in zip(n.value.parameters(),other.value.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)

    def test_bundle_round_trip_and_legal_targets(self):
        n=self.network()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'n.pt';n.save(path);n=Network.load(path)
        agent=deployment_agent(n)
        p=Planner(self.rules,agent,agent,Config(seconds=2,nodes=4000))
        original=self.state.key()
        after,examples,_=search_turn(p,self.state.clone(),random.Random(0),True)
        after.validate();self.assertEqual(self.state.key(),original)
        self.assertTrue(examples)
        for e in examples:
            self.assertAlmostEqual(sum(e.target.values()),1.)
            self.assertTrue(all(self.rules.is_legal(e.state,a) for a in e.target))

    def test_zero_budget_has_legal_fallback(self):
        agent=deployment_agent(self.network())
        p=Planner(self.rules,agent,agent,Config(seconds=0,nodes=0))
        result=p.analyze(self.state)
        p.execute(self.state,result).validate()
        self.assertFalse(result.alternatives[0].complete)
