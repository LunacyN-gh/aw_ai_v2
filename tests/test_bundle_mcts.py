import random,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from aw_ai.bundle_mcts import Node,Edge,backup,select,priors,analyze,validate
from aw_ai.planner import Planner,Config,Candidate
from aw_ai.model import Action,END
from aw_ai.scenarios import load,digest
from aw_ai.neural import Network,NetworkConfig
from aw_ai.guidance import deployment_agent
from aw_ai.search_training import search_turn,teacher_planner


class MathTests(unittest.TestCase):
    def test_opponent_minimizes_root_value(self):
        root=Node(SimpleNamespace(player=0));opponent=Node(SimpleNamespace(player=1))
        edges=[Edge(None,root,visits=10,total=8),Edge(None,root,visits=10,total=-8)]
        root.edges=opponent.edges=edges
        self.assertIs(select(root,0,0),edges[0])
        self.assertIs(select(opponent,0,0),edges[1])
        backup([root,opponent],[edges[0]],-1)
        self.assertEqual(edges[0].total,7);self.assertEqual(opponent.visits,1)

    def test_normalized_priors_and_invalid_config(self):
        edges=[Edge(None,None,log_prior=-1),Edge(None,None,log_prior=-100)]
        priors(edges);self.assertAlmostEqual(sum(e.prior for e in edges),1)
        self.assertGreater(edges[1].prior,0)
        for settings in ({'mcts_depth':0},{'mcts_c_puct':float('nan')},{'mcts_simulations':1.5}):
            with self.assertRaises(ValueError):validate(settings)

    def test_three_turn_tree_finds_safe_win_against_adversarial_reply(self):
        class State:
            def __init__(self,i):self.i=i;self.player={0:0,1:1,2:1,3:0,4:0,5:1}[i]
            def key(self):return self.i
            def clone(self):return State(self.i)
            def validate(self):pass
        graph={0:[1,2],1:[5,4],2:[3],3:[5]}
        outcome=lambda s:{4:1,5:0}.get(s.i)
        settings=dict(planner='mcts',mcts_simulations=250,mcts_depth=6,mcts_c_puct=1.5,mcts_widening=2.,mcts_max_children=4)
        ctx=SimpleNamespace(objectives={},regions=[])
        strategy=SimpleNamespace(plan=lambda *a,**kw:ctx)
        rules=SimpleNamespace(outcome=outcome)
        neural=SimpleNamespace(network=SimpleNamespace(allowed_builds=('infantry',)))
        planner=SimpleNamespace(policy=SimpleNamespace(network=SimpleNamespace(search_settings=settings),neural=neural),rules=rules,
                                config=Config(seconds=5,nodes=2000),value=None,strategist=strategy)
        counts={}
        def proposal(neural,s,rules,rng,*args,**kwargs):
            choices=graph[s.i];i=counts.get(s.i,0);counts[s.i]=i+1
            dest=choices[i%len(choices)]
            return Candidate((Action('wait',str(dest),dest),END),State(dest),0.,'toy')
        class Adapter:
            def __init__(self,*args,**kwargs):self.strategist=strategy
            def _score(self,s,p):return 0.
            def _complete(self,prefix,*args):return proposal(None,prefix.state,None,None)
            def analyze(self,s):
                c=proposal(None,s,None,None)
                return SimpleNamespace(alternatives=[c],actions=c.actions,metrics=SimpleNamespace(counts={'nodes':0}))
        with patch('aw_ai.bundle_mcts.OrderPlanner',Adapter),patch('aw_ai.bundle_mcts.policy_turn',proposal),patch('aw_ai.bundle_mcts.preference',return_value=0.):
            result=analyze(planner,State(0))
        self.assertEqual(result.actions[0].destination,2)
        self.assertGreaterEqual(result.metrics.counts['mcts_max_depth'],3)
        self.assertGreater(result.visit_weights[result.actions],.5)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def agent(self):
        network=Network(NetworkConfig(16,1,32,1,4));network.allowed_builds=('infantry','tank','artillery')
        network.search_settings=dict(mode='guided',planner='mcts',neural_value_weight=1.,mcts_simulations=12)
        return deployment_agent(network)

    def test_legal_replay_visit_targets_and_zero_budget(self):
        state=load('maps/6x6_ita/test_cities_6x6_ita.json');key=state.key()
        agent=self.agent();planner=Planner(policy=agent,value=agent,config=Config(seconds=1))
        analysis=planner.analyze(state)
        self.assertAlmostEqual(sum(analysis.visit_weights.values()),1.)
        self.assertTrue(all(c.complete for c in analysis.alternatives))
        planner.execute(state,analysis)
        self.assertEqual(state.key(),key)
        after,examples,_=search_turn(planner,state,random.Random(1))
        self.assertTrue(examples)
        for e in examples:self.assertAlmostEqual(sum(e.target.values()),1.)
        zero=planner.analyze(state,Config(seconds=0,nodes=0))
        self.assertEqual(len(zero.alternatives),1)
        planner.execute(state,zero)

    def test_training_uses_visits_not_q_and_checkpoint_load(self):
        state=load('maps/6x6_ita/test_cities_6x6_ita.json');agent=self.agent()
        planner=Planner(policy=agent,value=agent,config=Config(seconds=1))
        analysis=planner.analyze(state)
        for c in analysis.alternatives:c.score=-9999
        with patch.object(planner,'analyze',return_value=analysis):
            _,examples,_=search_turn(planner,state,random.Random(3))
        first=next(e for e in examples if e.site is None)
        expected={}
        for c in analysis.alternatives:
            a=c.actions[0];expected[a]=expected.get(a,0)+analysis.visit_weights[c.actions]
        self.assertEqual(first.target,expected)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'mcts.pt';agent.network.save(path)
            self.assertEqual(Network.load(path).search_settings['planner'],'mcts')
        from aw_ai.order_search import OrderPlanner
        self.assertIsInstance(teacher_planner(planner.rules,agent.network.allowed_builds,Config(),'ordered'),OrderPlanner)
