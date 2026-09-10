import unittest
import tempfile
from pathlib import Path
import random
from dataclasses import replace
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'optional torch dependency not installed')
class NeuralTests(unittest.TestCase):
    def test_fixed_map_training_preserves_position_and_limit(self):
        from unittest.mock import patch
        from aw_ai.neural_training import run_legacy as run
        from aw_ai.scenarios import load, digest
        path=Path(__file__).resolve().parents[1]/'maps/v1/test_cities_6x6.json'
        expected=load(path); expected.income_capture_limit=6
        observed=[]
        def turn(*args,**kwargs):
            state=args[1]; observed.append(state)
            self.assertEqual(digest(state),digest(expected))
            changed=state.clone(); changed.winner=0
            return changed,[]
        with tempfile.TemporaryDirectory() as d, patch('aw_ai.neural_training.teacher_turn',side_effect=turn), patch('aw_ai.neural_training.learned_turn',side_effect=turn):
            report=run(Path(d)/'test.pt',games=2,bootstrap=1,turns=1,tiny=True,map_path=path,capture_limit=6)
        self.assertEqual(len(observed),5)
        self.assertEqual(report['income_capture_limit'],6)
        self.assertEqual(report['training_sizes'],[6])
        self.assertIn('same fixed map',report['validation_scope'])

    def setUp(self):
        from aw_ai.neural import Network, NetworkConfig
        from aw_ai.rules import Rules
        torch.set_num_threads(1); torch.manual_seed(3)
        self.network=Network(NetworkConfig(16,1,32,1,4)).eval(); self.rules=Rules()

    def test_variable_sizes_and_side_perspective(self):
        from aw_ai.neural_training import training_map, decision_logits
        for size in (8,12,20):
            s=training_map(3,size,3)
            choices,logits,value=decision_logits(self.network,s,self.rules)
            self.assertEqual(len(choices),len(logits)); self.assertTrue(torch.isfinite(logits).all())
            self.assertLessEqual(abs(value.item()),1)
            flipped=s.clone(); flipped.player=1-s.player; flipped.funds.reverse()
            flipped.owners={p:1-o for p,o in s.owners.items()}
            flipped.units={uid:replace(u,owner=1-u.owner) for uid,u in s.units.items()}
            _,other,v=decision_logits(self.network,flipped,self.rules)
            torch.testing.assert_close(logits,other); torch.testing.assert_close(value,v)

    def test_capture_limit_is_visible_to_network_and_default_training(self):
        from aw_ai.neural_training import training_map
        from aw_ai.model import INCOME_TILES
        s=training_map(0)
        self.assertEqual(s.income_capture_limit,(2*sum(t in INCOME_TILES for t in s.board.tiles)+2)//3)
        with torch.no_grad(): before=self.network.encode(s,self.rules)[0]
        s.income_capture_limit=None
        with torch.no_grad(): after=self.network.encode(s,self.rules)[0]
        self.assertFalse(torch.equal(before,after))

    def test_legal_masks_and_checkpoint_roundtrip(self):
        from aw_ai.neural import Network, NeuralAgent
        from aw_ai.neural_training import training_map, build_choices
        from aw_ai.strategy import Strategist
        from aw_ai.budget import Budget
        s=training_map(4); s.funds[0]=999
        agent=NeuralAgent(self.network); ctx=Strategist().plan(s,self.rules,Budget(5,1000))
        proposals=agent.propose(s,ctx,self.rules,limit=1000)
        self.assertTrue(proposals)
        self.assertTrue(all(self.rules.is_legal(s,p.action) for p in proposals))
        site=next(p for p in s.board.properties if s.board.tiles[p]=='factory')
        self.assertEqual(build_choices(s,self.rules,site),[11])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'n.pt'; self.network.save(path)
            loaded=NeuralAgent(Network.load(path))
            self.assertAlmostEqual(agent.evaluate(s,0,self.rules),loaded.evaluate(s,0,self.rules),places=5)

    def test_policy_production_and_value_receive_gradients(self):
        from aw_ai.neural_training import training_map, decision_logits
        s=training_map(0); _,logits,value=decision_logits(self.network,s,self.rules)
        site=next(p for p in s.board.properties if s.board.tiles[p]=='factory' and s.owners.get(p)==0)
        _,builds,_=decision_logits(self.network,s,self.rules,site)
        loss=-logits.log_softmax(0)[0]-builds.log_softmax(0)[0]+(value-1).square()
        loss.backward()
        for prefix in ('spatial','transformer','actor','command','destination','target','production','value'):
            gradients=[p.grad for n,p in self.network.named_parameters() if n.startswith(prefix)]
            self.assertTrue(any(g is not None and g.abs().sum()>0 for g in gradients),prefix)

    def test_bootstrap_learns_and_terminal_update_changes_value(self):
        from aw_ai.neural_training import training_map, decision_logits, update
        s=training_map(0); choices,logits,_=decision_logits(self.network,s,self.rules)
        examples=[(s,None,choices[0])]; opt=torch.optim.AdamW(self.network.parameters(),lr=.002)
        initial=-logits.log_softmax(0)[0].item()
        for _ in range(6): update(self.network,opt,examples,self.rules,random.Random(0),'bootstrap')
        self.assertLess(-decision_logits(self.network,s,self.rules)[1].log_softmax(0)[0].item(),initial)
        before=self.network.value[-2].weight.detach().clone()
        update(self.network,opt,examples,self.rules,random.Random(0),'reinforce',winner=0)
        self.assertFalse(torch.equal(before,self.network.value[-2].weight))

    def test_learned_build_head_drives_planner(self):
        from aw_ai.neural import NeuralAgent, KINDS
        from aw_ai.neural_training import training_map
        from aw_ai.policy import production_plans
        from aw_ai.budget import Budget
        s=training_map(0); s.funds[0]=8000
        with torch.no_grad():
            self.network.production[-1].weight.zero_(); self.network.production[-1].bias.fill_(-10)
            self.network.production[-1].bias[KINDS.index('anti_air')]=10
        agent=NeuralAgent(self.network)
        plans=production_plans(s,self.rules,Budget(5,1000),4,scorer=agent.build_scores)
        best=max(plans,key=lambda x:x[2])
        self.assertTrue(any(a.build=='anti_air' for a in best[1]))
