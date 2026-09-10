"""Numerical batching and spawn/round-boundary regressions."""
import copy
import random
import tempfile
import unittest
from pathlib import Path
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'optional torch dependency unavailable')
class BatchingTests(unittest.TestCase):
    def setUp(self):
        from aw_ai.neural import Network, NetworkConfig
        from aw_ai.model import Board, State, Unit, Action
        from aw_ai.search_training import Example, ITA
        from aw_ai.rules import Rules
        torch.set_num_threads(1); torch.manual_seed(7)
        self.network = Network(NetworkConfig(16,1,32,1,4))
        self.network.allowed_builds = ITA
        self.rules = Rules()
        self.examples = []
        for i, (w,h) in enumerate(((6,6),(6,6),(8,4))):
            s = State(Board.plain(w,h,{(0,0):'factory',(2,0):'city'}),
                      {'a':Unit('a',0,'infantry',1),'e':Unit('e',1,'tank',w*h-1)},
                      funds=[12000,3000],owners={0:0},income_capture_limit=2)
            if i == 1:
                s.units['b'] = Unit('b',0,'artillery',w+1)
                s.__post_init__()
            self.examples += [Example(s,None,{Action('wait','a',1):.3,Action('wait','a',2):.7},1.),
                              Example(s,0,{3:.6,11:.4},None)]

    def test_logits_and_gradients_match_scalar_mixed_shapes(self):
        from aw_ai.neural_training import decision_logits
        reference = copy.deepcopy(self.network)
        self.network.train(); reference.train()
        choices, logits, values = self.network.decision_batch(self.examples,self.rules)
        scalar_loss = 0
        for i,e in enumerate(self.examples):
            c,l,v = decision_logits(reference,e.state,self.rules,e.site)
            self.assertEqual(choices[i],c)
            torch.testing.assert_close(logits[i],l,rtol=2e-5,atol=2e-6)
            torch.testing.assert_close(values[i],v,rtol=2e-5,atol=2e-6)
            scalar_loss = scalar_loss+l.square().mean()+v.square()
        scalar_loss.backward()
        (sum(l.square().mean() for l in logits)+values.square().sum()).backward()
        for actual,expected in zip(self.network.parameters(),reference.parameters()):
            if expected.grad is not None:
                torch.testing.assert_close(actual.grad,expected.grad,rtol=2e-4,atol=1e-5)

    def test_gpu_masks_and_censored_value_head(self):
        from aw_ai.search_training import fit
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable')
        self.network.cuda()
        examples = [copy.copy(e) for e in self.examples]
        for e in examples: e.result = None
        before = copy.deepcopy(self.network.value.state_dict())
        metrics = fit(self.network,torch.optim.AdamW(self.network.parameters()),examples,self.rules,random.Random(0),1,8)
        self.assertTrue(all(v is None or __import__('math').isfinite(v) for v in metrics.values()))
        self.assertIsNone(metrics['value_mse'])
        for k,v in before.items(): torch.testing.assert_close(v,self.network.value.state_dict()[k],rtol=0,atol=0)

    def test_soft_target_loss_and_update_match_scalar_reference(self):
        from aw_ai.neural_training import decision_logits
        from aw_ai.search_training import fit
        reference = copy.deepcopy(self.network)
        optimizer = torch.optim.SGD(reference.parameters(),lr=.001)
        losses = []
        for e in self.examples:
            choices,logits,value = decision_logits(reference,e.state,self.rules,e.site)
            log_probs = logits.log_softmax(0)
            target = torch.tensor([e.target.get(a,0.) for a in choices])
            loss = -(target*log_probs).sum()+.01*(log_probs.exp()*log_probs).sum()
            if e.result is not None:
                loss = loss+.5*(value-e.result).square()*len(self.examples)/sum(x.result is not None for x in self.examples)
            losses.append(loss.item())
            (loss/len(self.examples)).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
        optimizer.step()
        metrics = fit(self.network,torch.optim.SGD(self.network.parameters(),lr=.001),
                      self.examples,self.rules,random.Random(3),1,6)
        self.assertAlmostEqual(metrics['total_loss'],sum(losses)/len(losses),places=5)
        self.assertEqual(metrics['value_samples'],3)
        for actual,expected in zip(self.network.parameters(),reference.parameters()):
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)


@unittest.skipIf(torch is None, 'optional torch dependency unavailable')
class WorkerTests(unittest.TestCase):
    def test_spawn_games_preserve_targets_and_capture_rule(self):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        from aw_ai.search_training import play_job, worker_init, ITA
        from aw_ai.scenarios import load,digest
        from aw_ai.planner import Config
        state = load(Path(__file__).resolve().parents[1]/'maps/v1/test_cities_6x6.json')
        job = dict(state=state,seed=4,boot=True,opponent='teacher',side=0,allowed=ITA,
                   search=Config(seconds=30,nodes=150),turns=4,current=None,other=None)
        serial = play_job(job)
        with ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn'),initializer=worker_init) as pool:
            results = list(pool.map(play_job,[job,job]))
        for result in results:
            self.assertEqual(serial[1:4],result[1:4])
            self.assertEqual(len(serial[0]),len(result[0]))
            for a,b in zip(serial[0],result[0]):
                self.assertEqual((digest(a.state),a.site,a.target,a.result),
                                 (digest(b.state),b.site,b.target,b.result))
                self.assertEqual(b.state.income_capture_limit,7)


if __name__ == '__main__':
    unittest.main()
