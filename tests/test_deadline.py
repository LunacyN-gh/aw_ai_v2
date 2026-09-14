import json
import random
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from aw_ai.model import Action, Board, DRAW, END, State, Unit, outcome_value
from aw_ai.rules import Rules
from aw_ai.scenarios import digest, from_data, load, to_data
from aw_ai.neural import Network, NetworkConfig, NeuralAgent, migrate_checkpoint
from aw_ai.evaluation import HeuristicValue
from aw_ai.planner import Config, Planner
from aw_ai.search_training import (Example, label_result, fit, model_payload,
                                   restore_model, evaluation_job, play_job, termination_reason)
from aw_ai.promotion import match_score


def position(limit=2):
    board=Board.plain(4,2,{(0,0):'factory',(3,1):'factory',(1,0):'city'})
    return State(board,{},owners={0:0,7:1},turn_limit=limit)


class DeadlineRulesTests(unittest.TestCase):
    def test_deadline_after_both_players_no_extra_income_or_repair(self):
        rules=Rules(); s=position(); s.owners[1]=1
        s.units['a']=Unit('a',0,'infantry',0,hp=60); s.__post_init__()
        rules.start_turn(s)
        s=rules.apply(s,END)
        self.assertIsNone(rules.outcome(s))
        self.assertEqual(s.funds[1],2000)
        funds=s.funds.copy(); hp=s.units['a'].hp
        s=rules.apply(s,END)
        self.assertEqual((s.turn,rules.outcome(s)),(2,1))
        self.assertEqual(s.funds,funds)
        self.assertEqual(s.units['a'].hp,hp)
        self.assertFalse(list(rules.legal_actions(s)))
        with self.assertRaises(ValueError): rules.apply(s,END)

    def test_equal_income_draw_even_with_unequal_funds_and_army(self):
        rules=Rules(); s=position(1); s.funds=[99000,0]
        s.units['tank']=Unit('tank',0,'tank',2); s.__post_init__()
        s=rules.apply(s,END)
        self.assertEqual(rules.outcome(s),DRAW)
        for player in (0,1):
            self.assertEqual(outcome_value(DRAW,player),0.)
            self.assertEqual(HeuristicValue().evaluate(s,player,rules),0.)
        self.assertEqual(termination_reason(s,rules,{0:0,7:1}),'deadline_draw')
        self.assertEqual(digest(from_data(to_data(s))),digest(s))

    def test_final_turn_capture_changes_deadline_winner(self):
        rules=Rules(); s=position(2); s.turn=1; s.player=1
        s.units['cap']=Unit('cap',1,'infantry',1); s.__post_init__()
        s.captures[1]=('cap',10)
        s=rules.apply(s,Action('capture','cap',1))
        self.assertIsNone(rules.outcome(s))
        s=rules.apply(s,END)
        self.assertEqual(rules.outcome(s),1)

    def test_early_capture_victory_and_elimination_precede_deadline(self):
        rules=Rules(); s=position(36); s.income_capture_limit=2
        s.units['cap']=Unit('cap',0,'infantry',1); s.__post_init__(); s.captures[1]=('cap',10)
        s=rules.apply(s,Action('capture','cap',1))
        self.assertEqual((s.turn,rules.outcome(s)),(0,0))
        s=position(1); s.owners={0:0}; s.turn=1
        self.assertEqual(rules.outcome(s),0)

    def test_clock_roundtrip_cache_keys_and_legacy_maps(self):
        s=position(36); later=s.clone(); later.turn=1
        self.assertNotEqual(s.key(),later.key()); self.assertNotEqual(digest(s),digest(later))
        self.assertEqual(from_data(to_data(later)).turn_limit,36)
        s.turn_limit=None; later.turn_limit=None
        self.assertEqual(s.key(),later.key())
        self.assertNotIn('turn_limit',to_data(s))
        self.assertIsNone(from_data(to_data(s)).turn_limit)
        for limit in (0,-1,True,1.5):
            s.turn_limit=limit
            with self.assertRaises(ValueError): s.validate()

    def test_map_pool_deadlines(self):
        paths=list((Path(__file__).resolve().parents[1]/'maps/6x6_ita').glob('*.json'))
        self.assertEqual(len(paths),8)
        for path in paths:
            s=load(path)
            self.assertEqual(s.turn_limit,40)
            self.assertEqual(s.turn,0)


class DeadlineNeuralTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(4)
        self.net=Network(NetworkConfig(16,1,32,1,4)).eval()
        self.rules=Rules()

    def test_legacy_migration_preserves_logits_value_and_all_old_weights(self):
        # Construct the actual old architecture: seven global scalars.
        old=Network(self.net.config).eval()
        old.global_token=torch.nn.Linear(23,32)
        original_inputs=old.inputs
        def legacy_inputs(state,rules):
            grid,ids,positions,features,economy=original_inputs(state,rules)
            return grid,ids,positions,features,economy[:7]
        old.inputs=legacy_inputs
        s=position(36); s.funds=[7000,7000]
        actions=[a for a in self.rules.legal_actions(s) if a.kind!='build']
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'old.pt'; dest=Path(folder)/'clock.pt'
            old.save(source,{'tag':'preserved'})
            data=torch.load(source,weights_only=True); data['version']=2; torch.save(data,source)
            before=source.read_bytes()
            migrate_checkpoint(source,dest)
            migrated=Network.load(dest)
            self.assertEqual(source.read_bytes(),before)
            self.assertEqual(torch.load(dest,weights_only=True)['metadata']['tag'],'preserved')
            for name,weights in old.state_dict().items():
                actual=migrated.state_dict()[name]
                torch.testing.assert_close(actual[:,:23] if name=='global_token.weight' else actual,weights,rtol=0,atol=0)
            self.assertEqual(migrated.global_token.weight[:,23:].count_nonzero().item(),0)
            with torch.no_grad():
                for turn in (0,17,35):
                    s.turn=turn
                    a=old.encode(s,self.rules); b=migrated.encode(s,self.rules)
                    torch.testing.assert_close(old.action_logits(a,actions),migrated.action_logits(b,actions),rtol=1e-5,atol=1e-6)
                    torch.testing.assert_close(old.value(a[0]),migrated.value(b[0]),rtol=1e-5,atol=1e-6)
                    torch.testing.assert_close(old.build_logits(a,s,self.rules,0),migrated.build_logits(b,s,self.rules,0),rtol=1e-5,atol=1e-6)
            with self.assertRaises(ValueError): migrate_checkpoint(source,dest)

    def test_clock_features_and_worker_payload(self):
        s=position(36); s.turn=34
        torch.testing.assert_close(self.net.inputs(s,self.rules)[-1][-3:],torch.tensor([1.,.02,.36]))
        other=restore_model(model_payload(self.net))
        torch.testing.assert_close(self.net.encode(s,self.rules)[0],other.encode(s,self.rules)[0])

    def test_draw_targets_train_value_instead_of_being_censored(self):
        s=position(); s.funds=[1000,0]
        samples=[Example(s,0,{len(self.net.allowed_builds):1.})]
        labeled=label_result(samples,DRAW)
        self.assertEqual(labeled[0].result,0.)
        self.assertIsNone(label_result(samples,None)[0].result)
        opt=torch.optim.AdamW(self.net.parameters(),lr=.001)
        before=self.net.value[0].weight.detach().clone()
        metrics=fit(self.net,opt,labeled,self.rules,random.Random(0),1,1)
        self.assertEqual(metrics['value_samples'],1)
        self.assertFalse(torch.equal(before,self.net.value[0].weight))
        self.assertEqual(match_score(dict(win=1,loss=1,draw=2,censored=0)),.5)

    def test_all_search_modes_value_draw_as_zero(self):
        from aw_ai.guidance import deployment_agent
        from aw_ai.tree import SampledPlanner
        s=position(1); s.turn=1
        for version in (1,2):
            self.net.search_settings=dict(mode='guided',planner='bundle',bundle_version=version,neural_value_weight=.5,bundle_candidates=2)
            agent=deployment_agent(self.net)
            planner=Planner(policy=agent,value=agent,config=Config(seconds=.1,nodes=100))
            self.assertEqual(planner.analyze(s).score,0.)
        self.assertEqual(NeuralAgent(self.net).evaluate(s,0,self.rules),0.)
        self.assertEqual(Planner().analyze(s).score,0.)
        self.assertEqual(SampledPlanner().analyze(s).score,0.)

    def test_real_worker_games_report_draws_and_income_wins(self):
        s=position(2)
        payload=model_payload(self.net)
        job=dict(network=payload,state=s,seed=1,player=0,mode='policy',turns=2,search=Config(seconds=.05,nodes=40))
        result=evaluation_job(job)
        self.assertEqual((result['winner'],result['result'],result['termination']),(DRAW,'draw','deadline_draw'))
        s=position(2); s.owners[1]=0
        job['state']=s
        result=evaluation_job(job)
        self.assertEqual((result['winner'],result['result'],result['termination']),(0,'win','deadline_income'))


if __name__=='__main__': unittest.main()
