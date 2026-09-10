import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from aw_ai.scenarios import load,digest
from aw_ai.rules import Rules

ROOT=Path(__file__).resolve().parents[1]

class MultiMapTests(unittest.TestCase):
    def test_ita_copy_preserves_layout_and_restricts_both_sides(self):
        original=json.loads((ROOT/'maps/v1/second_map_6x6.json').read_text())
        copied=json.loads((ROOT/'maps/second_map_6x6_ita.json').read_text())
        self.assertEqual(copied.pop('allowed_builds'),['infantry','tank','artillery'])
        self.assertEqual(copied.pop('income_capture_limit'),7)
        self.assertEqual(original,copied)
        state=load(ROOT/'maps/second_map_6x6_ita.json')
        for side in (0,1):
            self.assertEqual(set(Rules().roster('factory',state)),{'infantry','tank','artillery'})
            self.assertTrue(any(owner==side for owner in state.owners.values()))
        self.assertEqual(state.income_capture_limit,7)

    def test_train_and_evaluate_both_maps_without_opponent_side_bias(self):
        from aw_ai.search_training import run
        paths=[ROOT/'maps/test_cities_6x6_ita.json',ROOT/'maps/second_map_6x6_ita.json']
        observed=[];evals=[]
        def job(j):
            observed.append((digest(j['state']),j['opponent'],j['side']))
            return [],0,0,0,0.,dict(corrections=[])
        def evaluate(n,initial,config,games):
            evals.append([digest(initial(1000000+i)) for i in range(2)])
            return dict(summary={m:dict(win=2,loss=2,censored=0) for m in ('policy','beam')},games=[])
        with tempfile.TemporaryDirectory() as d,patch('aw_ai.search_training.play_job',side_effect=job),patch('aw_ai.search_training.fit',return_value=None),patch('aw_ai.search_training.evaluate',side_effect=evaluate):
            report=run(Path(d)/'test.pt',train_maps=paths,games=16,bootstrap=0,turns=1,tiny=True,
                       roster='ita',device='cpu',workers=1,eval_games=4)
        self.assertEqual(len(set(observed)),16) # two maps x four opponents x both sides
        self.assertTrue(all(a!=b for a,b in evals))
        self.assertEqual(len(report['training_maps']),2)
        self.assertEqual(len(set(r['training_map_hash'] for r in report['runs'])),2)
