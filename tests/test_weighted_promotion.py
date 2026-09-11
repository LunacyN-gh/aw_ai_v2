import unittest
from unittest.mock import patch
from aw_ai.promotion import match_score,promotion_scores,promotion_decision


class WeightedPromotionTests(unittest.TestCase):
    def report(self,c=(8,8,0),i=(8,8,0),h=(8,8,0)):
        return {'summary':{k:dict(zip(('win','loss','censored'),v)) for k,v in zip(('beam','incumbent','gate'),(c,i,h))}}

    def test_half_points_and_ties(self):
        self.assertEqual(match_score(dict(win=4,loss=4,censored=8)),.5)
        self.assertEqual(match_score(dict(win=0,loss=0,censored=16)),.5)
        self.assertFalse(promotion_decision(self.report(),margin=0)[0])

    def test_teacher_drop_can_be_offset_by_head_to_head(self):
        r=self.report(c=(6,10,0),i=(8,8,0),h=(12,4,0))
        self.assertAlmostEqual(promotion_scores(r)['combined_delta'],.1875)
        self.assertTrue(promotion_decision(r)[0])

    def test_margin_and_no_historical_fallback(self):
        r=self.report(h=(9,7,0))
        self.assertTrue(promotion_decision(r)[0])
        self.assertFalse(promotion_decision(r,margin=.0625)[0])
        del r['summary']['incumbent']
        self.assertFalse(promotion_decision(r,self.report())[0])

    def test_current_incumbent_evaluated_on_same_maps_sides_budget(self):
        from aw_ai.search_training import evaluate
        from aw_ai.neural import Network,NetworkConfig
        from aw_ai.opening_suite import fixtures
        from aw_ai.planner import Config
        n=Network(NetworkConfig(16,1,32,1,4))
        _,state,_,_=next(fixtures());seen=[];incumbent={'marker':'frozen'}
        def worker(job):
            seen.append(job)
            return dict(mode=job['mode'],result='censored')
        with patch('aw_ai.search_training.evaluation_job',side_effect=worker),patch('aw_ai.opening_suite.opening_checks',return_value={}):
            r=evaluate(n,lambda seed:state.clone(),dict(reference=incumbent,turns=50,search=Config()),4)
        groups={m:[j for j in seen if j['mode']==m] for m in ('beam','gate','incumbent')}
        for c,i,h in zip(groups['beam'],groups['incumbent'],groups['gate']):
            self.assertEqual((c['state'].key(),c['seed'],c['player'],c['search']),
                             (i['state'].key(),i['seed'],i['player'],i['search']))
            self.assertIs(i['network'],incumbent)
            self.assertNotIn('opponent',i)
            self.assertIs(h['opponent'],incumbent)
        self.assertEqual(r['summary']['incumbent']['censored'],4)
