import unittest
from unittest.mock import patch
from types import SimpleNamespace
from aw_ai.model import Action,END
from aw_ai.scenarios import tactical,digest
from aw_ai.review import rank_search_actions
from aw_ai.planner import Candidate

class SearchReviewTests(unittest.TestCase):
    def test_groups_first_actions_and_uses_full_continuations(self):
        state=tactical('blocker');before=digest(state)
        first=Action('wait','friend',7)
        candidates=[Candidate((first,END),state,2.,'test'),Candidate((first,),state,1.,'other')]
        analysis=SimpleNamespace(alternatives=candidates,actions=candidates[0].actions)
        with patch('aw_ai.guidance.deployment_agent',return_value=object()),patch('aw_ai.planner.Planner') as factory:
            factory.return_value.analyze.return_value=analysis
            rows=rank_search_actions(object(),state,5,.5)
            self.assertEqual(factory.call_args.kwargs['config'].seconds,.5)
        self.assertEqual(len(rows),1)
        self.assertIn('End turn',rows[0]['plan'])
        self.assertTrue(rows[0]['recommended'])
        self.assertEqual(digest(state),before)

    def test_terminal_position_has_no_candidates(self):
        state=tactical('blocker');state.winner=0
        self.assertEqual(rank_search_actions(object(),state),[])
