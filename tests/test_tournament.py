import unittest
from aw_ai.tournament import schedule,summarize,run_game
from aw_ai.model import Board,State
from aw_ai.scenarios import to_data,digest
from aw_ai.notation import loads

class TournamentTests(unittest.TestCase):
    def test_sixteen_games_per_pair(self):
        entrants=[dict(name=x) for x in ('teacher','50','75','100')]
        jobs=schedule(entrants,range(8))
        self.assertEqual(len(jobs),96)
        self.assertEqual(len({g['id'] for g in jobs}),96)
        from collections import Counter
        self.assertEqual(set(Counter(tuple(g['pair']) for g in jobs).values()),{16})
        self.assertEqual(set(Counter((tuple(g['pair']),g['map_index'],g['blue']) for g in jobs).values()),{1})

    def test_scores_use_player_identity_and_draw_half_points(self):
        jobs=schedule([dict(name='a'),dict(name='b')],[0,1])
        for g,w in zip(jobs,(0,0,-1,None)):g['winner']=w
        s=summarize(['a','b'],jobs)
        for name in ('a','b'):
            self.assertEqual(s['standings'][name]['wins'],1)
            self.assertEqual(s['standings'][name]['losses'],1)
            self.assertEqual(s['standings'][name]['draws'],1)
            self.assertEqual(s['standings'][name]['censored'],1)
            self.assertEqual(s['standings'][name]['score_rate'],.5)

    def test_worker_plays_and_roundtrips_replay(self):
        b=Board.plain(2,1,{(0,0):'factory',(1,0):'factory'})
        state=State(b,{},owners={0:0,1:1},turn_limit=2)
        entrants=[dict(name='a',type='heuristic'),dict(name='b',type='heuristic')]
        maps=[dict(name='test',state=to_data(state))]
        manifest=dict(entrants=entrants,maps=maps,seconds=.05,nodes=100,max_turns=2)
        game=schedule(entrants,maps)[0]
        row,pgn=run_game(dict(game=game,manifest=manifest,output='.'))
        self.assertNotIn('error',row)
        self.assertEqual(row['winner'],-1)
        self.assertEqual(row['termination'],'deadline_draw')
        self.assertEqual(row['final_state_sha256'],digest(loads(pgn).states[-1]))

if __name__=='__main__':unittest.main()
