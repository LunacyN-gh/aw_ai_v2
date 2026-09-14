"""Configurable, resumable side-swapped round robin with frozen inputs and AWPGN."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
from time import perf_counter


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path,data):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)

def schedule(entrants,maps):
    return [dict(id=f'g{n:04d}',pair=[a['name'],b['name']],map_index=m,swap=swap,
                 blue=(b if swap else a)['name'],red=(a if swap else b)['name'])
            for n,(a,b,m,swap) in enumerate(((a,b,m,swap) for a,b in itertools.combinations(entrants,2)
                                            for m in range(len(maps)) for swap in (0,1)),1)]

def summarize(names,games):
    totals={name:dict(played=0,wins=0,losses=0,draws=0,censored=0,errors=0,points=0.) for name in names}
    pairs={}
    for game in games:
        key=' vs '.join(game['pair'])
        pair=pairs.setdefault(key,{name:dict(wins=0,losses=0,draws=0,censored=0,errors=0,points=0.) for name in game['pair']})
        for side,name in enumerate((game['blue'],game['red'])):
            if game.get('error'):outcome='errors';points=0
            elif game['winner'] is None:outcome='censored';points=0
            elif game['winner']==-1:outcome='draws';points=.5
            elif game['winner']==side:outcome='wins';points=1
            else:outcome='losses';points=0
            totals[name]['played']+=1
            for row in (totals[name],pair[name]):row[outcome]+=1;row['points']+=points
    for row in totals.values():
        complete=row['wins']+row['losses']+row['draws']
        row['score_rate']=row['points']/complete if complete else None
    return dict(standings=totals,pairs=pairs)

def prepare(config_path,output):
    from .scenarios import load,to_data,digest
    from .planner import Config
    config_path=Path(config_path).resolve();output=Path(output).resolve()
    config=json.loads(config_path.read_text(encoding='utf-8-sig'))
    entrants=config['entrants'];maps=config['maps']
    if len(entrants)<2 or not maps:raise ValueError('Need at least two entrants and one map')
    names=[e['name'] for e in entrants]
    if any(not isinstance(n,str) or not n.strip() for n in names) or len(set(names))!=len(names):
        raise ValueError('Entrant names must be nonempty and unique')
    seconds=config.get('seconds',1.);nodes=config.get('nodes',4000)
    Config(seconds=seconds,nodes=nodes)
    max_turns=config.get('max_turns',100)
    if type(max_turns) is not int or max_turns<1:raise ValueError('max_turns must be a positive safety limit')
    deadline=config.get('turn_limit')
    if deadline is not None and (type(deadline) is not int or deadline<1):raise ValueError('Invalid turn_limit')
    frozen_maps=[];hashes=[]
    for path in maps:
        path=(config_path.parent/path).resolve();state=load(path)
        if deadline is not None:state.turn_limit=state.turn+deadline
        state.validate()
        from .rules import Rules
        if Rules().outcome(state) is not None:raise ValueError(f'Map already terminal: {path}')
        if state.turn_limit and state.turn_limit-state.turn>max_turns:
            raise ValueError('max_turns must cover the game deadline')
        frozen_maps.append(dict(name=path.name,path=str(path),file_sha256=sha(path),state=to_data(state),state_sha256=digest(state)))
        hashes.append(digest(state))
    if len(set(hashes))!=len(hashes):raise ValueError('Duplicate initial map positions')
    frozen_entrants=[]
    for entrant in entrants:
        if entrant.get('search','old') not in ('old','ordered','mcts'):raise ValueError('search must be old, ordered or mcts')
        kind=entrant.get('type','neural')
        if kind not in ('heuristic','neural'):raise ValueError('Entrant type must be heuristic or neural')
        frozen=dict(entrant,type=kind)
        if kind=='neural':
            path=(config_path.parent/entrant['checkpoint']).resolve()
            frozen['checkpoint_sha256']=sha(path);frozen['source_checkpoint']=str(path)
            frozen['checkpoint']='inputs/'+frozen['checkpoint_sha256']+'.pt'
            weight=entrant.get('neural_value_weight')
            if weight is not None and (not isinstance(weight,(int,float)) or not math.isfinite(weight) or not 0<=weight<=1):
                raise ValueError('Neural value weight must be in [0,1]')
        frozen_entrants.append(frozen)
    manifest=dict(format='aw-ai-tournament',version=1,config_path=str(config_path),
                  seconds=seconds,nodes=nodes,max_turns=max_turns,maps=frozen_maps,entrants=frozen_entrants,
                  code_sha256={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    if (output/'manifest.json').exists():
        old=json.loads((output/'manifest.json').read_text())
        if old!=manifest:raise ValueError('Inputs/settings changed; choose a new output directory')
        for e in frozen_entrants:
            if e['type']=='neural' and sha(output/e['checkpoint'])!=e['checkpoint_sha256']:
                raise ValueError('Frozen checkpoint hash mismatch')
    else:
        if output.exists() and any(output.iterdir()):raise ValueError('Output must be empty or a matching tournament')
        (output/'inputs').mkdir(parents=True,exist_ok=True)
        for e in frozen_entrants:
            if e['type']=='neural':
                destination=output/e['checkpoint']
                if not destination.exists():shutil.copy2(e['source_checkpoint'],destination)
                assert sha(destination)==e['checkpoint_sha256']
        atomic_json(output/'manifest.json',manifest)
    return manifest

def run_game(job):
    import torch
    from .model import SPECS
    from .rules import Rules
    from .planner import Planner,Config
    from .policy import HeuristicPolicy
    from .neural import Network
    from .guidance import deployment_agent
    from .notation import GameRecord
    from .scenarios import from_data,digest
    from .search_training import termination_reason
    torch.set_num_threads(1)
    started=perf_counter();manifest=job['manifest'];row=dict(job['game'])
    try:
        initial=from_data(manifest['maps'][row['map_index']]['state']);state=initial.clone()
        rules=Rules();config=Config(seconds=manifest['seconds'],nodes=manifest['nodes'])
        by_name={e['name']:e for e in manifest['entrants']};planners=[];effective={}
        for name in (row['blue'],row['red']):
            e=by_name[name]
            from .order_search import OrderPlanner
            from .bundle_mcts import MCTSPlanner,DEFAULTS
            backend=MCTSPlanner if e.get('search')=='mcts' else OrderPlanner if e.get('search','old')=='ordered' else Planner
            if e['type']=='heuristic':
                planner=backend(rules=rules,policy=HeuristicPolicy(state.allowed_builds or tuple(SPECS)),config=config)
                effective[name]=dict(type='heuristic')
            else:
                network=Network.load(Path(job['output'])/e['checkpoint'])
                if e.get('neural_value_weight') is not None:
                    if not network.search_settings:raise ValueError('Weight override requires a guided checkpoint')
                    network.search_settings=dict(network.search_settings,neural_value_weight=e['neural_value_weight'])
                if e.get('search')=='mcts':network.search_settings=dict(network.search_settings or {},mode='guided',planner='mcts',neural_value_weight=e.get('neural_value_weight',1.))
                agent=deployment_agent(network)
                planner=backend(rules=rules,policy=agent,value=agent,config=config)
                effective[name]=dict(type='neural',search_settings=network.search_settings)
            effective[name]['search']=e.get('search','old')
            if e.get('search')=='mcts':
                settings=effective[name].get('search_settings',{})
                effective[name]['mcts_parameters']={k:settings.get(k,v) for k,v in DEFAULTS.items()}
            planners.append(planner)
        record=GameRecord(initial);counts=[{},{}];seconds=[0.,0.];turns=[0,0]
        for _ in range(manifest['max_turns']):
            if rules.outcome(state) is not None:break
            side=state.player;planner=planners[side];t=perf_counter()
            analysis=planner.analyze(state)
            state=planner.execute(state,analysis)
            seconds[side]+=perf_counter()-t;turns[side]+=1
            for key,value in analysis.metrics.counts.items():
                old=counts[side].get(key,0)
                counts[side][key]=max(old,value) if key=='mcts_max_depth' else old+value
            for action in analysis.actions:record.append(action)
            if record.states[-1].key()!=state.key():raise ValueError('Replay differs from planner execution')
        row.update(map=manifest['maps'][row['map_index']]['name'],winner=rules.outcome(state),
                   termination=termination_reason(state,rules,initial.owners),played_turns=sum(turns),turns_by_side=turns,
                   final_income=[rules.income(state,p) for p in (0,1)],
                   final_state_sha256=digest(state),search_seconds=seconds,search_counts=counts,
                   effective_settings=effective,elapsed_seconds=round(perf_counter()-started,3))
        return row,record.dumps()
    except Exception as e:
        row.update(error=f'{type(e).__name__}: {e}',elapsed_seconds=round(perf_counter()-started,3))
        return row,None

def write_summary(output,manifest,games,elapsed):
    names=[e['name'] for e in manifest['entrants']]
    expected=len(schedule(manifest['entrants'],manifest['maps']))
    summary=dict(expected_games=expected,finished_games=len(games),complete=len(games)==expected and not any(g.get('error') for g in games),
                 updated_utc=datetime.now(timezone.utc).isoformat(),session_elapsed_seconds=round(elapsed,3),
                 **summarize(names,games),games=games)
    atomic_json(output/'summary.json',summary)
    lines=['# Tournament results','',f'{len(games)}/{expected} games recorded. Wins = 1 point; draws = 0.5. Unfinished games excluded from score rate.','',
           '| Entrant | W | L | D | Points | Score |','|---|---:|---:|---:|---:|---:|']
    for name,row in sorted(summary['standings'].items(),key=lambda item:-(item[1]['score_rate'] or 0)):
        rate='â€”' if row['score_rate'] is None else f'{100*row["score_rate"]:.1f}%'
        lines.append(f'| {name} | {row["wins"]} | {row["losses"]} | {row["draws"]} | {row["points"]:g} | {rate} |')
    lines+=['','## Pairings','']
    for pair,rows in summary['pairs'].items():
        a,b=rows;row=rows[a]
        lines.append(f'- {pair}: **{row["wins"]}â€“{row["losses"]}â€“{row["draws"]}** from {a}â€™s perspective; points {row["points"]:g}â€“{rows[b]["points"]:g}.')
    lines+=['','Each pairing uses every map with entrant sides swapped. Search has a wall-clock budget; CPU scheduling can change candidate coverage. These are paired games on the configured maps, not independent estimates of general playing strength.']
    (output/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary

def run(config,output,workers=4):
    if workers<1:raise ValueError('workers must be positive')
    output=Path(output).resolve();manifest=prepare(config,output);started=perf_counter()
    lock=output/'.running'
    # Prevent concurrent writers. Remove this lock manually only after verifying a crashed run is no longer active.
    with lock.open('x') as handle:
        import os
        handle.write(str(os.getpid()))
    try:
        (output/'games').mkdir(exist_ok=True);(output/'replays').mkdir(exist_ok=True)
        all_games=schedule(manifest['entrants'],manifest['maps']);completed={}
        for game in all_games:
            path=output/'games'/(game['id']+'.json')
            if path.exists():
                result=json.loads(path.read_text())
                if not result.get('error'):
                    if not (output/result['replay']).exists():raise ValueError('Completed game is missing replay')
                    completed[game['id']]=result
        write_summary(output,manifest,list(completed.values()),0)
        pending=[dict(manifest=manifest,game=g,output=str(output)) for g in all_games if g['id'] not in completed]
        print(json.dumps(dict(event='startup',total=len(all_games),pending=len(pending),workers=workers)),flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(run_game,job) for job in pending]
            for future in as_completed(futures):
                row,pgn=future.result()
                if pgn is not None:
                    row['replay']='replays/'+row['id']+'.awpgn'
                    (output/row['replay']).write_text(pgn+'\n',encoding='utf-8')
                atomic_json(output/'games'/(row['id']+'.json'),row);completed[row['id']]=row
                report=write_summary(output,manifest,sorted(completed.values(),key=lambda g:g['id']),perf_counter()-started)
                print(json.dumps(dict(event='game',finished=len(completed),total=len(all_games),
                                      id=row['id'],blue=row['blue'],red=row['red'],map=row.get('map'),
                                      winner=row.get('winner'),termination=row.get('termination'),
                                      error=row.get('error'),elapsed_seconds=row['elapsed_seconds'])),flush=True)
        return report if pending else write_summary(output,manifest,list(completed.values()),perf_counter()-started)
    finally:lock.unlink(missing_ok=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();result=run(args.config,args.output,args.workers)
    if not result['complete']:raise SystemExit('Tournament incomplete; inspect summary.json and rerun to retry failed games')

if __name__=='__main__':main()
