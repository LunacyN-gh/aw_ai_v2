"""Shared rollout utilities and the legacy raw actor-critic experiment.

The public run() now forwards to search_training, which implements the default
search-distillation trainer. run_legacy() retains the original experiment.
"""
import copy
import json
import random
from pathlib import Path
import torch
from .model import Action, END, SPECS, INCOME_TILES
from .neural import Network, NetworkConfig, KINDS
from .planner import Planner, Config
from .rules import Rules
from .scenarios import league_scale, load, digest


def training_map(seed, size=8, units=4, capture_fraction=2/3, roster=None):
    # Disjoint seed ranges are enforced by the runner. No evaluation puzzles.
    state=league_scale(size,units,seed,contact=True)
    if capture_fraction:
        import math
        if not .5<capture_fraction<=1:
            raise ValueError('capture fraction must be greater than one half and at most one, or zero to disable')
        total=sum(t in INCOME_TILES for t in state.board.tiles)
        state.income_capture_limit=math.ceil(total*capture_fraction)
    rng=random.Random(seed)
    from dataclasses import replace
    roster=roster or ('infantry','tank','artillery','anti_air','b_copter','recon')
    for uid,u in list(state.units.items()):
        kind=roster[rng.randrange(len(roster))]
        # The shared generator places units on ground traversable by this roster.
        state.units[uid]=replace(u,kind=kind)
    state.validate()
    return state


def unit_actions(state,rules):
    return [a for a in rules.legal_actions(state) if a.kind not in ('build','end')]


def build_choices(state,rules,site,allowed=None):
    return [i for i,k in enumerate(KINDS) if k in rules.roster(state.board.tiles[site], state)
            and (allowed is None or k in allowed)
            and SPECS[k].cost<=state.funds[state.player]]+[len(KINDS)]


def decision_logits(network,state,rules,site=None):
    encoded=network.encode(state,rules)
    if site is None:
        choices=unit_actions(state,rules)
        logits=network.action_logits(encoded,choices)
    else:
        choices=build_choices(state,rules,site,network.allowed_builds)
        logits=network.build_logits(encoded,state,rules,site)[choices]
    return choices,logits,network.value(encoded[0]).squeeze()


def learned_turn(network,state,rules,rng,record=True,greedy=False):
    samples=[]
    while rules.outcome(state) is None and any(not u.acted for u in state.army(state.player)):
        with torch.no_grad():
            choices,logits,_=decision_logits(network,state,rules)
            # Explicit uniform exploration guarantees every legal action support.
            probs=.95*logits.softmax(0)+.05/len(choices)
            index=int(logits.argmax()) if greedy else rng.choices(range(len(choices)),weights=probs.tolist())[0]
        if record: samples.append((state.clone(),None,choices[index]))
        state=rules.apply(state,choices[index])
    if rules.outcome(state) is None:
        sites=[p for p in state.board.properties if state.owners.get(p)==state.player
               and rules.roster(state.board.tiles[p], state) and p not in state.occupancy]
        for site in sites:
            with torch.no_grad():
                choices,logits,_=decision_logits(network,state,rules,site)
                probs=.95*logits.softmax(0)+.05/len(choices)
                index=int(logits.argmax()) if greedy else rng.choices(range(len(choices)),weights=probs.tolist())[0]
            selected=choices[index]
            if record: samples.append((state.clone(),site,selected))
            if selected<len(KINDS): state=rules.apply(state,Action('build',destination=site,build=KINDS[selected]))
        state=rules.apply(state,END)
    return state,samples


def teacher_turn(planner,state):
    analysis=planner.analyze(state)
    samples=[]; visited=set()
    for action in analysis.actions:
        if action.kind not in ('end','build'):
            samples.append((state.clone(),None,action))
        elif action.kind=='build':
            samples.append((state.clone(),action.destination,KINDS.index(action.build)))
            visited.add(action.destination)
        elif action.kind=='end':
            for p in state.board.properties:
                if p not in visited and p not in state.occupancy and state.owners.get(p)==state.player and planner.rules.roster(state.board.tiles[p], state):
                    samples.append((state.clone(),p,len(KINDS)))
        state=planner.rules.apply(state,action)
    return state,samples


def update(network,optimizer,examples,rules,rng,mode,winner=None,max_samples=64):
    if not examples: return None
    chosen=rng.sample(examples,min(max_samples,len(examples)))
    optimizer.zero_grad(); total=0.
    network.train()
    for state,site,selected in chosen:
        choices,logits,value=decision_logits(network,state,rules,site)
        index=choices.index(selected)
        probs=.95*logits.softmax(0)+.05/len(choices)
        if mode=='bootstrap':
            loss=-logits.log_softmax(0)[index]
        else:
            target=torch.tensor(1. if winner==state.player else -1.)
            advantage=target-value.detach()
            loss=-probs[index].log()*advantage+.5*(value-target).square()
            loss+=.01*(probs*probs.log()).sum()
        (loss/len(chosen)).backward(); total+=float(loss.detach())
    torch.nn.utils.clip_grad_norm_(network.parameters(),1.)
    optimizer.step(); network.eval()
    return total/len(chosen)


def run_legacy(output,games=4,turns=60,seed=0,size=8,units=4,bootstrap=2,checkpoint=None,tiny=False,capture_fraction=2/3,
        map_path=None,capture_limit=None):
    if min(games,turns)<1 or (map_path is None and (units<1 or not 8<=size<=64)) or bootstrap<0 or seed<0 or seed+games+bootstrap>=1000000:
        raise ValueError('positive run parameters required; training seeds must be below 1000000')
    torch.set_num_threads(1); torch.manual_seed(seed); rng=random.Random(seed)
    network=Network.load(checkpoint) if checkpoint else Network(NetworkConfig(16,1,32,1,4) if tiny else None)
    optimizer=torch.optim.AdamW(network.parameters(),lr=3e-4,weight_decay=1e-4)
    rules=Rules(); teacher=Planner(rules=rules,config=Config(seconds=.15,nodes=1000))
    logs=[]
    fixed=load(map_path) if map_path is not None else None
    if fixed is not None:
        if capture_limit is not None:
            fixed.income_capture_limit=capture_limit
        elif capture_fraction:
            import math
            if not .5<capture_fraction<=1:
                raise ValueError('capture fraction must be greater than one half and at most one, or zero to disable')
            fixed.income_capture_limit=math.ceil(sum(t in INCOME_TILES for t in fixed.board.tiles)*capture_fraction)
        else:
            fixed.income_capture_limit=None
        fixed.validate()
        if rules.outcome(fixed) is not None:
            raise ValueError('training map is already terminal with this capture limit')
    sizes=[fixed.board.width] if fixed is not None else sorted(set((8,max(8,size//2),size)))
    def initial(map_seed):
        if fixed is not None:
            return fixed.clone()
        width=random.Random(map_seed).choice(sizes)
        state=training_map(map_seed,width,units,capture_fraction)
        if capture_limit is not None:
            state.income_capture_limit=capture_limit
            state.validate()
        return state
    for game in range(bootstrap):
        initial_seed=seed+game
        state=initial(initial_seed); examples=[]
        for _ in range(turns):
            state,new=teacher_turn(teacher,state); examples.extend(new)
            if rules.outcome(state) is not None: break
        loss=update(network,optimizer,examples,rules,rng,'bootstrap')
        logs.append({'phase':'bootstrap','seed':initial_seed,'board_size':state.board.width,'decisions':len(examples),'loss':loss})
        print(json.dumps(logs[-1]),flush=True)
    snapshots=[copy.deepcopy(network).eval()]
    for game in range(games):
        initial_seed=seed+bootstrap+game
        state=initial(initial_seed); examples=[]; learner=game%2
        opponent=('self','snapshot','teacher')[game%3]
        frozen=rng.choice(snapshots)
        for _ in range(turns):
            if state.player==learner or opponent=='self':
                state,new=learned_turn(network,state,rules,rng); examples.extend(new)
            elif opponent=='snapshot': state,_=learned_turn(frozen,state,rules,rng,record=False)
            else: state,_=teacher_turn(teacher,state)
            if rules.outcome(state) is not None: break
        winner=rules.outcome(state)
        loss=update(network,optimizer,examples,rules,rng,'reinforce',winner) if winner is not None else None
        logs.append({'phase':'self_play','seed':initial_seed,'board_size':state.board.width,'opponent':opponent,'winner':winner,
                     'censored':winner is None,'decisions':len(examples),'loss':loss})
        print(json.dumps(logs[-1]),flush=True)
        if (game+1)%4==0: snapshots=(snapshots+[copy.deepcopy(network).eval()])[-4:]
    # Fixed-map runs evaluate the same position on both sides, not held-out maps.
    validation=[]
    for i in range(2):
        state=initial(1000000+seed+i)
        for _ in range(turns):
            if state.player==i: state,_=learned_turn(network,state,rules,rng,record=False,greedy=True)
            else: state,_=teacher_turn(teacher,state)
            if rules.outcome(state) is not None: break
        validation.append({'seed':1000000+seed+i,'board_size':state.board.width,'learner_side':i,'winner':rules.outcome(state),
                           'censored':rules.outcome(state) is None})
    report={'seed':seed,'size':size,'training_sizes':sizes,'units':units,'turn_limit':turns,'runs':logs,'validation':validation,
            'income_capture_fraction':capture_fraction,
            'income_capture_limit':fixed.income_capture_limit if fixed is not None else capture_limit,
            'map_path':str(Path(map_path).resolve()) if map_path is not None else None,
            'initial_state_hash':digest(fixed) if fixed is not None else None,
            'validation_scope':'same fixed map, alternating learner sides' if fixed is not None else 'held-out synthetic map seeds',
            'training':'search-policy bootstrap; terminal-only on-policy actor-critic',
            'tactical_families_used_for_training':False,
            'parameters':sum(p.numel() for p in network.parameters())}
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    network.save(output,report)
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def run(*args, **kwargs):
    """Default trainer: search distillation and terminal value supervision."""
    from .search_training import run as search_run
    return search_run(*args, **kwargs)
