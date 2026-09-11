"""Search-policy iteration with terminal value labels and fixed-opponent checks.

Policy targets are conditional distributions over beam candidates, NOT MCTS
visit counts. The legacy raw actor-critic experiment remains in neural_training.
"""
from collections import deque
from dataclasses import asdict, dataclass, replace
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import multiprocessing
import json
import math
from pathlib import Path
import random
from time import perf_counter
import torch

from .model import Action, END, INCOME_TILES
from .neural import Network, NetworkConfig, KINDS
from .neural_training import training_map, learned_turn
from .planner import Planner, Config
from .policy import HeuristicPolicy
from .rules import Rules
from .scenarios import load, digest
from .guidance import deployment_agent
from .teacher_replay import TeacherReplay, training_batch
from .promotion import promotion_decision, promotion_scores

ITA = ('infantry', 'tank', 'artillery')


def opponent_kind(episode, baseline=False):
    types = ('self','snapshot','teacher','reference')+ (('baseline',) if baseline else ())
    # Side-swapped pairs for every opponent, even when the cycle length is even.
    return types[(episode//2) % len(types)]


@dataclass
class Example:
    state: object
    site: int | None
    target: dict
    result: float | None = None
    source: str = 'search'


def distribution(pairs):
    result = {}
    for choice, weight in pairs:
        result[choice] = result.get(choice, 0.)+weight
    total = sum(result.values())
    return {key: weight/total for key, weight in result.items()} if total else {}


def candidate_weights(candidates, temperature=2000.):
    best = max(c.score for c in candidates)
    # Terminal scores are categorical, not billions of heuristic units.
    if abs(best) >= 1e8:
        raw = [1. if c.score == best else 0. for c in candidates]
    else:
        raw = [math.exp(max(-30., min(0., (c.score-best)/temperature))) for c in candidates]
    return [x/sum(raw) for x in raw]


def search_turn(planner, state, rng, explore=False):
    analysis = planner.analyze(state)
    candidates = [c for c in analysis.alternatives if c.complete and c.verified]
    candidates = candidates or [c for c in analysis.alternatives if c.complete] or analysis.alternatives
    if not candidates:
        return state, [], {'verified': 0, 'partial': 0}
    settings = getattr(getattr(getattr(planner,"policy",None),"network",None),"search_settings",None) or {}
    temperature = settings.get("target_temperature",.15) if settings.get("planner") == "bundle" else 2000.
    weights = candidate_weights(candidates,temperature)
    selected = next((c for c in candidates if c.actions == analysis.actions), candidates[0])
    if explore:
        selected = rng.choices(candidates, weights=[(1-settings.get("execution_exploration",.1))*w+settings.get("execution_exploration",.1)/len(weights) for w in weights])[0]
    # Search emits all unit actions before its production phase.
    sequences = [tuple(a for a in c.actions if a.kind not in ('end','build')) for c in candidates]
    chosen_units = tuple(a for a in selected.actions if a.kind not in ('end','build'))
    build_maps = [{a.destination: KINDS.index(a.build) for a in c.actions if a.kind == 'build'} for c in candidates]
    chosen_builds = {a.destination: KINDS.index(a.build) for a in selected.actions if a.kind == 'build'}
    examples = []
    rules = planner.rules
    for i, action in enumerate(chosen_units):
        pairs = [(seq[i], w) for seq, w in zip(sequences, weights)
                 if len(seq) > i and seq[:i] == chosen_units[:i]]
        target = {action: 1.} if getattr(planner,'expert_targets',False) else distribution(pairs) or {action: 1.}
        examples.append(Example(state.clone(), None, target))
        state = rules.apply(state, action)
    if rules.outcome(state) is None:
        if selected.complete and selected.production_complete:
            previous = {}
            sites = [p for p in state.board.properties if state.owners.get(p) == state.player
                     and p not in state.occupancy and rules.roster(state.board.tiles[p], state)]
            for site in sites:
                choice = chosen_builds.get(site, len(KINDS))
                pairs = [(builds.get(site, len(KINDS)), w)
                         for seq, builds, w, candidate in zip(sequences, build_maps, weights, candidates)
                         if (candidate.production_complete or site in builds)
                         and seq == chosen_units and all(builds.get(p, len(KINDS)) == v for p,v in previous.items())]
                target = {choice: 1.} if getattr(planner,'expert_targets',False) else distribution(pairs) or {choice: 1.}
                examples.append(Example(state.clone(), site, target))
                if choice < len(KINDS):
                    state = rules.apply(state, Action('build', destination=site, build=KINDS[choice]))
                previous[site] = choice
        else:
            # A timeout must not manufacture "save" targets at unvisited sites.
            for action in selected.actions:
                if action.kind == 'build':
                    examples.append(Example(state.clone(), action.destination, {KINDS.index(action.build): 1.}))
                    state = rules.apply(state, action)
        state = rules.apply(state, END)
    return state, examples, {'search_counts':dict(getattr(getattr(analysis,'metrics',None),'counts',{})),
                            'soft_targets':sum(len(e.target)>1 for e in examples),
                            'verified': sum(c.verified for c in candidates),
                            'partial': int(not selected.complete or not selected.production_complete)}


def label_result(examples, winner):
    return [replace(e, result=None if winner is None else (1. if e.state.player == winner else -1.)) for e in examples]


def fit(network, optimizer, replay, rules, rng, updates=4, batch_size=64, teacher_replay=None, teacher_fraction=.5, value_loss_weight=.5, entropy_weight=.01):
    """Balance unit and production examples; learn value only from true results."""
    if not replay and not teacher_replay:
        return None
    totals = dict(policy_loss=0., production_loss=0., value_mse=0., entropy=0., total_loss=0.,
                  teacher_ce=0., search_ce=0., policy_samples=0, production_samples=0, value_samples=0,
                  samples=0, teacher_samples=0, search_samples=0, correction_samples=0)
    network.train()
    for _ in range(updates):
        batch = training_batch(replay,teacher_replay,teacher_fraction,batch_size,rng)
        if not batch: return None
        optimizer.zero_grad()
        choices, logits, values = network.decision_batch(batch, rules)
        targets = []
        for example, options in zip(batch, choices):
            invalid = set(example.target)-set(options)
            if invalid:
                raise ValueError(f'search target violates training action mask: {invalid}')
            targets.append(torch.tensor([example.target.get(a, 0.) for a in options]))
        device = values.device
        padded = torch.nn.utils.rnn.pad_sequence(logits, batch_first=True, padding_value=float('-inf'))
        target = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True).to(device)
        valid = torch.arange(padded.shape[1],device=device)[None] < torch.tensor([len(c) for c in choices],device=device)[:,None]
        log_probs = padded.log_softmax(1).masked_fill(~valid, 0.)
        ce = -(target*log_probs).sum(1)
        entropy = -(log_probs.exp()*log_probs).sum(1)
        losses = ce-entropy_weight*entropy
        terminal = [i for i,e in enumerate(batch) if e.result is not None]
        if terminal:
            mse = (values[terminal]-torch.tensor([batch[i].result for i in terminal],device=device)).square()
            losses = losses.index_add(0,torch.tensor(terminal,device=device),value_loss_weight*mse*len(batch)/len(terminal))
        else:
            mse = torch.zeros((),device=device)
        # A single backward pass. No value term for censored batches also means
        # AdamW does not decay the value head on those batches.
        losses.mean().backward()
        policy = torch.tensor([e.site is None for e in batch],device=device)
        expert = torch.tensor([e.source != 'search' for e in batch],device=device)
        metrics = torch.stack([ce[policy].sum(),ce[~policy].sum(),mse.sum(),entropy.sum(),losses.sum(),
                               ce[expert].sum(),ce[~expert].sum()]).detach().tolist()
        for key, value in zip(('policy_loss','production_loss','value_mse','entropy','total_loss','teacher_ce','search_ce'),metrics):
            totals[key] += value
        totals['policy_samples'] += sum(e.site is None for e in batch)
        totals['production_samples'] += sum(e.site is not None for e in batch)
        totals['value_samples'] += len(terminal); totals['samples'] += len(batch)
        totals['teacher_samples'] += sum(e.source != 'search' for e in batch)
        totals['search_samples'] += sum(e.source == 'search' for e in batch)
        totals['correction_samples'] += sum(e.source == 'correction' for e in batch)
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.)
        optimizer.step()
    network.eval()
    for key, count in [('policy_loss','policy_samples'), ('production_loss','production_samples'),
                       ('value_mse','value_samples'), ('entropy','samples'), ('total_loss','samples'),
                       ('teacher_ce','teacher_samples'), ('search_ce','search_samples')]:
        totals[key] = totals[key]/totals[count] if totals[count] else None
    totals['updates'] = updates
    return totals


def model_payload(network):
    """Immutable CPU snapshot: workers never receive CUDA tensors or live parameters."""
    return {'config': asdict(network.config), 'allowed': tuple(network.allowed_builds),
            'search_settings':network.search_settings,
            'weights': {k: v.detach().cpu().clone() for k,v in network.state_dict().items()}}


def restore_model(payload):
    network = Network(NetworkConfig(**payload['config']))
    network.load_state_dict(payload['weights'])
    network.allowed_builds = payload['allowed']
    network.search_settings = payload.get('search_settings')
    return network.eval()


def worker_init():
    torch.set_num_threads(1)


def teacher_planner(rules, allowed, config):
    planner = Planner(rules=rules,policy=HeuristicPolicy(allowed),config=config)
    planner.expert_targets = True
    return planner


def play_job(job):
    """Spawn-safe searched episode, with independent RNG and CPU inference."""
    started = perf_counter()
    worker_init()
    torch.manual_seed(job['seed'])
    rng = random.Random(job['seed'])
    rules = Rules()
    state = job['state']
    boot, opponent, side = job['boot'], job['opponent'], job['side']
    planners = [teacher_planner(rules,job['allowed'],job['search']) for _ in range(2)]
    if not boot:
        current = deployment_agent(restore_model(job['current']))
        planners[side] = Planner(rules=rules, policy=current, value=current, config=job['search'])
        if opponent != 'teacher':
            other = current if opponent == 'self' else deployment_agent(restore_model(job['other']))
            planners[1-side] = Planner(rules=rules, policy=other, value=other, config=job['search'])
    examples = []; experts = []; verified = partial = played = 0
    search_counts = {}
    learner_turns = [0,0]
    corrections_seconds = 0.
    initial_owners = state.owners.copy()
    teacher = teacher_planner(rules,job['allowed'],job['search'])
    for _ in range(job['turns']):
        if rules.outcome(state) is not None:
            break
        player = state.player
        learner_move = not boot and (opponent == 'self' or player == side)
        if learner_move:
            every = job.get('teacher_every',2)
            if every and learner_turns[player] % every == 0:
                query_started = perf_counter()
                _, corrections, _ = search_turn(teacher,state.clone(),random.Random(job['seed']+played+2000000))
                # Counterfactual teacher continuations never inherit the actual
                # game's result. Only their policy targets are supervised.
                experts.extend(replace(e,source='correction',result=None) for e in corrections)
                corrections_seconds += perf_counter()-query_started
            learner_turns[player] += 1
        state, samples, metrics = search_turn(planners[player], state, rng,
                                              explore=not getattr(planners[player],'expert_targets',False))
        if boot:
            examples.extend(replace(e,source='bootstrap') for e in samples)
        elif opponent == 'teacher' and player != side:
            examples.extend(replace(e,source='teacher') for e in samples)
        elif learner_move:
            examples.extend(samples)
        played += 1
        verified += metrics['verified']; partial += metrics['partial']
        for key,count in metrics.get('search_counts',{}).items():
            search_counts[key] = search_counts.get(key,0)+count
        search_counts['soft_targets'] = search_counts.get('soft_targets',0)+metrics.get('soft_targets',0)
    winner = rules.outcome(state)
    labeled = label_result(examples,winner)
    diagnostics = dict(played_turns=played,termination=termination_reason(state,rules,initial_owners),
                       final_income=[rules.income(state,p) for p in (0,1)],
                       search_counts=search_counts,teacher_query_seconds=corrections_seconds,corrections=experts)
    return labeled, winner, verified, partial, perf_counter()-started, diagnostics


def termination_reason(state, rules, initial_owners):
    winner = rules.outcome(state)
    if winner is None: return 'turn_limit'
    if state.income_capture_limit and rules.income(state,winner)//1000 >= state.income_capture_limit:
        return 'income_capture_limit'
    if any(state.board.tiles[p]=='hq' and owner == 1-winner and state.owners.get(p)==winner
           for p,owner in initial_owners.items()):
        return 'hq_capture'
    return 'elimination'


def evaluation_job(job):
    worker_init()
    network = restore_model(job['network'])
    state, seed, player, mode = job['state'], job['seed'], job['player'], job['mode']
    rules = Rules()
    teacher = Planner(rules=rules, policy=HeuristicPolicy(network.allowed_builds), config=job['search'])
    if job.get('opponent'):
        other = deployment_agent(restore_model(job['opponent']))
        teacher = Planner(rules=rules,policy=other,value=other,config=job['search'])
    agent = deployment_agent(network)
    learner = Planner(rules=rules, policy=agent, value=agent, config=job['search'])
    partial = verified = played = 0
    initial_owners = state.owners.copy()
    rng = random.Random(seed)
    for _ in range(job['turns']):
        if rules.outcome(state) is not None:
            break
        if state.player == player and mode == 'policy':
            state, _ = learned_turn(network, state, rules, rng, record=False, greedy=True)
        else:
            planner = learner if state.player == player else teacher
            analysis = planner.analyze(state)
            if state.player == player:
                partial += sum(not c.complete for c in analysis.alternatives)
                verified += sum(c.verified for c in analysis.alternatives)
            state = planner.execute(state, analysis)
        played += 1
    winner = rules.outcome(state)
    return {'mode': mode, 'seed': seed, 'map_hash':digest(job['state']), 'learner_side': player, 'winner': winner,
            'result': 'censored' if winner is None else 'win' if winner == player else 'loss',
            'partial_candidates': partial, 'verified_candidates': verified,
            'played_turns':played, 'termination':termination_reason(state,rules,initial_owners)}


def evaluate(network, initial, config, games=4):
    """Fixed seeds and teacher, alternating sides; raw policy and deployed beam."""
    payload = model_payload(network)
    jobs = [dict(network=payload, state=initial(1000000+game//2), seed=1000000+game//2,
                 player=game%2, mode=mode, turns=config['turns'], search=config['search'])
            for mode in ('policy','beam') for game in range(games)]
    modes = ['policy','beam']
    if config.get('reference'):
        modes.extend(('gate','incumbent'))
        jobs.extend(dict(network=config['reference'],state=initial(1000000+game//2),seed=1000000+game//2,
                         player=game%2,mode='incumbent',turns=config['turns'],search=config['search'])
                    for game in range(games))
        jobs.extend(dict(network=payload,opponent=config['reference'],state=initial(1000000+game//2),
                         seed=1000000+game//2,player=game%2,mode='gate',turns=config['turns'],search=config['search'])
                    for game in range(games))
    executor = config.get('executor')
    rows = list(executor.map(evaluation_job, jobs) if executor else map(evaluation_job, jobs))
    summary = {mode: {result: sum(r['mode'] == mode and r['result'] == result for r in rows)
                      for result in ('win','loss','censored')} for mode in modes}
    from .opening_suite import opening_checks
    cpu_network = restore_model(payload)
    return {'summary': summary, 'games': rows, 'openings':opening_checks(cpu_network)}


def run(*args, workers=1, device='auto', **kwargs):
    """Own worker lifetime, including cleanup on training/evaluation failure."""
    if workers < 1:
        raise ValueError('workers must be positive')
    if device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    selected = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    if selected == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; install CUDA PyTorch or use --device cpu')
    pool = (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                initializer=worker_init) if workers > 1 else nullcontext(None))
    with pool as executor:
        return _run(*args, workers=workers, device=selected, executor=executor, **kwargs)


def _run(output, games=4, turns=60, seed=0, size=8, units=4, bootstrap=2, checkpoint=None,
        tiny=False, capture_fraction=2/3, map_path=None, capture_limit=None, roster=None,
        opponent_checkpoint=None, train_seconds=.5, train_nodes=4000, eval_seconds=1.,
        eval_every=25, eval_games=4, checkpoint_every=10, updates_per_game=4, batch_size=64,
        replay_size=8192, workers=1, device='cpu', executor=None, teacher_fraction=.5,
        teacher_every=2, teacher_replay_size=8192, neural_value_weight=.5, promotion_win_rate=None,
        search_mode="bundle", bundle_candidates=6, target_temperature=.15,
        exploration_fraction=.25, execution_exploration=.1, heuristic_scale=20000.,
        value_loss_weight=.5, entropy_weight=.01, bundle_version=2,
        proposal_temperature=.7, policy_prior_weight=.1, eval_maps=None, train_maps=None, promotion_margin=.05, promotion_teacher_weight=.5):
    if (min(games,turns,train_nodes,eval_every,eval_games,checkpoint_every,updates_per_game,batch_size,replay_size) < 1
            or batch_size < 2 or eval_games % 2 or bootstrap < 0 or seed < 0 or seed+games+bootstrap >= 1000000
            or (map_path is None and not train_maps and (not 8 <= size <= 64 or units < 1))):
        raise ValueError('invalid run parameters; evaluation games must be positive and even')
    if (not 0 <= teacher_fraction <= 1 or teacher_every < 0 or teacher_replay_size < 2
            or not 0 <= neural_value_weight <= 1 or (promotion_win_rate is not None and not .5 < promotion_win_rate <= 1)):
        raise ValueError('invalid teacher replay, value blend, or promotion settings')
    if (search_mode not in ("bundle","beam") or bundle_candidates < 2
            or not all(math.isfinite(x) for x in (target_temperature,heuristic_scale,value_loss_weight,entropy_weight,exploration_fraction,execution_exploration))
            or min(target_temperature,heuristic_scale) <= 0 or min(value_loss_weight,entropy_weight)<0
            or not 0 <= exploration_fraction <= 1 or not 0 <= execution_exploration <= 1):
        raise ValueError("invalid bundle search or loss settings")
    if (bundle_version not in (1,2) or not math.isfinite(proposal_temperature) or proposal_temperature<=0
            or not math.isfinite(policy_prior_weight) or policy_prior_weight<0):
        raise ValueError("invalid policy bundle settings")
    if promotion_win_rate is not None:promotion_margin=promotion_win_rate-.5
    if not math.isfinite(promotion_margin) or not 0<=promotion_margin<=1 or not math.isfinite(promotion_teacher_weight) or not 0<=promotion_teacher_weight<=1:
        raise ValueError("invalid promotion margin or teacher weight")
    run_started = perf_counter()
    torch.set_num_threads(1); torch.manual_seed(seed)
    rng = random.Random(seed)
    network = Network.load(checkpoint) if checkpoint else Network(NetworkConfig(16,1,32,1,4) if tiny else None)
    if roster is not None:
        if roster not in ('all','ita'):
            raise ValueError('roster must be all or ita')
        network.allowed_builds = ITA if roster == 'ita' else KINDS
    network.search_settings = dict(mode='guided',neural_value_weight=neural_value_weight,planner=search_mode,
        bundle_candidates=bundle_candidates,target_temperature=target_temperature,exploration_fraction=exploration_fraction,
        execution_exploration=execution_exploration,heuristic_scale=heuristic_scale,bundle_version=bundle_version,
        proposal_temperature=proposal_temperature,policy_prior_weight=policy_prior_weight)
    network.to(device)
    allowed = tuple(network.allowed_builds)
    if map_path and train_maps:
        raise ValueError("use either --map or repeated --train-map, not both")
    training_paths=list(train_maps or ([map_path] if map_path else []))
    fixed = load(map_path) if map_path else None
    def configure(state):
        if capture_limit is not None:
            state.income_capture_limit = capture_limit
        elif capture_fraction:
            if not .5 < capture_fraction <= 1:
                raise ValueError('capture fraction must be in (0.5, 1], or zero to disable')
            state.income_capture_limit = math.ceil(sum(t in INCOME_TILES for t in state.board.tiles)*capture_fraction)
        else:
            state.income_capture_limit = None
        state.validate()
        if any(u.kind not in allowed for u in state.units.values()):
            raise ValueError('initial map contains units excluded by the training roster')
        if Rules().outcome(state) is not None:
            raise ValueError('training position is already terminal')
        return state
    training_states=[configure(load(p)) for p in training_paths]
    fixed=training_states[0] if len(training_states)==1 else None
    sizes=sorted({s.board.width for s in training_states}) if training_states else sorted(set((8,max(8,size//2),size)))
    def initial(map_seed):
        if training_states:
            # Whole opponent cycles avoid binding one map to one opponent or side.
            cycle=2*(5 if opponent_checkpoint else 4)
            return training_states[(map_seed//cycle)%len(training_states)].clone()
        width = random.Random(map_seed).choice(sizes)
        return configure(training_map(map_seed,width,units,capture_fraction,allowed))
    evaluation_paths=list(eval_maps or training_paths)
    evaluation_states=[configure(load(p)) for p in evaluation_paths]
    if evaluation_states and eval_games % (2*len(evaluation_states)):
        raise ValueError('eval_games must include equally many side-swapped pairs per evaluation map')
    def evaluation_initial(map_seed):
        return evaluation_states[(map_seed-1000000)%len(evaluation_states)].clone() if evaluation_states else initial(map_seed)
    training_config = Config(seconds=train_seconds,nodes=train_nodes)
    evaluation_config = {'turns':turns,'search':Config(seconds=eval_seconds,nodes=train_nodes), 'executor':executor}
    rules = Rules()
    optimizer = torch.optim.AdamW(network.parameters(),lr=3e-4,weight_decay=1e-4)
    replay = deque(maxlen=replay_size)
    teacher_replay = TeacherReplay(teacher_replay_size,seed)
    snapshots = []
    reference = None
    reference_evaluation = None
    fixed_opponent = Network.load(opponent_checkpoint) if opponent_checkpoint else None
    if fixed_opponent and not set(fixed_opponent.allowed_builds).issubset(allowed):
        raise ValueError('baseline opponent can build units excluded by this training roster')
    fixed_opponent = model_payload(fixed_opponent) if fixed_opponent else None
    output = Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    report = {'training':'value-guided bundle distillation + terminal value regression' if search_mode=='bundle' else 'teacher-guided beam distillation + terminal value regression',
              'seed':seed,'map_path':str(Path(map_path).resolve()) if map_path else None,
              'initial_state_hash':digest(fixed) if fixed else None, 'training_sizes':sizes,
              'allowed_builds':list(allowed),'income_capture_fraction':capture_fraction,
              'income_capture_limit':fixed.income_capture_limit if fixed else capture_limit,
              'turn_limit':turns,'runs':[],'evaluations':[], 'validation_scope':
              'explicit evaluation maps, alternating learner sides' if evaluation_states else 'same fixed map, alternating learner sides' if fixed else 'fixed held-out synthetic seeds',
              'opponent_checkpoint':str(opponent_checkpoint) if opponent_checkpoint else None,
              'tactical_families_used_for_training':False,
              'evaluation_maps':[str(Path(p).resolve()) for p in evaluation_paths],
              'training_maps':[str(Path(p).resolve()) for p in training_paths],
              'training_map_hashes':[digest(s) for s in training_states],
              'parameters':sum(p.numel() for p in network.parameters()),
              'config':dict(train_seconds=train_seconds,train_nodes=train_nodes,eval_seconds=eval_seconds,
                            eval_every=eval_every,eval_games=eval_games,checkpoint_every=checkpoint_every,
                            updates_per_game=updates_per_game,batch_size=batch_size,replay_size=replay_size,
                            workers=workers,device=device,worker_device='cpu',worker_threads=1,
                            policy_lag_games_max=workers-1,teacher_fraction=teacher_fraction,teacher_every=teacher_every,
                            teacher_replay_size=teacher_replay_size,neural_value_weight=neural_value_weight,
                            promotion_margin=promotion_margin,promotion_teacher_weight=promotion_teacher_weight,
                            promotion_censored_score=.5,search_settings=network.search_settings,
                            value_loss_weight=value_loss_weight,entropy_weight=entropy_weight),
              'promotions':[], 'complete':False}
    def save(path=output, model=None):
        temporary = path.with_suffix(path.suffix+'.tmp')
        (model if model is not None else network).save(temporary, report)
        temporary.replace(path)
        summary = path.with_suffix('.json')
        temporary = summary.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        temporary.replace(summary)
    def check(game):
        nonlocal reference, reference_evaluation, snapshots
        evaluation_started = perf_counter()
        evaluation_config['reference'] = snapshots[-1] if reference is not None else None
        result = evaluate(network, evaluation_initial, evaluation_config, eval_games)
        result['elapsed_seconds'] = round(perf_counter()-evaluation_started, 3)
        result['run_elapsed_seconds'] = round(perf_counter()-run_started, 3)
        result['after_game'] = game
        report['evaluations'].append(result)
        print(json.dumps({'phase':'evaluation','after_game':game,'elapsed_seconds':result['elapsed_seconds'],
                          'run_elapsed_seconds':result['run_elapsed_seconds'],
                          'openings':result.get('openings'),**result['summary']}),flush=True)
        if reference is None:
            reference = model_payload(network)
            snapshots = [reference]
            reference_evaluation = result
            report['reference_game'] = game
            report['best_measured_game'] = game
            report['promotions'].append({'after_game':game,'accepted':True,'reason':'initialized frozen reference'})
            save(output.with_name(output.stem+'.reference.pt'))
            save(output.with_name(output.stem+'.best.pt'))
        else:
            promotion_started = perf_counter()
            accepted,reason = promotion_decision(result,margin=promotion_margin,teacher_weight=promotion_teacher_weight)
            details = promotion_scores(result,promotion_teacher_weight) if all(k in result["summary"] for k in ("beam","incumbent","gate")) else {}
            details["margin"] = promotion_margin
            report['promotions'].append({'after_game':game,'accepted':accepted,'reason':reason,**details})
            print(json.dumps({'phase':'promotion','after_game':game,'accepted':accepted,'reason':reason,**details,
                              'elapsed_seconds':round(perf_counter()-promotion_started,3),
                              'run_elapsed_seconds':round(perf_counter()-run_started,3)}),flush=True)
            if accepted:
                snapshots = (snapshots+[model_payload(network)])[-4:]
                reference_evaluation = result
                report['best_measured_game'] = game
                save(output.with_name(output.stem+'.best.pt'))
        save()
    print(json.dumps({'phase':'startup','device':device,'workers':workers,'worker_device':'cpu',
                      'gpu':torch.cuda.get_device_name(0) if device == 'cuda' else None,
                      'search_mode':search_mode,'neural_value_weight':neural_value_weight,'teacher_fraction':teacher_fraction,
                      'batch_size':batch_size,'elapsed_seconds':round(perf_counter()-run_started,3)}),flush=True)
    game = 0
    if bootstrap == 0:
        check(0)
    while game < bootstrap+games:
        round_started = perf_counter()
        boot = game < bootstrap
        # Rounds cannot cross evaluation, snapshot, or bootstrap boundaries.
        stop = min(game+workers, bootstrap if boot else bootstrap+games)
        for boundary in range(game+1, stop+1):
            if (boundary == bootstrap or boundary % checkpoint_every == 0 or
                    (boundary > bootstrap and (boundary-bootstrap) % eval_every == 0)):
                stop = boundary
                break
        current = None if boot else model_payload(network)
        jobs = []
        for index in range(game, stop):
            episode = index-bootstrap
            opponent = 'teacher' if boot else opponent_kind(episode,fixed_opponent is not None)
            other = random.Random(seed+index).choice(snapshots) if opponent == 'snapshot' else fixed_opponent if opponent == 'baseline' else reference if opponent == 'reference' else None
            jobs.append(dict(seed=seed+index, state=initial(seed+index), boot=boot,
                             opponent=opponent, side=episode%2, allowed=allowed, search=training_config,
                             turns=turns, current=current, other=other,teacher_every=teacher_every))
        # Ingest in seed order. All games in a round use the same frozen learner.
        # Training can overlap later workers while keeping policy lag bounded.
        results = executor.map(play_job, jobs) if executor else map(play_job, jobs)
        version = game
        for job, result in zip(jobs, results):
            labeled, winner, verified, partial, play_seconds, diagnostics = result
            corrections = diagnostics.pop('corrections')
            experts = [e for e in labeled if e.source != 'search']+corrections
            teacher_replay.extend(experts)
            replay.extend(e for e in labeled if e.source == 'search')
            training_started = perf_counter()
            losses = fit(network,optimizer,replay,rules,rng,updates_per_game,batch_size,teacher_replay,teacher_fraction,value_loss_weight,entropy_weight)
            training_seconds = perf_counter()-training_started
            episode = game-bootstrap
            row = {'phase':'bootstrap' if boot else 'self_play','game':game+1,'seed':seed+game,
                   'training_map_hash':digest(job['state']),
                   'learner_side':None if boot else job['side'],'opponent':job['opponent'],'winner':winner,
                   'censored':winner is None,'decisions':len(labeled),'production_decisions':sum(e.site is not None for e in labeled),
                   'new_value_labels':sum(e.result is not None for e in labeled),
                   'teacher_decisions':len(experts),'correction_decisions':len(corrections),
                   'teacher_replay':len(teacher_replay),'bootstrap_anchors':len(teacher_replay.anchors),
                   'search_replay':len(replay),**diagnostics,
                   'verified_candidates':verified,'partial_turns':partial,'losses':losses,
                   'elapsed_seconds':round(play_seconds+training_seconds,3),
                   'play_seconds':round(play_seconds,3),'training_seconds':round(training_seconds,3),
                   'policy_after_game':version,'policy_lag_games':game-version,
                   'run_elapsed_seconds':round(perf_counter()-run_started,3)}
            report['runs'].append(row)
            print(json.dumps(row),flush=True)
            game += 1
            if game % checkpoint_every == 0 or game == bootstrap:
                save()
        print(json.dumps({'phase':'round','after_game':game,'games':len(jobs),
                          'elapsed_seconds':round(perf_counter()-round_started,3),
                          'run_elapsed_seconds':round(perf_counter()-run_started,3)}),flush=True)
        if game == bootstrap or (not boot and (game-bootstrap) % eval_every == 0):
            check(game)
    if not report['evaluations'] or report['evaluations'][-1]['after_game'] != bootstrap+games:
        check(bootstrap+games)
    report['complete'] = True
    report['total_elapsed_seconds'] = round(perf_counter()-run_started, 3)
    save()
    print(f'Saved {output}; total time {perf_counter()-run_started:.3f}s; compare policy and beam evaluation results before adopting this model.',flush=True)
    return report
