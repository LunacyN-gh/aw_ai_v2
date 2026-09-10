"""Policy-proposed complete turns with equal-coverage reply comparisons.

This backend is prior-regularized candidate search, not PUCT/MCTS.
"""
import hashlib
import math
import random
from time import perf_counter
import torch

from .budget import Budget
from .evaluation import HeuristicValue
from .model import Action, END
from .neural import KINDS
from .neural_training import unit_actions, build_choices
from .planner import Analysis, Candidate, Planner
from .policy import HeuristicPolicy


def policy_turn(neural, state, rules, rng, temperature=0., budget=None, until=None):
    """Same legal ordering/argmax as raw learned_turn, including purchases.

    The mandatory baseline is atomic (budget=None). Other proposals may abort.
    Returns None on timeout; aborted proposals never become training targets.
    """
    state=state.clone(); actions=[]; logs=[]
    def choose(site=None):
        if budget is not None and not budget.take('policy_steps',until): return None
        with torch.no_grad():
            encoded=neural.encoded(state,rules)
            choices=unit_actions(state,rules) if site is None else build_choices(state,rules,site,neural.network.allowed_builds)
            logits=neural.network.action_logits(encoded,choices) if site is None else neural.network.build_logits(encoded,state,rules,site)[choices]
            logp=logits.log_softmax(0)
            i=int(logits.argmax()) if not temperature else rng.choices(range(len(choices)),weights=(logits/temperature).softmax(0).tolist())[0]
            logs.append(float(logp[i]))
            return choices[i]
    while rules.outcome(state) is None and any(not u.acted for u in state.army(state.player)):
        a=choose()
        if a is None:return None
        rules.apply_inplace(state,a);actions.append(a)
    if rules.outcome(state) is None:
        sites=[p for p in state.board.properties if state.owners.get(p)==state.player and p not in state.occupancy and rules.roster(state.board.tiles[p],state)]
        for site in sites:
            k=choose(site)
            if k is None:return None
            if k<len(KINDS):
                a=Action('build',destination=site,build=KINDS[k]);rules.apply_inplace(state,a);actions.append(a)
        rules.apply_inplace(state,END);actions.append(END)
    c=Candidate(tuple(actions),state,0.,'greedy policy' if not temperature else 'sampled policy')
    # Mean log likelihood avoids favoring short bundles solely for their length.
    # This is a normalized policy preference, not a joint turn probability.
    c.policy_preference=sum(logs)/max(1,len(logs))
    return c


def preference(neural,start,candidate,rules):
    """Score a teacher turn under exactly the same autoregressive policy."""
    state=start.clone(); logs=[]
    def record(choice,site=None):
        with torch.no_grad():
            encoded=neural.encoded(state,rules)
            choices=unit_actions(state,rules) if site is None else build_choices(state,rules,site,neural.network.allowed_builds)
            logits=neural.network.action_logits(encoded,choices) if site is None else neural.network.build_logits(encoded,state,rules,site)[choices]
            logs.append(float(logits.log_softmax(0)[choices.index(choice)]))
    for a in candidate.actions:
        if a.kind not in ('end','build'):
            record(a);rules.apply_inplace(state,a)
    if rules.outcome(state) is None:
        builds={a.destination:KINDS.index(a.build) for a in candidate.actions if a.kind=='build'}
        sites=[p for p in state.board.properties if state.owners.get(p)==state.player and p not in state.occupancy and rules.roster(state.board.tiles[p],state)]
        for site in sites:
            k=builds.get(site,len(KINDS));record(k,site)
            if k<len(KINDS):rules.apply_inplace(state,Action('build',destination=site,build=KINDS[k]))
    return sum(logs)/max(1,len(logs))


def analyze(planner,state):
    state.validate()
    settings=planner.policy.network.search_settings
    neural=planner.policy.neural;rules=planner.rules;player=state.player
    budget=Budget(planner.config.seconds,planner.config.nodes)
    heuristic=HeuristicValue();weight=settings['neural_value_weight']
    scale=settings.get('heuristic_scale',20000.)
    rng=random.Random(settings.get('exploration_seed',0)+int.from_bytes(hashlib.sha256(repr(state.key()).encode()).digest()[:8],'little'))
    def values(states):
        if weight:neural.prefetch([s for s in states if rules.outcome(s) is None],rules)
        result=[]
        for s in states:
            budget.metrics.counts['value_evaluations']+=1
            winner=rules.outcome(s)
            result.append((1. if winner==player else -1.) if winner is not None else
                (1-weight)*math.tanh(heuristic.evaluate(s,player,rules)/scale)+weight*(neural.evaluate(s,player,rules)/20000 if weight else 0.))
        return result
    if rules.outcome(state) is not None:
        return Analysis((),values([state])[0],[],{},[],budget.finish(),state.key(),player)
    # Preserve the exact raw turn even if completing it uses the whole budget.
    raw=policy_turn(neural,state,rules,rng)
    budget.metrics.counts['raw_baseline_retained']=1
    candidates=[raw];seen={raw.state.key()}
    count=settings.get('bundle_candidates',6)
    context=planner.strategist.plan(state,rules,budget,remember=True)
    teacher=Planner(rules,HeuristicPolicy(neural.network.allowed_builds),heuristic,planner.config)
    until=budget.fraction(.4)
    if budget.available(until):
        c=teacher._finish(state,(),context,budget,min(until,perf_counter()+max(0,until-perf_counter())/3),reason='teacher baseline')
        if c.complete and c.production_complete and c.state.key() not in seen:
            c.policy_preference=preference(neural,state,c,rules)
            candidates.append(c);seen.add(c.state.key())
    for attempt in range(count*3):
        if len(candidates)>=count or not budget.available(until):break
        temperature=settings.get('proposal_temperature',.7)
        if rng.random()<settings.get('exploration_fraction',.25):temperature=max(1.2,temperature)
        c=policy_turn(neural,state,rules,rng,temperature,budget,until)
        if c is None:break
        if c.state.key() not in seen:candidates.append(c);seen.add(c.state.key())
    budget.metrics.counts['complete_bundles']=len(candidates)
    budget.metrics.counts['distinct_bundles']=len(candidates)
    root_values=values([c.state for c in candidates])
    qs=list(root_values);reply_actions=[() for _ in candidates];rounds=0
    # Commit a reply method only when it completes for EVERY live candidate.
    # Extra partial coverage can never penalize one candidate in isolation.
    for method in ('policy','teacher')[:planner.config.replies]:
        replies=[];success=True
        for c in candidates:
            if rules.outcome(c.state) is not None:replies.append(c);continue
            if not budget.available():success=False;break
            if method=='policy':
                reply=policy_turn(neural,c.state,rules,rng,budget=budget)
            else:
                ctx=teacher.strategist.plan(c.state,rules,budget)
                reply=teacher._finish(c.state,(),ctx,budget,budget.deadline)
            if reply is None or not reply.complete or not reply.production_complete:success=False;break
            replies.append(reply);budget.metrics.counts['reply_bundles']+=1
        if not success:
            budget.metrics.counts['discarded_reply_rounds']+=1
            break
        scores=values([r.state for r in replies])
        for i,(score,reply) in enumerate(zip(scores,replies)):
            if rounds==0 or score<qs[i]:qs[i]=score;reply_actions[i]=reply.actions
        rounds+=1
    budget.metrics.counts['common_reply_rounds']=rounds
    budget.metrics.counts['reply_checked_roots']=len(candidates) if rounds else 0
    strength=settings.get('policy_prior_weight',.1)
    diagnostics=[]
    for i,c in enumerate(candidates):
        c.score=qs[i]+strength*c.policy_preference
        c.reply=reply_actions[i];c.verified=bool(rounds)
        winner=rules.outcome(c.state)
        if winner is not None:c.score=1e9 if winner==player else -1e9;c.verified=True
        diagnostics.append(dict(source=c.reason,actions=repr(c.actions),raw=i==0,root_value=root_values[i],
            reply_value=qs[i],policy_preference=c.policy_preference,selection_score=c.score,reply_rounds=rounds))
    # With no common reply coverage, do not override a sound policy using a
    # shallower speculative value estimate. Exact terminal outcomes still count.
    pool=candidates if rounds else [raw]+[c for c in candidates[1:] if rules.outcome(c.state)==player]
    best=max(pool,key=lambda c:c.score)
    budget.metrics.counts['raw_baseline_selected']=int(best is raw)
    budget.metrics.counts['soft_targets']=0
    result=Analysis(best.actions,best.score,pool,context.objectives,context.regions,budget.finish(),state.key(),player)
    result.bundle_diagnostics=diagnostics
    return result
