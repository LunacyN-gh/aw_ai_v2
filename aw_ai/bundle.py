"""Budgeted whole-turn policy improvement. Scores are bounded outcome values.

This is candidate evaluation, not MCTS: targets are score softmaxes, not visits.
"""
import hashlib
import math
import random
from time import perf_counter

from .budget import Budget
from .evaluation import HeuristicValue
from .planner import Analysis, Planner
from .policy import HeuristicPolicy
from .tree import SamplePolicy


class ExploratoryPolicy(SamplePolicy):
    def __init__(self, base, seed):
        super().__init__(base, seed)
        # SamplePolicy binds build_scores on the instance; replace it too.
        self.build_scores = self.sample_builds

    def sample_builds(self, state, rules, site):
        scores = self.base.build_scores(state, rules, site)
        return {k: v-2000*math.log(-math.log(max(1e-12,self.random.random())))
                if math.isfinite(v) else v for k,v in scores.items()}


def analyze(planner, state):
    if planner.policy.network.search_settings.get('bundle_version',1)==2:
        from .policy_bundle import analyze as policy_analyze
        return policy_analyze(planner,state)
    state.validate()
    settings = planner.policy.network.search_settings
    budget = Budget(planner.config.seconds, planner.config.nodes)
    player = state.player
    heuristic = HeuristicValue()
    weight = settings['neural_value_weight']
    scale = settings.get('heuristic_scale', 20000.)
    seed = settings.get('exploration_seed', 0)
    seed += int.from_bytes(hashlib.sha256(repr(state.key()).encode()).digest()[:8], 'little')
    neural = planner.policy.neural

    def value(position):
        budget.metrics.counts['value_evaluations'] += 1
        winner = planner.rules.outcome(position)
        if winner is not None:
            return 1. if winner == player else -1.
        h = math.tanh(heuristic.evaluate(position,player,planner.rules)/scale)
        n = neural.evaluate(position,player,planner.rules)/20000 if weight else 0.
        return (1-weight)*h+weight*n

    context = planner.strategist.plan(state,planner.rules,budget,remember=True)
    if planner.rules.outcome(state) is not None:
        return Analysis((),value(state),[],context.objectives,context.regions,budget.finish(),state.key(),player)
    teacher = HeuristicPolicy(planner.policy.network.allowed_builds)
    count = settings.get('bundle_candidates',16)
    fraction = settings.get('exploration_fraction',.25)
    exploratory = max(1,math.ceil((count-2)*fraction)) if fraction else 0
    policies = [teacher,neural]
    policies += [ExploratoryPolicy(neural if i%2 else teacher,seed+i)
                 for i in range(exploratory)]
    policies += [ExploratoryPolicy(planner.policy,seed+100+i)
                 for i in range(max(0,count-len(policies)))]
    candidates = []; seen = set(); fallback = None
    generation_deadline = budget.fraction(.55)
    for i,policy in enumerate(policies[:count]):
        if i and not budget.available(generation_deadline): break
        generator = Planner(planner.rules,policy,heuristic,planner.config)
        # A bounded slice prevents one sampled plan consuming all generation.
        until = min(generation_deadline,perf_counter()+max(0,generation_deadline-perf_counter())/max(1,min(4,count-i)))
        proposal = generator._finish(state,(),context,budget,until,reason='bundle '+('teacher' if i==0 else 'sampled' if i>1 else 'neural'))
        fallback = fallback or proposal
        for candidate in [proposal,*proposal.variants]:
            key = candidate.state.key()
            if candidate.complete and candidate.production_complete and key not in seen and len(candidates)<count:
                seen.add(key); candidates.append(candidate)
    budget.metrics.counts['complete_bundles'] = len(candidates)
    if not candidates:
        candidates = [fallback]
    # Score every root before reply work; do not compare evaluated and stale scores.
    if weight and hasattr(neural,'prefetch'):
        neural.prefetch([c.state for c in candidates if planner.rules.outcome(c.state) is None],planner.rules)
        budget.metrics.counts['root_value_batches'] += 1
    for candidate in candidates:
        candidate.score = value(candidate.state)
    for i,candidate in enumerate(candidates):
        if planner.rules.outcome(candidate.state) is not None:
            candidate.score = 1e9 if candidate.score>0 else -1e9
            candidate.verified = True
            continue
        if not budget.available(): break
        until = perf_counter()+max(0,budget.deadline-perf_counter())/max(1,len(candidates)-i)
        replies = []
        for policy in (teacher,neural)[:planner.config.replies]:
            if not budget.available(until): break
            generator = Planner(planner.rules,policy,heuristic,planner.config)
            ctx = generator.strategist.plan(candidate.state,planner.rules,budget)
            reply = generator._finish(candidate.state,(),ctx,budget,until)
            if reply.complete and reply.production_complete:
                replies.append((value(reply.state),reply.actions))
                budget.metrics.counts['reply_bundles'] += 1
        if replies:
            candidate.score,candidate.reply = min(replies,key=lambda r:r[0])
            candidate.verified = True
    # Prefer reply-checked candidates when any are available, as in beam training.
    pool = [c for c in candidates if c.verified] or candidates
    budget.metrics.counts['reply_checked_roots'] = sum(c.verified for c in candidates)
    best = max(pool,key=lambda c:c.score)
    budget.metrics.counts['distinct_bundles'] = len(seen)
    return Analysis(best.actions,best.score,candidates,context.objectives,context.regions,
                    budget.finish(),state.key(),player)
