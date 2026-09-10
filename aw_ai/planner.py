from __future__ import annotations

from dataclasses import dataclass, field
import math
from time import perf_counter

from .budget import Budget, Metrics
from .evaluation import HeuristicValue, objective_score
from .model import END, Action, State
from .policy import HeuristicPolicy, Threats, action_key, production_plans
from .rules import Rules
from .strategy import Strategist


@dataclass(frozen=True)
class Config:
    seconds: float = 1.0
    nodes: int = 4000
    beam: int = 4
    local_depth: int = 3
    actors: int = 6
    rollout_actors: int = 2
    candidates: int = 4
    replies: int = 2
    continuation: bool = True

    def __post_init__(self):
        if not math.isfinite(self.seconds) or self.seconds < 0 or self.nodes < 0 or min(self.beam, self.local_depth, self.actors, self.rollout_actors, self.candidates, self.replies) < 1:
            raise ValueError("invalid search configuration")


@dataclass
class Candidate:
    actions: tuple[Action, ...]
    state: State
    score: float
    reason: str
    reply: tuple[Action, ...] = ()
    verified: bool = False
    complete: bool = True
    variants: list = field(default_factory=list, repr=False)
    production_complete: bool = True


@dataclass
class Analysis:
    actions: tuple[Action, ...]
    score: float
    alternatives: list[Candidate]
    objectives: dict
    regions: list
    metrics: Metrics
    root_key: tuple
    root_player: int


class Planner:
    def __init__(self, rules=None, policy=None, value=None, config=None):
        self.rules = rules or Rules()
        self.policy = policy or HeuristicPolicy()
        self.value = value or HeuristicValue()
        self.config = config or Config()
        self.strategist = Strategist()

    def _rank(self, state, player, context):
        value = self.value.evaluate(state, player, self.rules)
        if abs(value) >= 1e8:
            return value
        guarding = 0.
        for pos, (uid, remaining) in state.captures.items():
            cap = state.units.get(uid)
            if cap and state.board.tiles[pos] == "hq":
                screens = sum(u.hp/100 for u in state.army(cap.owner)
                              if u.id != uid and state.board.distance(u.pos, pos) == 1)
                guarding += (1 if cap.owner == player else -1)*screens*1200
        return value + objective_score(state, context, self.rules) + guarding

    def _actors(self, state, allowed=None):
        enemies = state.army(1-state.player)
        units = [u for u in state.army(state.player) if not u.acted and (allowed is None or u.id in allowed)]
        ordered = [u.id for u in sorted(units, key=lambda u: (
            0 if u.pos in state.captures else 1,
            min((state.board.distance(u.pos, e.pos) for e in enemies), default=1000), u.id))]
        return ordered if getattr(self.policy, "selects_actors", False) else ordered[:self.config.actors]

    def _finish(self, start, prefix, context, budget, until, quiet=False, reason="policy turn"):
        state = start.clone()
        actions = list(prefix)
        player = state.player
        while self.rules.outcome(state) is None:
            actors = self._actors(state)
            if not getattr(self.policy, "selects_actors", False):
                actors = actors[:self.config.rollout_actors]
            if not actors or not budget.take("policy_steps", until):
                break
            proposals = self.policy.propose(state, context, self.rules, actors, limit=8, quiet=quiet)
            if not proposals:
                break
            selected = proposals[0].action
            self.rules.apply_inplace(state, selected)
            actions.append(selected)
        produced = []
        before_builds = tuple(actions)
        production_complete = self.rules.outcome(state) is not None or not any(state.owners.get(p)==player and p not in state.occupancy
                                      and self.rules.roster(state.board.tiles[p],state) for p in state.board.properties)
        if self.rules.outcome(state) is None and budget.available(until):
            produced = production_plans(state, self.rules, budget, self.config.beam, until,
                                        getattr(self.policy, "build_scores", None))
            # Conservatively withhold save labels if production exhausted the
            # budget: an absent purchase may mean the site was never searched.
            production_complete = production_complete or budget.available(until)
            state, builds, _ = max(produced, key=lambda item: item[2])
            state = state.clone()
            actions.extend(builds)
        complete = self.rules.outcome(state) is not None or all(u.acted for u in state.army(player))
        if not complete:
            budget.metrics.counts["truncated_policy_turns"] += 1
        score = self._rank(state, player, context)
        if self.rules.outcome(state) is None:
            self.rules.apply_inplace(state, END)
            actions.append(END)
        candidate = Candidate(tuple(actions), state, score, reason, complete=complete,
                              production_complete=production_complete)
        for alternative, builds, utility in produced:
            branch = alternative.clone()
            sequence = before_builds+builds
            branch_score = self._rank(branch, player, context)
            if self.rules.outcome(branch) is None:
                self.rules.apply_inplace(branch, END)
                sequence += (END,)
            if sequence != candidate.actions:
                candidate.variants.append(Candidate(sequence, branch, branch_score, "alternative production",
                                                    complete=complete,production_complete=production_complete))
        return candidate

    def _local_prefixes(self, state, context, ids, budget, until):
        """Search preparatory moves and coordinated attacks before completing a turn."""
        player = state.player
        beam = [(state, (), self._rank(state, player, context))]
        completed = list(beam)
        for depth in range(self.config.local_depth):
            expanded = []
            seen = set()
            for branch, actions, _ in beam:
                if self.rules.outcome(branch) is not None:
                    continue
                actors = self._actors(branch, set(ids))
                if not actors or not budget.available(until):
                    continue
                proposals = self.policy.propose(branch, context, self.rules, actors,
                                                limit=max(8, self.config.beam*3))
                for proposal in proposals:
                    if not budget.take("local_expansions", until):
                        break
                    child = branch.clone()
                    self.rules.apply_inplace(child, proposal.action)
                    key = child.key()
                    if key in seen:
                        budget.metrics.counts["transpositions"] += 1
                        continue
                    seen.add(key)
                    value = self._rank(child, player, context)
                    # This ranking admits negative immediate material transitions.
                    value += proposal.score*.08
                    expanded.append((child, actions+(proposal.action,), value))
            if not expanded:
                break
            expanded.sort(key=lambda item: (-item[2], tuple(action_key(a) for a in item[1])))
            beam = expanded[:self.config.beam]
            completed.extend(beam)
        completed.sort(key=lambda item: (-item[2], len(item[1])))
        return completed[:self.config.beam]

    def _turns(self, state, context, budget, until, count, include_quiet=True):
        candidates = []
        fallback = getattr(self.policy, 'fallback_policy', None)
        if fallback is not None and budget.available(until):
            # A complete heuristic turn survives even when neural inference is
            # expensive. The same fallback is present in opponent reply search.
            teacher = Planner(self.rules,fallback,self.value,self.config)
            teacher_until = min(until,perf_counter()+max(0,until-perf_counter())*.4)
            safe = teacher._finish(state,(),context,budget,teacher_until,reason='heuristic fallback')
            candidates.extend([safe,*safe.variants])
            budget.metrics.counts['heuristic_fallback_turns'] += 1
        if budget.available(until) or not candidates:
            primary = self._finish(state, (), context, budget, until)
            candidates.extend([primary,*primary.variants])
        if include_quiet and budget.available(until):
            candidates.append(self._finish(state, (), context, budget, until, quiet=True,
                                           reason="capture / defensive alternative"))
        # Each region adds alternatives; no Cartesian product across regions.
        for region_index, region in enumerate(context.regions):
            if not budget.available(until):
                break
            anchor, carried = state, ()
            if region_index:
                # Coordinate descent: carry earlier improvements into the next
                # region, validating shared occupancy/dependencies on the full map.
                incumbent = max(candidates, key=lambda c: c.score)
                anchor = state.clone()
                carry = []
                for action in incumbent.actions:
                    if action.kind in ("build", "end") or action.actor in region:
                        continue
                    if self.rules.is_legal(anchor, action):
                        self.rules.apply_inplace(anchor, action)
                        carry.append(action)
                    else:
                        budget.metrics.counts["coordination_repairs"] += 1
                carried = tuple(carry)
                from .strategy import interaction_regions
                new_regions, _ = interaction_regions(anchor)
                expanded_ids = set(region)
                for connected in new_regions:
                    if expanded_ids.intersection(connected):
                        expanded_ids.update(connected)
                region = tuple(sorted(expanded_ids))
            for branch, prefix, _ in self._local_prefixes(anchor, context, region, budget, until):
                if not prefix or not budget.available(until):
                    continue
                candidates.append(self._finish(branch, carried+prefix, context, budget, until,
                                               reason=f"coordinated local sequence ({len(region)} interacting units)"))
            budget.metrics.counts["regions_searched"] += 1
            # Keep a bounded complete-turn frontier plus the defensive incumbent.
            candidates = self._deduplicate(candidates, count)
        return self._deduplicate(candidates, count)

    def _deduplicate(self, candidates, count):
        completed = [c for c in candidates if c.complete]
        if completed:
            candidates = completed
        unique = {}
        for c in candidates:
            key = c.state.key()
            if key not in unique or c.score > unique[key].score:
                unique[key] = c
        ordered = sorted(unique.values(), key=lambda c: -c.score)
        retained = ordered[:count]
        quiet = next((c for c in ordered if not any(a.kind == "attack" for a in c.actions)), None)
        if quiet and quiet not in retained and count > 1:
            retained[-1] = quiet
        return retained

    def analyze(self, state, config=None):
        settings = getattr(getattr(self.policy, "network", None), "search_settings", None) or {}
        if settings.get("planner") == "bundle":
            from .bundle import analyze
            active = Planner(self.rules,self.policy,self.value,config) if config is not None else self
            return analyze(active,state)
        if config is not None:
            return Planner(self.rules, self.policy, self.value, config).analyze(state)
        state.validate()
        cfg = self.config
        budget = Budget(cfg.seconds, cfg.nodes)
        hits, misses = self.rules.cache_hits, self.rules.cache_misses
        root_key = state.key()
        root_player = state.player
        if self.rules.outcome(state) is not None:
            return Analysis((), self.value.evaluate(state, root_player, self.rules), [], {}, [],
                            budget.finish(), root_key, root_player)
        # Legal incumbent exists before any expensive work, even for a zero budget.
        passed = self.rules.apply(state, END)
        fallback = Candidate((END,), passed, self.value.evaluate(passed, root_player, self.rules),
                             "deadline fallback",complete=False,production_complete=False)
        context = self.strategist.plan(state, self.rules, budget, remember=True)
        budget.metrics.counts["regions"] = len(context.regions)
        budget.metrics.counts["units"] = len(state.units)
        candidates = [fallback]
        if budget.available():
            candidates = self._turns(state, context, budget, budget.fraction(.48), cfg.candidates)
        evaluated = []
        reply_sets = []
        # Opponent uses the same policy and local-sequence generator, including production/captures.
        for index, candidate in enumerate(candidates):
            if not budget.available():
                break
            terminal = self.rules.outcome(candidate.state)
            if terminal is not None:
                candidate.score = self.value.evaluate(candidate.state, root_player, self.rules)
                candidate.verified = True
                evaluated.append(candidate)
                continue
            remaining = len(candidates)-index
            reply_deadline = budget.fraction(.85) if cfg.continuation else budget.deadline
            until = perf_counter() + max(0, reply_deadline-perf_counter())/remaining
            reply_context = self.strategist.plan(candidate.state, self.rules, budget)
            replies = self._turns(candidate.state, reply_context, budget, until, cfg.replies)
            scored = []
            for reply in replies:
                if not reply.complete:
                    budget.metrics.counts["incomplete_replies"] += 1
                    continue
                result = reply.state
                score = self.value.evaluate(result, root_player, self.rules)
                scored.append((score, reply))
                budget.metrics.counts["replies_evaluated"] += 1
            if scored:
                score, worst = min(scored, key=lambda item: item[0])
                candidate.score = score
                candidate.reply = worst.actions
                candidate.verified = True
                evaluated.append(candidate)
                reply_sets.append((candidate, [reply for _, reply in scored]))
        # Extend every considered reply at the same horizon, or retain all two-turn
        # scores. Never compare a partial continuation against completed horizons.
        extensions = []
        extension_complete = bool(reply_sets)
        if cfg.continuation:
            for candidate, replies in reply_sets:
                scores = []
                for reply in replies:
                    if self.rules.outcome(reply.state) is not None:
                        scores.append((self.value.evaluate(reply.state, root_player, self.rules), reply))
                        continue
                    if not budget.available():
                        extension_complete = False
                        break
                    own_context = self.strategist.plan(reply.state, self.rules, budget)
                    continuation = self._finish(reply.state, (), own_context, budget, budget.deadline)
                    if not continuation.complete:
                        extension_complete = False
                        break
                    scores.append((self.value.evaluate(continuation.state, root_player, self.rules), reply))
                if not extension_complete:
                    break
                extensions.append((candidate, min(scores, key=lambda item: item[0])))
            if extension_complete:
                for candidate, (score, worst) in extensions:
                    candidate.score, candidate.reply = score, worst.actions
                budget.metrics.counts["horizon_turns"] = 3
        if not budget.metrics.counts["horizon_turns"]:
            budget.metrics.counts["horizon_turns"] = 2 if evaluated else 1
        eligible = evaluated or candidates
        def capture_safety(candidate):
            # Prefer an intact ongoing HQ capture when continuation values tie.
            # This is a cheap direct-threat estimate; the test oracle additionally
            # checks coordinated screen destruction and subsequent attacks.
            total = 0
            branch = candidate.state
            for pos, (uid, _) in branch.captures.items():
                cap = branch.units.get(uid)
                if not cap or cap.owner != root_player or branch.board.tiles[pos] != "hq":
                    continue
                damage = 0
                for enemy in branch.army(1-root_player):
                    if any(a.target == uid for a in self.rules.attack_actions(branch, enemy.id, ignore_acted=True)):
                        damage += self.rules.damage(branch, enemy, cap)
                total += max(0, cap.hp-damage)
            return total
        best = max(eligible, key=lambda c: (c.score, capture_safety(c)))
        alternatives = sorted(candidates, key=lambda c: (not c.verified, -c.score))
        budget.metrics.counts["complete_candidates"] = len(candidates)
        budget.metrics.counts["candidates_verified"] = len(evaluated)
        budget.metrics.counts["incomplete_candidates"] = sum(not c.complete for c in candidates)
        budget.metrics.counts["movement_cache_hits"] = self.rules.cache_hits-hits
        budget.metrics.counts["movement_cache_misses"] = self.rules.cache_misses-misses
        result = Analysis(best.actions, best.score, alternatives, context.objectives, context.regions,
                          budget.finish(), root_key, root_player)
        # External execution revalidates; it also refuses a plan for a changed state.
        return result

    def execute(self, state, analysis):
        if state.key() != analysis.root_key:
            raise ValueError("plan is stale: analyze the current state again")
        result = state.clone()
        for action in analysis.actions:
            result = self.rules.apply(result, action)
        return result


class PolicyOnlyPlanner(Planner):
    """Fast ablation opponent: the same policy and economics, without lookahead."""
    def analyze(self, state, config=None):
        if config is not None:
            return PolicyOnlyPlanner(self.rules, self.policy, self.value, config).analyze(state)
        state.validate()
        budget = Budget(self.config.seconds, self.config.nodes)
        context = self.strategist.plan(state, self.rules, budget, remember=True)
        candidate = self._finish(state, (), context, budget, budget.deadline)
        return Analysis(candidate.actions, candidate.score, [candidate], context.objectives,
                        context.regions, budget.finish(), state.key(), state.player)
