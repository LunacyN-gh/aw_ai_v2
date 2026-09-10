from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Protocol

from .model import Action, BASE_DAMAGE, DEFENSE, PROPERTY_VALUE, SPECS, material
from .evaluation import HeuristicValue


@dataclass(frozen=True)
class Proposal:
    action: Action
    score: float
    reason: str


class Policy(Protocol):
    def propose(self, state, context, rules, actors=None, limit=12, quiet=False) -> list[Proposal]: ...


class Threats:
    """Shared sparse attack sources for one state/player, used for ranking only."""
    def __init__(self, state, rules, defender):
        self.sources = {}
        self.state = state
        self.rules = rules
        self.losses = {}
        self.towers = sum(owner == 1-defender and state.board.tiles[p] == "tower"
                          for p, owner in state.owners.items())
        for e in state.army(1-defender):
            for p in rules.attack_squares(state, e.id):
                self.sources.setdefault(p, []).append(e)

    def loss(self, unit, p):
        key = (unit.kind, unit.hp, p)
        if key in self.losses:
            return self.losses[key]
        damage = sum(self.rules.damage(self.state, e, unit, p, self.towers) for e in self.sources.get(p, ()))
        damage = min(unit.hp, damage)
        value = unit.spec.cost*(damage/100 + (.15 if damage == unit.hp else 0))
        self.losses[key] = value
        return value


class HeuristicPolicy:
    def __init__(self, allowed_builds=None):
        self.allowed_builds = frozenset(allowed_builds) if allowed_builds is not None else frozenset(SPECS)

    def build_scores(self, state, rules, site):
        from .production import utility_scores
        return {k: v if k == "save" or k in self.allowed_builds else float("-inf")
                for k, v in utility_scores(state, rules, site).items()}

    def propose(self, state, context, rules, actors=None, limit=12, quiet=False):
        allowed = set(actors) if actors is not None else None
        units = [u for u in state.army(state.player) if not u.acted and (allowed is None or u.id in allowed)]
        threats = Threats(state, rules, state.player)
        proposals = []
        per_actor = []
        for unit in units:
            options = []
            objective = context.objectives.get(unit.id)
            field = rules.distances(state.board, unit.spec.mode, objective.destination) if objective else {}
            start_distance = field.get(unit.pos, 1000)
            position_scores = {}

            def position_score(p):
                if p in position_scores:
                    return position_scores[p]
                progress = start_distance-field.get(p, 1000) if objective else 0
                if objective and objective.kind == "front":
                    progress = max(0, start_distance-unit.spec.maximum)-max(0, field.get(p, 1000)-unit.spec.maximum)
                score = progress*(120 if objective and objective.kind == "capture" else 65)
                stars = 0 if unit.spec.mode == "air" else DEFENSE[state.board.tiles[p]]
                score += stars*unit.hp*.12
                score -= threats.loss(unit, p)*.48
                score -= state.board.distance(unit.pos, p)*.5
                for cap_pos, (cap_id, remaining) in state.captures.items():
                    cap = state.units.get(cap_id)
                    if (cap and cap.owner == unit.owner and cap.id != unit.id
                            and state.board.tiles[cap_pos] == "hq"
                            and state.board.distance(p, cap_pos) == 1):
                        score += 2500*unit.hp/100
                if state.board.tiles[p] in ("factory", "airport") and state.owners.get(p) == unit.owner:
                    if state.funds[unit.owner] >= min((SPECS[k].cost for k in rules.roster(state.board.tiles[p], state)), default=float("inf")):
                        score -= 600
                if unit.hp <= 60 and state.owners.get(p) == unit.owner:
                    if (state.board.tiles[p] == "airport" if unit.spec.mode == "air"
                            else state.board.tiles[p] in ("city", "factory", "hq")):
                        score += unit.spec.cost*(100-unit.hp)/100*.3
                active = state.captures.get(unit.pos)
                if active and active[0] == unit.id:
                    score -= PROPERTY_VALUE[state.board.tiles[unit.pos]]*.3
                position_scores[p] = score
                return score

            for p in rules.reachable(state, unit.id):
                score = position_score(p)
                options.append(Proposal(Action("wait", unit.id, p), score,
                                        objective.reason if objective else "hold or reposition"))
                if unit.spec.capture and p in state.board.properties and state.owners.get(p) != unit.owner:
                    active = state.captures.get(p)
                    remaining = active[1] if active and active[0] == unit.id else 20
                    swing = 2 if state.owners.get(p) == 1-unit.owner else 1
                    gain = PROPERTY_VALUE[state.board.tiles[p]]*swing
                    capture = gain*(.8 if remaining <= unit.displayed_hp else .23)
                    if active and active[0] == unit.id:
                        capture += gain*.35
                    options.append(Proposal(Action("capture", unit.id, p), score+capture,
                                            "complete capture" if remaining <= unit.displayed_hp else "advance capture"))
            if not quiet:
                for action in rules.attack_actions(state, unit.id):
                    damage, counter = rules.preview(state, action)
                    target = state.units[action.target]
                    gain = target.spec.cost*(damage/100 + (.15 if damage == target.hp else 0))
                    gain -= unit.spec.cost*(counter/100 + (.15 if counter == unit.hp else 0))
                    capture = state.captures.get(target.pos)
                    if capture and damage < target.hp:
                        before = ceil(capture[1]/target.displayed_hp)
                        after = ceil(capture[1]/ceil((target.hp-damage)/10))
                        gain += max(0, after-before)*PROPERTY_VALUE[state.board.tiles[target.pos]]*.4
                    # Remove the destroyed target's projected threat from this ranking.
                    if damage == target.hp and target in threats.sources.get(action.destination, ()):
                        gain += .48*rules.damage(state, target, unit, action.destination)*unit.spec.cost/100
                    if damage == target.hp:
                        # Rank a clearance by the NEW follow-up attacks it opens.
                        # This admits low-value screen kills before expensive strikes.
                        child = rules.apply(state, action)
                        unlocked = 0.
                        for friend in child.army(unit.owner):
                            if friend.acted or friend.spec.indirect:
                                continue
                            previous = {(a.destination, a.target) for a in rules.attack_actions(state, friend.id)}
                            for follow in rules.attack_actions(child, friend.id):
                                if (follow.destination, follow.target) in previous:
                                    continue
                                hit, retaliation = rules.preview(child, follow)
                                victim = child.units[follow.target]
                                benefit = victim.spec.cost*(hit/100 + (.15 if hit == victim.hp else 0))
                                benefit -= friend.spec.cost*retaliation/100
                                unlocked = max(unlocked, benefit)
                        gain += unlocked*.9
                    options.append(Proposal(action, gain+position_score(action.destination),
                                            f"destroy {target.id}" if damage == target.hp else f"damage {target.id}"))
            options.sort(key=lambda p: (-p.score, action_key(p.action)))
            if options:
                per_actor.append(options[0])
                # Keep quiet, capture and attack options even when a different family ranks highest.
                seen = set()
                kept = []
                for family in ("attack", "capture", "wait"):
                    first = next((p for p in options if p.action.kind == family), None)
                    if first:
                        kept.append(first)
                        seen.add(first.action)
                for proposal in options[:4]:
                    if proposal.action not in seen:
                        kept.append(proposal)
                        seen.add(proposal.action)
                proposals.extend(kept)
        proposals.sort(key=lambda p: (-p.score, action_key(p.action)))
        # Reserve alternatives from other actors, including a possible blocker mover.
        reserved = sorted(per_actor, key=lambda p: (-p.score, action_key(p.action)))[:max(1, limit//3)]
        result = list(reserved)
        seen = {p.action for p in result}
        for proposal in proposals:
            if proposal.action not in seen:
                result.append(proposal)
                seen.add(proposal.action)
            if len(result) >= limit:
                break
        return sorted(result[:limit], key=lambda p: (-p.score, action_key(p.action)))


def action_key(action):
    return action.kind, action.actor, action.destination, action.target, action.build


def production_plans(state, rules, budget, width=4, until=None, scorer=None):
    """A single bounded beam across factories AND airports; saving is always legal."""
    width = max(2, width)  # One purchasing candidate plus the explicit save option.
    frontier = [(state, (), 0.)]
    for p in state.board.properties:
        if state.owners.get(p) != state.player or p in state.occupancy or not rules.roster(state.board.tiles[p], state):
            continue
        expanded = list(frontier)  # Explicit do-not-buy at this production site.
        for branch, actions, bonus in frontier:
            from .production import utility_scores
            scores = scorer(branch, rules, p) if scorer else utility_scores(branch, rules, p)
            for kind in rules.roster(state.board.tiles[p], state):
                if not isfinite(scores.get(kind, float("-inf"))):
                    continue
                if SPECS[kind].cost > branch.funds[state.player]:
                    continue
                if not budget.take("production_expansions", until):
                    break
                action = Action("build", destination=p, build=kind)
                candidate = branch.clone()
                rules.apply_inplace(candidate, action)
                gain = scores[kind]-scores.get("save", 0.)
                expanded.append((candidate, actions+(action,), bonus+gain))
        # Rank utility relative to saving, not raw army count/presence bonuses.
        expanded.sort(key=lambda item: (-item[2], len(item[1]), tuple(action_key(a) for a in item[1])))
        frontier = expanded[:width]
        if not any(not actions for _, actions, _ in frontier):
            frontier[-1] = (state, (), 0.)
        budget.metrics.counts["production_peak_frontier"] = max(budget.metrics.counts["production_peak_frontier"], len(frontier))
        if not budget.available(until):
            break
    return frontier


def build_utility(state, rules, p, kind):
    from .production import utility_scores
    return utility_scores(state, rules, p)[kind]
