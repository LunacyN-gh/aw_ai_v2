"""Factory-conditioned composition and reinforcement features.

All values are marginal: a new purchase updates the friendly force before the
next site is considered. Front association is soft and terrain-aware.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .model import BASE_DAMAGE, SPECS
from .rules import movement_cost


KINDS = tuple(SPECS)


def arrival(state, rules, origin, kind, destination):
    spec = SPECS[kind]
    if movement_cost(spec.mode, state.board.tiles[destination]) is not None:
        distance = rules.distances(state.board, spec.mode, destination).get(origin, 10**6)
    else:
        # A ground counter can shoot an aircraft from an adjacent legal tile.
        distance = min((rules.distances(state.board, spec.mode, p).get(origin, 10**6)
                        for p in state.board.neighbors[destination]
                        if movement_cost(spec.mode, state.board.tiles[p]) is not None), default=10**6)
    return max(0, distance-spec.maximum)/spec.move


@dataclass
class Front:
    enemy: dict[str, float]
    friendly: dict[str, float]
    capture_demand: float
    enemies: list
    weights: list[float]
    site: int

    def vector(self):
        return [self.enemy[k]/20000 for k in KINDS] + [self.friendly[k]/20000 for k in KINDS] + [self.capture_demand/5]


def facing_front(state, rules, site):
    player = state.player
    reference = "b_copter" if state.board.tiles[site] == "airport" else "tank"
    sites = [p for p in state.board.properties if state.owners.get(p) == player
             and state.board.tiles[p] == state.board.tiles[site]]

    def relevance(pos):
        times = [arrival(state, rules, p, reference, pos) for p in sites]
        here = arrival(state, rules, site, reference, pos)
        if here > 1000:
            return 0.
        best = min(times, default=here)
        affinities = [math.exp(-min(50, max(0, t-best))/1.2) for t in times]
        share = math.exp(-min(50, max(0, here-best))/1.2)/max(1e-8, sum(affinities))
        return share/(1+here/5)

    enemy = {k: 0. for k in KINDS}
    friendly = {k: 0. for k in KINDS}
    enemies, weights = [], []
    for unit in state.units.values():
        weight = relevance(unit.pos)
        (friendly if unit.owner == player else enemy)[unit.kind] += unit.spec.cost*unit.hp/100*weight
        if unit.owner != player and weight > .001:
            enemies.append(unit)
            weights.append(weight)
    properties = [p for p in state.board.properties if state.owners.get(p) != player]
    # Counts only realistically accessible capture work, divided among bases.
    demand = sum(relevance(p)/(1+arrival(state, rules, site, "infantry", p)/4)
                 for p in properties)
    return Front(enemy, friendly, min(4., demand*.65), enemies, weights, site)


def utility_scores(state, rules, site, style="balanced"):
    front = facing_front(state, rules, site)
    enemy_mass = sum(front.enemy.values())
    enemy_count = sum(front.enemy[k]/SPECS[k].cost for k in KINDS)
    capturers = sum(front.friendly[k]/SPECS[k].cost for k in KINDS if SPECS[k].capture)
    screen_need = min(3., enemy_mass/14000)
    capture_deficit = front.capture_demand + screen_need - capturers
    result = {}
    for kind in rules.roster(state.board.tiles[site], state):
        if kind == "mech" and not any(u.owner != state.player and u.spec.mode in ("tread", "wheel")
                                       for u in state.units.values()):
            result[kind] = float("-inf")
            continue
        spec = SPECS[kind]
        inflicted = sum(front.enemy[k]*BASE_DAMAGE[kind, k]/100 for k in KINDS)
        received = sum(front.enemy[k]/max(1, SPECS[k].cost)*BASE_DAMAGE[k, kind]*spec.cost/100 for k in KINDS)
        exchange = max(0., (inflicted-.7*received)/max(1., enemy_count))
        own_coverage = sum(front.friendly[k]*sum(front.enemy[e]*BASE_DAMAGE[k, e]/100 for e in KINDS)
                           / max(1, enemy_mass) for k in KINDS)
        deficit = max(0., 1-own_coverage/max(1500., enemy_mass*.65))
        travel = sum(w*arrival(state, rules, site, kind, e.pos) for e, w in zip(front.enemies, front.weights))/max(.001, sum(front.weights))
        delivery = 1/(1+travel/5)
        combat = 1800*math.sqrt(exchange/spec.cost)*deficit*delivery
        score = combat - spec.cost*.08
        if spec.capture:
            score += 700*capture_deficit - (450 if kind == "mech" else 0)
        else:
            # A modest standing-force prior handles opening positions where no
            # combat has occurred yet; composition and coverage dominate later.
            ground = sum(front.friendly[k] for k in KINDS if not SPECS[k].capture and SPECS[k].mode != "air")
            if kind == "tank" and ground < 5000:
                score += 800*(1-ground/5000)
            score -= 160*front.friendly[kind]/spec.cost
        if style == "capture" and spec.capture:
            score += 250
        elif style == "air" and spec.mode == "air":
            score += 500
        elif style == "defensive" and kind in ("artillery", "anti_air"):
            score += 300
        result[kind] = score
    result["save"] = 0.
    return result


def production_features(state, rules, site):
    front = facing_front(state, rules, site)
    result = front.vector()
    result += [state.funds[state.player]/20000, state.funds[1-state.player]/20000,
               rules.income(state, state.player)/10000]
    for kind in KINDS:
        eta = sum(w*arrival(state, rules, site, kind, e.pos) for e, w in zip(front.enemies, front.weights))/max(.001, sum(front.weights))
        result.append(min(20., eta)/10)
    return result
