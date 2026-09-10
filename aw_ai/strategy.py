from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

from .model import PROPERTY_VALUE, State
from .rules import Rules


@dataclass(frozen=True)
class Objective:
    kind: str
    destination: int
    importance: float
    reason: str


@dataclass
class Context:
    objectives: dict[str, Objective] = field(default_factory=dict)
    regions: list[tuple[str, ...]] = field(default_factory=list)
    contested: set[str] = field(default_factory=set)


class Strategist:
    def __init__(self):
        self.memory: dict[tuple[int, str], Objective] = {}
        self.board = None

    def plan(self, state: State, rules: Rules, budget=None, remember=False):
        context = Context()
        if self.board is not state.board:
            self.memory.clear()
            self.board = state.board
        player = state.player
        army = sorted(state.army(player), key=lambda u: u.id)
        enemy = state.army(1-player)
        properties = [p for p in state.board.properties if state.owners.get(p) != player]
        occupied_goals = set()
        # Finish commitments first, then sparse unit/property bidding.
        for p, (uid, remaining) in state.captures.items():
            if uid in state.units and state.units[uid].owner == player:
                context.objectives[uid] = Objective("capture", p, PROPERTY_VALUE[state.board.tiles[p]],
                                                    f"continue capture ({remaining} points remain)")
                occupied_goals.add(p)
        bids = []
        for u in army:
            if budget and not budget.available():
                break
            if not u.spec.capture or u.id in context.objectives:
                continue
            # Bound distance-field construction by a spatial shortlist.
            shortlist = sorted(properties, key=lambda p: (state.board.distance(u.pos, p), p))[:8]
            for p in shortlist:
                cost = rules.distances(state.board, u.spec.mode, p).get(u.pos)
                if cost is None:
                    continue
                turns = ceil(cost/u.spec.move) + ceil(20/u.displayed_hp)
                swing = 2 if state.owners.get(p) == 1-player else 1
                danger = min((state.board.distance(e.pos, p) for e in enemy), default=100)
                value = PROPERTY_VALUE[state.board.tiles[p]] * swing / (turns+1)
                if danger <= 3:
                    value *= .65
                previous = self.memory.get((player, u.id))
                if previous and previous.destination == p and previous.kind == "capture":
                    value *= 1.15
                bids.append((-value, u.id, p))
        for negative, uid, p in sorted(bids):
            if uid not in context.objectives and p not in occupied_goals:
                context.objectives[uid] = Objective("capture", p, -negative, "secure property and future income")
                occupied_goals.add(p)
        for u in army:
            if budget and not budget.available():
                break
            if u.hp <= 60:
                repairs = [p for p in state.board.properties if state.owners.get(p) == player
                           and (state.board.tiles[p] == "airport" if u.spec.mode == "air"
                                else state.board.tiles[p] in ("city", "factory", "hq"))]
                reachable = [(rules.distances(state.board, u.spec.mode, p).get(u.pos, 10**6), p)
                             for p in sorted(repairs, key=lambda p: state.board.distance(u.pos, p))[:3]]
                if reachable and min(reachable)[0] < 10**6 and u.id not in context.objectives:
                    context.objectives[u.id] = Objective("repair", min(reachable)[1], 2500, "recover damaged unit")
            if u.id not in context.objectives:
                threats = [e for e in enemy if state.board.tiles[e.pos] == "hq"
                           and state.owners.get(e.pos) == player]
                targets = threats or enemy
                if targets:
                    target = min(targets, key=lambda e: (state.board.distance(u.pos, e.pos), e.id))
                    context.objectives[u.id] = Objective("front", target.pos, 1500, "support contested front")
                else:
                    context.objectives[u.id] = Objective("hold", u.pos, 0, "hold position")
        context.regions, context.contested = interaction_regions(state)
        if remember:
            self.memory = {key: value for key, value in self.memory.items()
                           if key[0] != player}
            self.memory.update({(player, uid): objective for uid, objective in context.objectives.items()})
        return context


def interaction_regions(state):
    """Conservative one-turn graph including latent attacks through screens.

    Geometry deliberately overconnects; exact terrain and occupancy are enforced
    in search. Full-board simulation retains support beyond each region.
    """
    units = sorted(state.units.values(), key=lambda u: u.id)
    parent = {u.id: u.id for u in units}
    contested = set()

    def find(uid):
        while parent[uid] != uid:
            parent[uid] = parent[parent[uid]]
            uid = parent[uid]
        return uid

    def join(a, b):
        parent[find(a)] = find(b)

    for i, a in enumerate(units):
        for b in units[i+1:]:
            if a.owner == b.owner:
                continue
            reach_a = a.spec.move + a.spec.maximum
            reach_b = b.spec.move + b.spec.maximum
            if state.board.distance(a.pos, b.pos) <= max(reach_a, reach_b) + 1:
                join(a.id, b.id)
                contested.update((a.id, b.id))
    # Nearby friendly movers can block endpoints even when only one is in contact.
    for i, a in enumerate(units):
        for b in units[i+1:]:
            if a.owner == b.owner and (a.id in contested or b.id in contested):
                if state.board.distance(a.pos, b.pos) <= 2:
                    join(a.id, b.id)
                    contested.update((a.id, b.id))
    groups = {}
    for uid in sorted(contested):
        groups.setdefault(find(uid), []).append(uid)
    return sorted((tuple(g) for g in groups.values()), key=lambda g: (-len(g), g)), contested
