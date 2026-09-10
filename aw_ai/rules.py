from __future__ import annotations

import heapq
from collections import OrderedDict
from dataclasses import replace

from .model import (Action, BASE_DAMAGE, DEFENSE, END, INCOME_TILES, PROFILE,
                    SPECS, State, Unit)


def movement_cost(mode, tile):
    if mode == "air":
        return 1
    if tile == "water":
        return None
    if tile in ("mountain", "river"):
        return (1 if mode == "mech" else 2) if mode in ("foot", "mech") else None
    if tile == "forest":
        return {"foot": 1, "mech": 1, "tread": 2, "wheel": 3}[mode]
    return 2 if mode == "wheel" and tile == "plain" else 1


class Rules:
    """Exact execution for the declared deterministic experimental profile.

    Transit caching depends on enemy blockers, movement class and origin, not HP
    or acted flags. Friendly occupancy is checked separately at endpoints.
    """

    def __init__(self, cache_size=4096):
        self.cache_size = cache_size
        self.transit = OrderedDict()
        self.fields = OrderedDict()
        self.coverage = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def _remember(self, cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > self.cache_size:
            cache.popitem(last=False)
        return value

    def reachable(self, state, uid, ignore_acted=False):
        unit = state.units[uid]
        if unit.acted and not ignore_acted:
            return {}
        # This profile conservatively treats every enemy as a transit blocker.
        blocked = tuple(sorted(u.pos for u in state.units.values() if u.owner != unit.owner
                               and state.board.distance(unit.pos, u.pos) <= unit.spec.move))
        key = (state.board, state.profile, unit.pos, unit.spec.mode, unit.spec.move, blocked)
        if key in self.transit:
            self.cache_hits += 1
            costs = self.transit[key]
            self.transit.move_to_end(key)
        else:
            self.cache_misses += 1
            costs = {unit.pos: 0}
            heap = [(0, unit.pos)]
            blockers = set(blocked)
            while heap:
                cost, p = heapq.heappop(heap)
                if cost != costs[p]:
                    continue
                for q in state.board.neighbors[p]:
                    step = movement_cost(unit.spec.mode, state.board.tiles[q])
                    if step is None or q in blockers or cost + step > unit.spec.move:
                        continue
                    if cost + step < costs.get(q, 10**9):
                        costs[q] = cost + step
                        heapq.heappush(heap, (cost + step, q))
            self._remember(self.transit, key, costs)
        return {p: cost for p, cost in costs.items() if p == unit.pos or p not in state.occupancy}

    def attack_squares(self, state, uid):
        unit = state.units[uid]
        endpoints = (unit.pos,) if unit.spec.indirect else tuple(sorted(self.reachable(state, uid, True)))
        key = (state.board, unit.spec.minimum, unit.spec.maximum, endpoints)
        if key in self.coverage:
            self.coverage.move_to_end(key)
            return self.coverage[key]
        touched = set()
        board = state.board
        for p in endpoints:
            x, y = board.xy(p)
            for dx in range(-unit.spec.maximum, unit.spec.maximum+1):
                for dy in range(-unit.spec.maximum+abs(dx), unit.spec.maximum-abs(dx)+1):
                    if abs(dx)+abs(dy) < unit.spec.minimum:
                        continue
                    if 0 <= x+dx < board.width and 0 <= y+dy < board.height:
                        touched.add(board.pos(x+dx, y+dy))
        return self._remember(self.coverage, key, frozenset(touched))

    def distances(self, board, mode, goal):
        """Reverse Dijkstra: terrain-aware cost FROM any tile TO the goal."""
        key = (board, mode, goal)
        if key in self.fields:
            self.fields.move_to_end(key)
            return self.fields[key]
        costs = {goal: 0}
        heap = [(0, goal)]
        while heap:
            cost, p = heapq.heappop(heap)
            if costs[p] != cost:
                continue
            enter = movement_cost(mode, board.tiles[p])
            if enter is None:
                continue
            for q in board.neighbors[p]:
                if movement_cost(mode, board.tiles[q]) is None:
                    continue
                if cost + enter < costs.get(q, 10**9):
                    costs[q] = cost + enter
                    heapq.heappush(heap, (cost + enter, q))
        return self._remember(self.fields, key, costs)

    def attack_actions(self, state, uid, ignore_acted=False):
        u = state.units[uid]
        if u.acted and not ignore_acted:
            return []
        endpoints = (u.pos,) if u.spec.indirect else self.reachable(state, uid, ignore_acted)
        result = []
        for p in endpoints:
            for enemy in state.units.values():
                if enemy.owner == u.owner or BASE_DAMAGE[u.kind, enemy.kind] == 0:
                    continue
                d = state.board.distance(p, enemy.pos)
                if u.spec.minimum <= d <= u.spec.maximum:
                    result.append(Action("attack", uid, p, enemy.id))
        return result

    def roster(self, tile, state=None):
        return tuple(k for k, spec in SPECS.items()
                     if ((tile == "airport" and spec.mode == "air")
                         or (tile == "factory" and spec.mode != "air"))
                     and (state is None or state.allowed_builds is None or k in state.allowed_builds))

    def legal_actions(self, state, actor=None):
        if self.outcome(state) is not None:
            return
        for unit in state.units.values():
            if unit.owner != state.player or unit.acted or (actor is not None and actor != unit.id):
                continue
            yield from self.attack_actions(state, unit.id)
            for p in sorted(self.reachable(state, unit.id)):
                yield Action("wait", unit.id, p)
                if unit.spec.capture and p in state.board.properties and state.owners.get(p) != unit.owner:
                    yield Action("capture", unit.id, p)
        if actor is None:
            for p in state.board.properties:
                if state.owners.get(p) != state.player or p in state.occupancy:
                    continue
                for kind in self.roster(state.board.tiles[p], state):
                    if SPECS[kind].cost <= state.funds[state.player]:
                        yield Action("build", destination=p, build=kind)
            yield END

    def is_legal(self, state, action):
        if self.outcome(state) is not None:
            return False
        if action.kind == "end":
            return action == END
        if action.kind == "build":
            p = action.destination
            return (p in state.board.properties and state.owners.get(p) == state.player
                    and p not in state.occupancy and action.build in self.roster(state.board.tiles[p], state)
                    and SPECS[action.build].cost <= state.funds[state.player])
        unit = state.units.get(action.actor)
        if unit is None or unit.owner != state.player or unit.acted:
            return False
        if action.kind == "attack":
            return action in self.attack_actions(state, unit.id)
        if action.destination not in self.reachable(state, unit.id):
            return False
        if action.kind == "wait":
            return True
        return (action.kind == "capture" and unit.spec.capture
                and action.destination in state.board.properties
                and state.owners.get(action.destination) != unit.owner)

    def damage(self, state, attacker, defender, defender_pos=None, towers=None):
        base = BASE_DAMAGE[attacker.kind, defender.kind]
        if not base:
            return 0
        p = defender.pos if defender_pos is None else defender_pos
        stars = 0 if defender.spec.mode == "air" else DEFENSE[state.board.tiles[p]]
        if towers is None:
            towers = sum(owner == attacker.owner and state.board.tiles[q] == "tower"
                         for q, owner in state.owners.items())
        # No luck, no COs, no ammo selection. Internal HP 1..100, displayed HP ceil(hp/10).
        return min(defender.hp, base * attacker.displayed_hp * (100 + 10*towers)
                   * max(0, 100-stars*defender.displayed_hp) // 100000)

    def preview(self, state, action):
        attacker = replace(state.units[action.actor], pos=action.destination)
        target = state.units[action.target]
        damage = self.damage(state, attacker, target)
        counter = 0
        if damage < target.hp and not target.spec.indirect and state.board.distance(attacker.pos, target.pos) == 1:
            counter = self.damage(state, replace(target, hp=target.hp-damage), attacker)
        return damage, counter

    def apply(self, state, action):
        if not self.is_legal(state, action):
            raise ValueError(f"illegal action: {action}")
        result = state.clone()
        self.apply_inplace(result, action)
        return result

    def apply_inplace(self, state, action):
        """Trusted path: only use for actions generated/validated for this state."""
        if action.kind == "end":
            state.player = 1-state.player
            state.turn += 1
            self.start_turn(state)
            return
        if action.kind == "build":
            uid = f"p{state.player}_{state.next_id}"
            while uid in state.units:
                state.next_id += 1
                uid = f"p{state.player}_{state.next_id}"
            state.next_id += 1
            u = Unit(uid, state.player, action.build, action.destination, acted=True)
            state.units[uid] = u
            state.occupancy[u.pos] = uid
            state.funds[state.player] -= u.spec.cost
            return
        uid = action.actor
        state.clear_capture(uid, action.destination if action.kind == "capture" else -1)
        if action.kind == "attack":
            damage, counter = self.preview(state, action)
            target = state.units[action.target]
            if damage >= target.hp:
                state.remove(target.id)
            else:
                state.update(target.id, hp=target.hp-damage)
            attacker = state.units[uid]
            if counter >= attacker.hp:
                state.remove(uid)
            else:
                state.update(uid, hp=attacker.hp-counter, pos=action.destination, acted=True)
        else:
            state.update(uid, pos=action.destination, acted=True)
            if action.kind == "capture":
                p = action.destination
                old = state.captures.get(p)
                remaining = (old[1] if old and old[0] == uid else 20) - state.units[uid].displayed_hp
                if remaining <= 0:
                    old_owner = state.owners.get(p)
                    state.owners[p] = state.player
                    state.captures.pop(p, None)
                    if state.board.tiles[p] == "hq" and old_owner == 1-state.player:
                        state.winner = state.player
                else:
                    state.captures[p] = (uid, remaining)
        winner = self.outcome(state)
        if winner is not None:
            state.winner = winner

    def start_turn(self, state):
        """Call once for a fresh map; END calls it for subsequent turns."""
        player = state.player
        state.funds[player] += self.income(state, player)
        for u in sorted(state.army(player), key=lambda u: u.id):
            tile = state.board.tiles[u.pos]
            repairs = tile == "airport" if u.spec.mode == "air" else tile in ("city", "factory", "hq")
            heal = 0
            if repairs and state.owners.get(u.pos) == player:
                cost = max(1, u.spec.cost // 100)
                heal = min(20, 100-u.hp, state.funds[player] // cost)
                state.funds[player] -= heal * cost
            state.update(u.id, hp=u.hp+heal, acted=False)

    def income(self, state, player):
        return 1000 * sum(owner == player and state.board.tiles[p] in INCOME_TILES
                          for p, owner in state.owners.items())

    def outcome(self, state):
        if state.winner is not None:
            return state.winner
        if state.income_capture_limit is not None:
            for player in (0, 1):
                if self.income(state, player)//1000 >= state.income_capture_limit:
                    return player
        # Production-capable empty armies are not eliminated. This is a sandbox rule.
        present = [any(u.owner == p for u in state.units.values()) or
                   any(o == p and state.board.tiles[q] in ("factory", "airport")
                       for q, o in state.owners.items()) for p in (0, 1)]
        if present[0] != present[1]:
            return 0 if present[0] else 1
        return None

    def observe(self, state, player):
        if player not in (0, 1):
            raise ValueError("invalid observer")
        return state.clone()  # Explicit full-information boundary; fog not implemented.
