from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

from .model import Action, Board, PROFILE, SPECS, State, Unit
from .rules import movement_cost, Rules


def to_data(state):
    return {"format": "aw-ai-v2-state", "version": 2, "profile": state.profile,
            "width": state.board.width, "height": state.board.height,
            "player": state.player, "funds": state.funds, "turn": state.turn,
            "winner": state.winner, "next_id": state.next_id,
            **({"allowed_builds": list(state.allowed_builds)} if state.allowed_builds is not None else {}),
            **({"income_capture_limit": state.income_capture_limit} if state.income_capture_limit is not None else {}),
            "tiles": list(state.board.tiles), "owners": {str(k): v for k, v in state.owners.items()},
            "units": [asdict(u) for u in sorted(state.units.values(), key=lambda u: u.id)],
            "captures": {str(k): list(v) for k, v in state.captures.items()}}


def from_data(data):
    if data.get("format") == "advance-wars-ai-map" and data.get("version") == 1:
        width, height = int(data["width"]), int(data["height"])
        board = Board.plain(width, height, {(int(t["x"]), int(t["y"])): t["type"] for t in data.get("terrain", [])})
        units = {}
        for u in data.get("units", []):
            if not 0 <= int(u["x"]) < width or not 0 <= int(u["y"]) < height:
                raise ValueError("unit outside board")
            if u["id"] in units:
                raise ValueError("duplicate unit id")
            units[u["id"]] = Unit(u["id"], int(u["owner"]), u["type"], board.pos(int(u["x"]), int(u["y"])), int(u["hp"]))
        owners = {board.pos(int(t["x"]), int(t["y"])): int(t["owner"])
                  for t in data.get("terrain", []) if t.get("owner") is not None}
        captures = {board.pos(int(c["x"]), int(c["y"])): (c["unit_id"], int(c["remaining"]))
                    for c in data.get("captures", [])}
        funds = [int(data.get("funds", {}).get(str(p), 0)) for p in (0, 1)]
        state = State(board, units, int(data.get("current_player", 0)), funds, owners, captures,
                      income_capture_limit=data.get("income_capture_limit"))
    elif data.get("format") == "aw-ai-v2-state" and data.get("version") == 2:
        board = Board(int(data["width"]), int(data["height"]), tuple(data["tiles"]))
        units = {}
        for entry in data["units"]:
            unit = Unit(**entry)
            if unit.id in units:
                raise ValueError("duplicate unit id")
            units[unit.id] = unit
        state = State(board, units, int(data["player"]), list(data["funds"]),
                      {int(k): v for k, v in data["owners"].items()},
                      {int(k): tuple(v) for k, v in data["captures"].items()},
                      int(data["turn"]), data.get("winner"), int(data.get("next_id", 0)), data["profile"],
                      data.get("income_capture_limit"))
    else:
        raise ValueError("unsupported map/state format")
    if "allowed_builds" in data:
        state.allowed_builds = tuple(data["allowed_builds"])
    state.validate()
    if any(movement_cost(u.spec.mode, board.tiles[u.pos]) is None for u in state.units.values()):
        raise ValueError("unit on impassable terrain")
    # Legacy map files describe opening balances; v2 state files are snapshots.
    if data.get("format") == "advance-wars-ai-map" and not data.get("turn_started", False):
        Rules().start_turn(state)
    return state


def save(state, path):
    Path(path).write_text(json.dumps(to_data(state), indent=2)+"\n", encoding="utf-8")


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return from_data(data)


def digest(state):
    return hashlib.sha256(json.dumps(to_data(state), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def league_scale(size=20, units_per_side=12, seed=0, contact=False):
    """Synthetic scale fixture, NOT a downloaded or certified league map.

    20x20 default: three bases, one airport, one tower per side and 32 cities.
    Both axes and ownership are rotationally symmetric; seed changes terrain.
    """
    if size < 8 or size > 64 or units_per_side < 0:
        raise ValueError("fixture size must be 8..64 and units nonnegative")
    rng = random.Random(seed)
    terrain, owners_xy = {}, {}

    def mirror(p):
        return size-1-p[0], size-1-p[1]

    for p, tile in { (1, 1): "hq", (1, 2): "factory", (2, 1): "factory",
                     (1, 4): "factory", (3, 2): "airport", (3, 4): "tower"}.items():
        terrain[p] = tile
        terrain[mirror(p)] = tile
        owners_xy[p], owners_xy[mirror(p)] = 0, 1
    spaces = [(x, y) for y in range(size) for x in range(size)
              if (x, y) not in terrain and mirror((x, y)) not in terrain and (x, y) < mirror((x, y))]
    rng.shuffle(spaces)
    city_pairs = min(16, size*size//12, len(spaces))
    for p in spaces[:city_pairs]:
        terrain[p] = terrain[mirror(p)] = "city"
        if sum(p) < size//2:
            owners_xy[p], owners_xy[mirror(p)] = 0, 1
    for p in spaces[city_pairs:]:
        roll = rng.random()
        if roll < .12:
            terrain[p] = terrain[mirror(p)] = "forest"
        elif roll < .16:
            terrain[p] = terrain[mirror(p)] = "mountain"
    # Two transit corridors guarantee strategic connectivity for ground vehicles.
    for y in (size//3, size-1-size//3):
        for x in range(size):
            p = (x, y)
            if terrain.get(p) not in ("city", "factory", "airport", "tower", "hq"):
                terrain[p] = "road"
    board = Board.plain(size, size, terrain)
    candidates = [p for p in range(size*size) if sum(board.xy(p)) < size-1
                  and board.tiles[p] not in ("mountain", "water", "river")]
    if contact:
        candidates.sort(key=lambda p: (abs(sum(board.xy(p))-(size-3)), abs(board.xy(p)[0]-size//2), p))
    else:
        candidates.sort(key=lambda p: (sum(board.xy(p)), p))
    if units_per_side > len(candidates):
        raise ValueError("too many units for fixture")
    roster = ("infantry", "infantry", "tank", "infantry", "artillery", "tank",
              "anti_air", "infantry", "b_copter", "recon", "mech", "md_tank")
    units = {}
    for i, p in enumerate(candidates[:units_per_side]):
        for owner, q in ((0, p), (1, size*size-1-p)):
            uid = f"p{owner}_initial_{i}"
            units[uid] = Unit(uid, owner, roster[i % len(roster)], q)
    state = State(board, units, funds=[12000, 12000],
                  owners={board.pos(*p): owner for p, owner in owners_xy.items()})
    Rules().start_turn(state)
    state.validate()
    return state


def tactical(name):
    if name == "screen":
        board = Board.plain(7, 1)
        units = [Unit("gun", 0, "artillery", 0), Unit("tank", 0, "tank", 1),
                 Unit("screen", 1, "infantry", 3, hp=10), Unit("target", 1, "artillery", 5)]
        return State(board, {u.id: u for u in units})
    if name == "blocker":
        board = Board.plain(5, 2, {(x, 1): "mountain" for x in range(5) if x != 2})
        units = [Unit("tank", 0, "tank", 0), Unit("friend", 0, "infantry", 2),
                 Unit("enemy", 1, "artillery", 3, hp=40)]
        return State(board, {u.id: u for u in units})
    if name == "capture":
        board = Board.plain(6, 3, {(1, 1): "city", (4, 1): "factory"})
        units = [Unit("cap", 0, "infantry", 7), Unit("chip", 1, "infantry", 8),
                 Unit("enemy", 1, "tank", 16)]
        return State(board, {u.id: u for u in units}, owners={10: 1}, captures={7: ("cap", 10)})
    raise ValueError(f"unknown tactical fixture: {name}")


def action_text(action, board):
    def square(p):
        x, y = board.xy(p)
        label = ""
        x += 1
        while x:
            x, remainder = divmod(x-1, 26)
            label = chr(97+remainder)+label
        return f"{label}{board.height-y}"
    if action.kind == "end":
        return "End turn"
    if action.kind == "build":
        return f"Build {action.build} at {square(action.destination)}"
    extra = f" -> {action.target}" if action.target else ""
    return f"{action.actor}: {action.kind} {square(action.destination)}{extra}"
