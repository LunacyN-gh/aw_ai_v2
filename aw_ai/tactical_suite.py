"""Evaluation-only tactical families, with exhaustive small-position oracles.

Training code never imports these generators. Uniqueness refers to resulting
layouts, not permutations of independent actions.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

from .model import Action, Board, END, State, Unit
from .rules import Rules
from .scenarios import to_data


@dataclass
class Puzzle:
    name: str
    family: str
    state: State
    objective: str
    target: str
    required_hp: int = 0


def breakthrough(target="artillery"):
    striker = "anti_air" if target == "b_copter" else "tank"
    hp = {"artillery": 60, "anti_air": 55, "b_copter": 90}[target]
    board = Board.plain(9, 5, {(x, y): "water" for y in range(5) for x in range(9) if y in (0, 4)})
    units = [Unit("gun", 0, "artillery", board.pos(2, 2)),
             Unit("striker", 0, striker, board.pos(1, 2)),
             Unit("prize", 1, target, board.pos(6, 2), hp=hp)]
    for i, (x, y, hp) in enumerate(((4, 1, 100), (4, 2, 10), (4, 3, 100),
                                   (6, 1, 100), (6, 3, 100), (7, 2, 100))):
        units.append(Unit(f"screen_{i}", 1, "infantry", board.pos(x, y), hp=hp))
    return Puzzle(f"breakthrough_{target}", "wall_breakthrough", State(board, {u.id: u for u in units}),
                  "destroy", "prize")


def hq_wall():
    paths = {(2, 1), (3, 1), (4, 1), (5, 1), (5, 2), (4, 2),
             (3, 3), (4, 3), (5, 0), (6, 0), (2, 3)}
    terrain = {(x, y): "water" for y in range(5) for x in range(8) if (x, y) not in paths}
    terrain[5, 2] = "hq"
    board = Board.plain(8, 5, terrain)
    units = [Unit("capturer", 0, "infantry", board.pos(5, 2)),
             Unit("north", 0, "mech", board.pos(6, 0)),
             Unit("west", 0, "mech", board.pos(3, 3)),
             Unit("tank", 1, "tank", board.pos(2, 1)),
             Unit("gun", 1, "artillery", board.pos(2, 3))]
    hq = board.pos(5, 2)
    state = State(board, {u.id: u for u in units}, owners={hq: 1}, captures={hq: ("capturer", 20)})
    return Puzzle("hq_double_gate", "hq_screen", state, "protect_capture", "capturer", 100)


def transform(puzzle, rotation=0, swap=False):
    state = puzzle.state
    board = state.board
    width, height = (board.height, board.width) if rotation % 2 else (board.width, board.height)

    def convert(p):
        x, y = board.xy(p)
        if rotation % 4 == 1:
            x, y = y, board.width-1-x
        elif rotation % 4 == 2:
            x, y = board.width-1-x, board.height-1-y
        elif rotation % 4 == 3:
            x, y = board.height-1-y, x
        return y*width+x
    tiles = ["plain"]*(width*height)
    for p, tile in enumerate(board.tiles):
        tiles[convert(p)] = tile
    units = {uid: replace(u, pos=convert(u.pos), owner=1-u.owner if swap else u.owner)
             for uid, u in state.units.items()}
    changed = State(Board(width, height, tuple(tiles)), units, 1-state.player if swap else state.player,
                    list(reversed(state.funds)) if swap else state.funds.copy(),
                    {convert(p): 1-o if swap else o for p, o in state.owners.items()},
                    {convert(p): capture for p, capture in state.captures.items()})
    return replace(puzzle, name=f"{puzzle.name}_r{rotation}_s{int(swap)}", state=changed)


def cases():
    base = [breakthrough(k) for k in ("artillery", "anti_air", "b_copter")] + [hq_wall()]
    return [transform(p, r, r % 2 == 1) for p in base for r in range(4)]


class Oracle:
    def __init__(self, rules=None, max_states=200000):
        self.rules = rules or Rules()
        self.max_states = max_states
        self.states = 0
        self.safety_cache = {}

    def tick(self):
        self.states += 1
        if self.states > self.max_states:
            raise RuntimeError("oracle budget exceeded; uniqueness/protection is unproven")

    def protected(self, state, puzzle):
        uid = puzzle.target
        cap = state.units.get(uid)
        if cap is None or cap.hp < puzzle.required_hp:
            return False
        progress = state.captures.get(cap.pos)
        if progress is None or progress[0] != uid or progress[1] > 10:
            return False
        if state.player == cap.owner:
            state = self.rules.apply(state, END)
        responder = state.player

        def visit(branch):
            self.tick()
            unit = branch.units.get(uid)
            if unit is None or unit.hp < puzzle.required_hp:
                return False
            if branch.player != responder or self.rules.outcome(branch) is not None:
                return True
            key = (branch.key(), uid, puzzle.required_hp)
            if key in self.safety_cache:
                return self.safety_cache[key]
            for action in self.rules.legal_actions(branch):
                if action == END:
                    continue  # Ending early cannot damage a still-intact capturer.
                child = self.rules.apply(branch, action)
                if not visit(child):
                    self.safety_cache[key] = False
                    return False
            self.safety_cache[key] = True
            return True
        return visit(state)

    def success(self, state, puzzle):
        if puzzle.objective == "destroy":
            return puzzle.target not in state.units
        return self.protected(state, puzzle)

    def solve(self, puzzle):
        root = puzzle.state
        player = root.player
        visited, solutions = set(), {}

        def visit(state, path):
            self.tick()
            key = state.key()
            if key in visited:
                return
            visited.add(key)
            if puzzle.objective == "destroy" and puzzle.target not in state.units:
                solutions[key] = path
                return
            if all(u.acted for u in state.army(player)):
                if self.success(state, puzzle):
                    # Capturer and screens define a wall, not enemy reply choices.
                    layout = tuple(sorted((u.id, u.pos, u.hp) for u in state.army(player)))
                    solutions[layout] = path
                return
            for action in self.rules.legal_actions(state):
                if action.kind in ("end", "build"):
                    continue
                child = self.rules.apply(state, action)
                # Leaving the HQ cannot satisfy this objective later in the turn.
                if puzzle.objective == "protect_capture" and action.actor == puzzle.target and action.kind != "capture":
                    continue
                visit(child, path+(action,))
        visit(root, ())
        return list(solutions.values())


def evaluate_suite(planner_factory, puzzles=None, prove=False):
    report = []
    for puzzle in puzzles or cases():
        started = perf_counter()
        planner = planner_factory()
        analysis = planner.analyze(puzzle.state)
        state = planner.execute(puzzle.state, analysis)
        oracle = Oracle()
        passed = oracle.success(state, puzzle)
        found = passed or any(Oracle().success(c.state, puzzle) for c in analysis.alternatives)
        row = {"name": puzzle.name, "family": puzzle.family, "passed": passed,
               "solution_in_candidates": found,
               "elapsed": perf_counter()-started, "metrics": analysis.metrics.as_dict()}
        if prove:
            solutions = Oracle().solve(puzzle)
            row["oracle_solutions"] = len(solutions)
        report.append(row)
    return {"passed": sum(r["passed"] for r in report),
            "found": sum(r["solution_in_candidates"] for r in report), "total": len(report), "cases": report}
