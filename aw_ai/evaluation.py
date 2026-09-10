from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol

from .model import DEFENSE, INCOME_TILES, PROPERTY_VALUE, material


class ValueModel(Protocol):
    def evaluate(self, state, perspective, rules) -> float: ...


def features(state, perspective, rules):
    result = {"material": 0.0, "funds": (state.funds[perspective]-state.funds[1-perspective])/10000,
              "properties": 0.0, "capture": 0.0, "terrain": 0.0,
              "mobility": 0.0, "income": 0.0}
    for u in state.units.values():
        sign = 1 if u.owner == perspective else -1
        result["material"] += sign*material(u)/10000
        stars = 0 if u.spec.mode == "air" else DEFENSE[state.board.tiles[u.pos]]
        result["terrain"] += sign*stars*u.hp/10000
        # Counts and absolute economic scale stay visible on larger maps.
        result["mobility"] += sign*u.spec.move*u.hp/10000
    for p, owner in state.owners.items():
        sign = 1 if owner == perspective else -1
        result["properties"] += sign*PROPERTY_VALUE[state.board.tiles[p]]/10000
        if state.board.tiles[p] in INCOME_TILES:
            result["income"] += sign*.1
    for p, (uid, remaining) in state.captures.items():
        unit = state.units[uid]
        sign = 1 if unit.owner == perspective else -1
        swing = 2 if state.owners.get(p) == 1-unit.owner else 1
        # Progress is contingent: completing a property remains a discrete gain.
        result["capture"] += sign*PROPERTY_VALUE[state.board.tiles[p]]*swing*(20-remaining)/20*.35/10000
    return result


class HeuristicValue:
    weights = {"material": 1., "funds": .9, "properties": 1., "capture": 1.,
               "terrain": .3, "mobility": .1, "income": .5}

    def evaluate(self, state, perspective, rules):
        winner = rules.outcome(state)
        if winner is not None:
            return 1e9 if winner == perspective else -1e9
        return 10000 * sum(self.weights[k]*v for k, v in features(state, perspective, rules).items())


class LinearValue(HeuristicValue):
    """Small trainable value baseline. Deliberately not a calibrated win predictor."""
    def __init__(self, weights=None):
        self.weights = dict(weights or HeuristicValue.weights)
        if set(self.weights) != set(HeuristicValue.weights) or not all(math.isfinite(v) for v in self.weights.values()):
            raise ValueError("invalid value weights")

    def save(self, path):
        Path(path).write_text(json.dumps({"format": "aw-v2-linear-value", "version": 1,
                                         "weights": self.weights}, indent=2)+"\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "aw-v2-linear-value" or data.get("version") != 1:
            raise ValueError("unsupported value model")
        return cls(data["weights"])

    def fit(self, samples, epochs=20, rate=.01):
        """Fit tanh(value/4) to terminal outcomes; truncations must be excluded."""
        history = []
        for _ in range(epochs):
            loss = 0.
            for x, target in samples:
                score = sum(self.weights[k]*x[k] for k in self.weights)/4
                prediction = math.tanh(score)
                error = prediction-target
                loss += error*error
                scale = max(1., sum(v*v for v in x.values()))
                for k in self.weights:
                    self.weights[k] -= rate*2*error*(1-prediction*prediction)*x[k]/(4*scale)
            history.append(loss/max(1, len(samples)))
        return history


def objective_score(state, context, rules):
    value = 0.
    for uid, objective in context.objectives.items():
        u = state.units.get(uid)
        if u is None or objective.kind == "hold":
            continue
        distance = rules.distances(state.board, u.spec.mode, objective.destination).get(u.pos, 1000)
        if objective.kind == "front":
            distance = max(0, distance-u.spec.maximum)
        value -= min(100, distance)* (85 if objective.kind == "capture" else 40)
    return value
