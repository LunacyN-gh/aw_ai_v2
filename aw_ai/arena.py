from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .evaluation import features
from .model import Action
from .rules import Rules
from .scenarios import digest, from_data, to_data


def play(initial, planners, max_turns=100):
    state = initial.clone()
    rules = planners[0].rules
    record = {"format": "aw-ai-v2-replay", "version": 1, "initial": to_data(state), "turns": [],
              "result": None, "termination": "turn_limit", "samples": []}
    record["agents"] = [{"type": type(p).__name__, "config": asdict(p.config),
                         "value_weights": getattr(p.value, "weights", None)} for p in planners]
    seen = {}
    for _ in range(max_turns):
        winner = rules.outcome(state)
        if winner is not None:
            record["result"], record["termination"] = winner, "victory"
            break
        signature = state.key()
        seen[signature] = seen.get(signature, 0)+1
        if seen[signature] >= 3:
            record["termination"] = "repetition"
            break
        planner = planners[state.player]
        observation = rules.observe(state, state.player)
        analysis = planner.analyze(observation)
        record["samples"].append({"player": state.player, "features": features(state, state.player, rules)})
        state = planner.execute(state, analysis)
        record["turns"].append({"actions": [asdict(a) for a in analysis.actions],
                                "hash": digest(state), "metrics": analysis.metrics.as_dict()})
    winner = rules.outcome(state)
    if winner is not None:
        record["result"], record["termination"] = winner, "victory"
    return record, state


def save_replay(record, path):
    Path(path).write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")


def replay(record, rules=None):
    if record.get("format") != "aw-ai-v2-replay" or record.get("version") != 1:
        raise ValueError("unsupported replay")
    rules = rules or Rules()
    state = from_data(record["initial"])
    for index, turn in enumerate(record["turns"]):
        for raw in turn["actions"]:
            state = rules.apply(state, Action(**raw))
        if digest(state) != turn["hash"]:
            raise ValueError(f"replay diverged at turn {index}")
    return state


def terminal_samples(records):
    samples = []
    for record in records:
        # A time/turn limit is censored data, not a draw or loss label.
        if record["termination"] != "victory":
            continue
        for sample in record["samples"]:
            samples.append((sample["features"], 1. if sample["player"] == record["result"] else -1.))
    return samples
