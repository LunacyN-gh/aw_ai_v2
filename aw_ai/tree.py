"""Experimental sampled whole-turn UCT with progressive widening.

The proven-by-tests default remains the local beam planner. This alternative
uses the same exact simulator and policy interface; it makes no strength claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import random

from .budget import Budget
from .model import END
from .planner import Analysis, Candidate, Planner
from .policy import HeuristicPolicy


class SamplePolicy:
    def __init__(self, base, seed):
        self.base = base
        self.random = random.Random(seed)
        self.selects_actors = getattr(base, "selects_actors", False)
        if hasattr(base, "build_scores"):
            self.build_scores = base.build_scores

    def propose(self, *args, **kwargs):
        proposals = self.base.propose(*args, **kwargs)
        # Gumbel sampling preserves access to non-greedy proposals. Scores are
        # heuristic logits, not claimed learned policy probabilities.
        sampled = [replace(p, score=p.score-450*math.log(-math.log(max(1e-12, self.random.random()))))
                   for p in proposals]
        return sorted(sampled, key=lambda p: -p.score)


@dataclass
class Node:
    state: object
    actions: tuple = ()
    visits: int = 0
    total: float = 0.
    children: list = field(default_factory=list)

    @property
    def mean(self):
        return self.total/max(1, self.visits)


class SampledPlanner(Planner):
    def __init__(self, *args, seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed

    def analyze(self, state, config=None):
        if config is not None:
            return SampledPlanner(self.rules, self.policy, self.value, config, seed=self.seed).analyze(state)
        state.validate()
        budget = Budget(self.config.seconds, self.config.nodes)
        player = state.player
        if self.rules.outcome(state) is not None:
            return Analysis((), self.value.evaluate(state, player, self.rules), [], {}, [], budget.finish(), state.key(), player)
        root = Node(state)
        context = self.strategist.plan(state, self.rules, budget, remember=True)
        baseline = self._finish(state, (), context, budget, budget.fraction(.25))
        root.children.append(Node(baseline.state, baseline.actions))
        random_policy = SamplePolicy(self.policy, self.seed)
        generator = Planner(self.rules, random_policy, self.value, self.config)
        iteration = 0
        while budget.take("tree_iterations"):
            iteration += 1
            node = root
            path = [root]
            depth = 0
            while self.rules.outcome(node.state) is None and depth < 4 and budget.available():
                width = min(self.config.candidates, 1+int(math.sqrt(node.visits)))
                if len(node.children) < width:
                    ctx = self.strategist.plan(node.state, self.rules, budget)
                    # Bound each proposal so a single expansion cannot monopolize the tree.
                    until = min(budget.deadline, budget.started+(budget.deadline-budget.started)*(.25+.12*iteration))
                    proposal = generator._finish(node.state, (), ctx, budget, until,
                                                 quiet=(iteration % 5 == 0), reason="sampled whole turn")
                    if proposal.complete and not any(c.state.key() == proposal.state.key() for c in node.children):
                        child = Node(proposal.state, proposal.actions)
                        node.children.append(child)
                        node = child
                        path.append(node)
                        budget.metrics.counts["tree_expansions"] += 1
                        break
                if not node.children:
                    break
                sign = 1 if node.state.player == player else -1
                node = max(node.children, key=lambda c: sign*c.mean +
                           1.4*math.sqrt(math.log(node.visits+2)/(c.visits+1)))
                path.append(node)
                depth += 1
            raw = self.value.evaluate(node.state, player, self.rules)
            value = 1. if raw >= 1e8 else -1. if raw <= -1e8 else math.tanh(raw/20000)
            for visited in path:
                visited.visits += 1
                visited.total += value
            budget.metrics.counts["tree_max_depth"] = max(budget.metrics.counts["tree_max_depth"], len(path)-1)
        best = max(root.children, key=lambda c: (c.visits, c.mean))
        alternatives = []
        for child in sorted(root.children, key=lambda c: (-c.visits, -c.mean)):
            reply = max(child.children, key=lambda n: n.visits).actions if child.children else ()
            alternatives.append(Candidate(child.actions, child.state, child.mean,
                                           f"sampled tree: {child.visits} visits; bounded value, not win probability",
                                           reply=reply, verified=bool(child.children)))
        budget.metrics.counts["root_children"] = len(root.children)
        return Analysis(best.actions, best.mean, alternatives, context.objectives, context.regions,
                        budget.finish(), state.key(), player)
