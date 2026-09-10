from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class Metrics:
    counts: Counter = field(default_factory=Counter)
    elapsed: float = 0.0
    exhausted: bool = False
    reason: str = ""

    def as_dict(self):
        return {"elapsed": round(self.elapsed, 6), "exhausted": self.exhausted,
                "reason": self.reason, "counts": dict(sorted(self.counts.items()))}


class Budget:
    def __init__(self, seconds=1.0, nodes=5000, metrics=None):
        if seconds < 0 or nodes < 0:
            raise ValueError("budgets must be nonnegative")
        self.started = perf_counter()
        self.deadline = self.started + seconds
        self.limit = nodes
        self.metrics = metrics or Metrics()
        self.used = 0

    def available(self, until=None):
        now = perf_counter()
        if self.used >= self.limit or now >= self.deadline:
            self.metrics.exhausted = True
            self.metrics.reason = "nodes" if self.used >= self.limit else "time"
            return False
        return until is None or now < until

    def take(self, phase, until=None):
        if not self.available(until):
            return False
        self.used += 1
        self.metrics.counts["nodes"] += 1
        self.metrics.counts[phase] += 1
        return True

    def fraction(self, fraction):
        return self.started + (self.deadline-self.started)*fraction

    def finish(self):
        self.metrics.elapsed = perf_counter()-self.started
        return self.metrics
