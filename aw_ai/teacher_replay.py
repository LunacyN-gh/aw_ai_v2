"""Protected bootstrap reservoir plus recent teacher corrections."""
from collections import deque
import random


def balanced_sample(pool, count, rng):
    if not pool or count <= 0:
        return []
    units = [e for e in pool if e.site is None]
    builds = [e for e in pool if e.site is not None]
    if units and builds:
        return rng.sample(units,min(len(units),(count+1)//2))+rng.sample(builds,min(len(builds),count//2))
    return rng.sample(units or builds,min(len(pool),count))


class TeacherReplay:
    def __init__(self, size=8192, seed=0):
        if size < 2: raise ValueError('teacher replay size must be at least two')
        self.anchors = []
        self.capacity = size//2
        self.recent = deque(maxlen=size-self.capacity)
        self.seen = 0
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.anchors)+len(self.recent)

    def extend(self, examples):
        for e in examples:
            if e.source == 'bootstrap':
                self.seen += 1
                if len(self.anchors) < self.capacity: self.anchors.append(e)
                else:
                    index = self.rng.randrange(self.seen)
                    if index < self.capacity: self.anchors[index] = e
            else:
                self.recent.append(e)

    def sample(self, count, rng):
        def exact(pool, n):
            if not n or not pool: return []
            units = [e for e in pool if e.site is None]
            builds = [e for e in pool if e.site is not None]
            def draw(xs, size):
                return rng.sample(xs,size) if len(xs) >= size else rng.choices(xs,k=size)
            if units and builds:
                return draw(units,(n+1)//2)+draw(builds,n//2)
            return draw(units or builds,n)
        if self.anchors and self.recent:
            n = (count+1)//2
            return exact(self.anchors,n)+exact(list(self.recent),count-n)
        return exact(self.anchors or list(self.recent),count)


def training_batch(replay, teacher, fraction, count, rng):
    if teacher is None or not len(teacher):
        return balanced_sample(replay,count,rng)
    n = max(1,round(count*fraction)) if fraction > 0 else 0
    if not replay: n = count
    expert = teacher.sample(n,rng)
    return expert+balanced_sample(replay,count-len(expert),rng)
