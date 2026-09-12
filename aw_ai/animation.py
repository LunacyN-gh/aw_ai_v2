"""Presentation paths use the same terrain costs and blockers as the rules."""
import heapq
from .rules import movement_cost


def movement_path(state, uid, destination, rules):
    unit=state.units[uid]
    if destination not in rules.reachable(state,uid):
        raise ValueError('animation destination is not reachable')
    blocked={u.pos for u in state.units.values() if u.owner!=unit.owner}
    costs={unit.pos:0}; previous={}; heap=[(0,unit.pos)]
    while heap:
        cost,p=heapq.heappop(heap)
        if cost!=costs[p]: continue
        if p==destination:
            path=[p]
            while p!=unit.pos: p=previous[p]; path.append(p)
            return list(reversed(path))
        for q in state.board.neighbors[p]:
            step=movement_cost(unit.spec.mode,state.board.tiles[q])
            if step is None or q in blocked: continue
            total=cost+step
            if total<=unit.spec.move and total<costs.get(q,float('inf')):
                costs[q]=total; previous[q]=p; heapq.heappush(heap,(total,q))
    raise ValueError('no valid animation path')
