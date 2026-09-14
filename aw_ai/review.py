"""Rank individual legal actions by neural successor value, without search."""
from .model import outcome_value
from .rules import Rules
from .notation import action_text

def rank_actions(network,state,k=5):
    import torch
    if not 1<=k<=5:raise ValueError('k must be between 1 and 5')
    rules=Rules(); player=state.player
    if rules.outcome(state) is not None:return []
    actions=[a for a in rules.legal_actions(state) if a.kind!='build' or a.build in network.allowed_builds]
    rows=[]
    with torch.inference_mode():
        for offset in range(0,len(actions),32):
            batch=actions[offset:offset+32]
            children=[rules.apply(state,a) for a in batch]
            live=[i for i,s in enumerate(children) if rules.outcome(s) is None]
            scores={}
            if live:
                encoded=network.encode_batch([children[i] for i in live],rules)
                values=network.value(torch.stack([x[0] for x in encoded])).flatten().tolist()
                scores={i:v*(1 if children[i].player==player else -1) for i,v in zip(live,values)}
            for i,(action,child) in enumerate(zip(batch,children)):
                winner=rules.outcome(child)
                value=outcome_value(winner,player) if winner is not None else scores[i]
                rows.append(dict(action=action,text='End turn' if action.kind=='end' else action_text(state,action),
                                 value=value,terminal=winner is not None))
    return sorted(rows,key=lambda r:-r['value'])[:k]


def rank_search_actions(network,state,k=5,seconds=1.,backend=None):
    """Best searched continuation for each distinct first action, using deployment."""
    from .guidance import deployment_agent
    from .planner import Planner,Config
    if not 1<=k<=5:raise ValueError('k must be between 1 and 5')
    rules=Rules()
    if rules.outcome(state) is not None:return []
    agent=deployment_agent(network)
    from .bundle_mcts import MCTSPlanner
    planner=(MCTSPlanner if backend=='mcts' else Planner)(rules=rules,policy=agent,value=agent,config=Config(seconds=seconds))
    analysis=planner.analyze(state)
    rows=[];seen=set()
    candidates=sorted(analysis.alternatives,key=lambda c:(-getattr(c,'visits',0),-c.score))
    for candidate in candidates:
        if not candidate.actions or candidate.actions[0] in seen:continue
        first=candidate.actions[0];seen.add(first)
        child=state.clone();plan=[]
        for action in candidate.actions:
            plan.append('End turn' if action.kind=='end' else action_text(child,action))
            child=rules.apply(child,action)
        winner=rules.outcome(child)
        rows.append(dict(action=first,text='End turn' if first.kind=='end' else action_text(state,first),
                         visits=getattr(candidate,'visits',None),score=candidate.score,plan=', '.join(plan),verified=candidate.verified,
                         complete=candidate.complete and candidate.production_complete,
                         terminal=winner is not None,
                         terminal_value=outcome_value(winner,state.player) if winner is not None else None,
                         recommended=candidate.actions==analysis.actions))
        if len(rows)>=k:break
    return rows
