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
