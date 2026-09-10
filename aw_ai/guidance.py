"""Teacher-assisted deployment, without changing the network or legal actions."""
import math
from .evaluation import HeuristicValue
from .policy import HeuristicPolicy, Proposal


class GuidedAgent:
    selects_actors = True

    def __init__(self, neural, value_weight=0.):
        if not math.isfinite(value_weight) or not 0 <= value_weight <= 1:
            raise ValueError('neural value weight must be in [0, 1]')
        self.neural = neural
        self.network = neural.network
        self.value_weight = value_weight
        self.fallback_policy = HeuristicPolicy(self.network.allowed_builds)
        self.heuristic_value = HeuristicValue()

    def propose(self, state, context, rules, actors=None, limit=12, quiet=False):
        teacher = self.fallback_policy.propose(state,context,rules,actors,limit,quiet)
        learned = self.neural.propose(state,context,rules,actors,limit,quiet)
        # Reserve slots for each source. Logit magnitude cannot exclude teacher
        # proposals. Ranks provide a common, bounded tie-breaking scale.
        chosen = {}
        for rank in range(max(len(teacher),len(learned))):
            for source, label in ((teacher,'heuristic'),(learned,'neural')):
                if rank < len(source):
                    p = source[rank]
                    if p.action not in chosen and len(chosen) < limit:
                        chosen[p.action] = Proposal(p.action,1000/(rank+1),label+' mixed proposal')
        return list(chosen.values())

    def build_scores(self, state, rules, site):
        teacher = self.fallback_policy.build_scores(state,rules,site)
        learned = self.neural.build_scores(state,rules,site)
        # Keep teacher economic scale. The bounded neural adjustment cannot
        # replace a reliable purchase with a huge uncalibrated save logit.
        keys = [k for k,v in learned.items() if math.isfinite(v)]
        ranks = {k:i for i,k in enumerate(sorted(keys,key=lambda k:(-learned[k],k)))}
        return {k:(v+250*(1-ranks.get(k,len(keys))/max(1,len(keys))))
                if math.isfinite(v) and k in ranks else float('-inf') for k,v in teacher.items()}

    def evaluate(self, state, player, rules):
        heuristic = self.heuristic_value.evaluate(state,player,rules)
        if abs(heuristic) >= 1e8 or self.value_weight == 0:
            return heuristic
        return (1-self.value_weight)*heuristic+self.value_weight*self.neural.evaluate(state,player,rules)


def deployment_agent(network):
    from .neural import NeuralAgent
    neural = NeuralAgent(network)
    settings = getattr(network,'search_settings',None)
    return GuidedAgent(neural,settings['neural_value_weight']) if settings else neural
