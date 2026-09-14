"""PUCT over complete turns, with progressive widening and root visit targets."""
from dataclasses import dataclass,field
import hashlib,math,random
from .budget import Budget
from .model import outcome_value, END
from .planner import Analysis, Candidate, Config
from .policy_bundle import policy_turn,preference
from .order_search import OrderPlanner,Prefix
from .planner import Planner


class MCTSPlanner(Planner):
    """Explicit backend override for testing existing checkpoints."""
    def analyze(self,state,config=None):
        if not hasattr(self.policy,'network'):
            active=Planner(self.rules,self.policy,self.value,config or self.config)
            return analyze(active,state)
        from .guidance import GuidedAgent
        agent=self.policy if hasattr(self.policy,'neural') else GuidedAgent(self.policy,1.)
        active=Planner(self.rules,agent,agent,config or self.config)
        return analyze(active,state)


DEFAULTS=dict(mcts_simulations=128,mcts_depth=6,mcts_c_puct=1.5,
              mcts_widening=2.,mcts_max_children=16,mcts_visit_temperature=1.)


def validate(settings):
    for key,default in DEFAULTS.items():
        value=settings.get(key,default)
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError(f'{key} must be positive and finite')
        if key in ('mcts_simulations','mcts_depth','mcts_max_children') and (type(value) is not int):
            raise ValueError(f'{key} must be an integer')


@dataclass
class Node:
    state: object
    edges: list = field(default_factory=list)
    visits: int = 0
    attempts: int = 0
    duplicates: int = 0


@dataclass
class Edge:
    candidate: object
    child: Node
    log_prior: float = 0.
    visits: int = 0
    total: float = 0.
    prior: float = 1.

    @property
    def q(self):return self.total/self.visits if self.visits else 0.


def priors(edges):
    peak=max(e.log_prior for e in edges)
    raw=[math.exp(max(-20.,e.log_prior-peak)) for e in edges]
    total=sum(raw)
    for e,p in zip(edges,raw):e.prior=.9*p/total+.1/len(edges)


def select(node,root_player,c_puct):
    # All values are stored from root perspective. Opponent nodes minimize Q.
    sign=1 if node.state.player==root_player else -1
    return max(node.edges,key=lambda e:sign*e.q+c_puct*e.prior*math.sqrt(node.visits+1)/(1+e.visits))


def backup(nodes,edges,value):
    for node in nodes:node.visits+=1
    for edge in edges:edge.visits+=1;edge.total+=value


def heuristic_turn(generator,state,rng,budget=None,temperature=0.):
    """Rank-based heuristic proposals, no neural weights or inference.

    Use the ordered teacher's legal options and state transitions. Normalized
    rank probabilities avoid comparing arbitrary heuristic score magnitudes.
    """
    local_budget=budget or Budget(3600,100000)
    context=generator.strategist.plan(state,generator.rules,local_budget)
    prefix=Prefix(state.clone());player=state.player;logs=[]
    while prefix.state.player==player and generator.rules.outcome(prefix.state) is None:
        if budget is not None and not budget.take('heuristic_mcts_steps'):return None
        options=generator._options(prefix,context,True)
        if not options:options=[(END,None)]
        raw=[math.exp(-i/max(temperature,.7)) for i in range(len(options))]
        index=rng.choices(range(len(options)),weights=raw)[0] if temperature else 0
        logs.append(math.log(raw[index]/sum(raw)))
        prefix=generator._step(prefix,options[index])
    candidate=Candidate(prefix.actions,prefix.state,0.,'sampled heuristic' if temperature else 'greedy heuristic')
    candidate.policy_preference=sum(logs)/max(1,len(logs))
    return candidate


def analyze(planner,state):
    state.validate();network=getattr(planner.policy,'network',None)
    settings=(network.search_settings or {}) if network is not None else {};validate(settings)
    cfg={k:settings.get(k,v) for k,v in DEFAULTS.items()}
    budget=Budget(planner.config.seconds,planner.config.nodes)
    root_player=state.player;neural=planner.policy.neural if network is not None else None;rules=planner.rules
    rng=random.Random(settings.get('exploration_seed',0)+int.from_bytes(hashlib.sha256(repr(state.key()).encode()).digest()[:8],'little'))
    scorer=OrderPlanner(rules,planner.policy,planner.value,planner.config)
    from .policy import HeuristicPolicy
    allowed=neural.network.allowed_builds if neural is not None else getattr(planner.policy,'allowed_builds',None)
    teacher=OrderPlanner(rules,HeuristicPolicy(allowed),config=planner.config)
    context=planner.strategist.plan(state,rules,budget,remember=True)
    def value(s):
        budget.metrics.counts['mcts_value_evaluations']+=1
        winner=rules.outcome(s)
        if winner is not None:return outcome_value(winner,root_player)
        raw=scorer._score(s,root_player)
        return max(-1.,min(1.,raw)) if neural is not None else math.tanh(raw)
    if rules.outcome(state) is not None:
        return Analysis((),value(state),[],context.objectives,context.regions,budget.finish(),state.key(),root_player)
    root=Node(state.clone())
    # Atomic raw fallback. Remaining proposals and tree traversal are budgeted.
    def turn(s,temperature=0.,limited=True):
        if neural is not None:return policy_turn(neural,s,rules,rng,temperature,budget if limited else None)
        return heuristic_turn(teacher,s,rng,budget if limited else None,temperature)
    baseline=turn(state,limited=False)
    first=Edge(baseline,Node(baseline.state),getattr(baseline,'policy_preference',0.))
    root.edges.append(first)
    def propose(node):
        attempt=node.attempts;node.attempts+=1
        if attempt==0:
            return turn(node.state)
        if attempt==1:
            ctx=teacher.strategist.plan(node.state,rules,budget)
            c=teacher._complete(Prefix(node.state.clone()),ctx,budget)
            if c is not None:c.policy_preference=preference(neural,node.state,c,rules) if neural is not None else getattr(baseline,'policy_preference',0.)
            return c
        if attempt==2 and node is root:
            # Spend a bounded slice proposing an order-searched turn. Charge
            # its work to this tree's shared cap; never recursively call MCTS.
            from time import perf_counter
            seconds=max(0.,min(planner.config.seconds*.12,budget.deadline-perf_counter()))
            nodes=max(0,min(300,budget.limit-budget.used))
            if not seconds or not nodes:return None
            ordered=OrderPlanner(rules,planner.policy,planner.value,Config(seconds=seconds,nodes=nodes))
            result=ordered.analyze(node.state)
            budget.used+=result.metrics.counts['nodes']
            budget.metrics.counts['nodes']+=result.metrics.counts['nodes']
            budget.metrics.counts['mcts_order_proposals']+=1
            c=next(c for c in result.alternatives if c.actions==result.actions)
            c.policy_preference=preference(neural,node.state,c,rules) if neural is not None else getattr(baseline,'policy_preference',0.)
            return c
        temperature=settings.get('proposal_temperature',.7)
        if rng.random()<settings.get('exploration_fraction',.25):temperature=max(1.2,temperature)
        return turn(node.state,temperature)
    root.attempts=1
    for _ in range(cfg['mcts_simulations']):
        if not budget.take('mcts_simulations'):break
        node=root;nodes=[root];path=[];depth=0
        while rules.outcome(node.state) is None and depth<cfg['mcts_depth']:
            if not budget.available():break
            limit=min(cfg['mcts_max_children'],max(1,math.ceil(cfg['mcts_widening']*math.sqrt(node.visits+1))))
            expanded=None
            if len(node.edges)<limit and node.duplicates<8:
                c=propose(node)
                if c is not None:
                    if any(e.child.state.key()==c.state.key() for e in node.edges):
                        node.duplicates+=1;budget.metrics.counts['mcts_duplicates']+=1
                    else:
                        node.duplicates=0
                        expanded=Edge(c,Node(c.state),getattr(c,'policy_preference',0.))
                        node.edges.append(expanded);priors(node.edges)
                        budget.metrics.counts['mcts_expansions']+=1
            if not node.edges:break
            edge=expanded or select(node,root_player,cfg['mcts_c_puct'])
            path.append(edge);node=edge.child;nodes.append(node);depth+=1
            if expanded:break
        # Do not back up an unfinished simulation that never chose an action.
        if not path:break
        backup(nodes,path,value(node.state))
        budget.metrics.counts['mcts_completed_simulations']+=1
        budget.metrics.counts['mcts_depth_sum']+=depth
        budget.metrics.counts['mcts_max_depth']=max(depth,budget.metrics.counts['mcts_max_depth'])
    visited=[e for e in root.edges if e.visits]
    winners=[e for e in root.edges if rules.outcome(e.child.state)==root_player]
    pool=winners or visited or [first]
    best=max(pool,key=lambda e:(e.visits,e.q))
    # Preserve visit distributions, not a softmax over Q. Exact immediate wins
    # take precedence, even if discovered near the deadline with few visits.
    peak=max(e.visits for e in pool)
    weights=[math.exp((math.log(e.visits)-math.log(peak))/cfg['mcts_visit_temperature']) if e.visits else 0. for e in pool] if peak else [1.]*len(pool)
    for e in pool:
        e.candidate.score=value(e.child.state) if not e.visits else e.q
        e.candidate.visits=e.visits;e.candidate.verified=bool(e.visits)
    budget.metrics.counts['mcts_root_children']=len(root.edges)
    budget.metrics.counts['mcts_root_visits']=root.visits
    budget.metrics.counts['mcts_greedy_selected']=int(best is first)
    budget.metrics.counts['mcts_fallback']=int(not visited)
    result=Analysis(best.candidate.actions,best.candidate.score,[e.candidate for e in pool],
                    context.objectives,context.regions,budget.finish(),state.key(),root_player)
    result.visit_weights={e.candidate.actions:w/sum(weights) for e,w in zip(pool,weights)}
    result.mcts_diagnostics=[dict(visits=e.visits,q=e.q,prior=e.prior,source=e.candidate.reason) for e in root.edges]
    return result
