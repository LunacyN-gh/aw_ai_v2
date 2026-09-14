"""Shared partial-turn beam with rollout ranking and matched greedy replies.

This is an order-sensitive beam, not MCTS. All evaluators use the same legal
state transitions, transpositions, production branching and reply scheduling.
"""
import math
from dataclasses import dataclass
from .budget import Budget
from .evaluation import HeuristicValue
from .model import Action, END, outcome_value
from .planner import Planner, Analysis, Candidate


@dataclass
class Prefix:
    state: object
    actions: tuple = ()
    sites: tuple = ()


class OrderPlanner(Planner):
    def _policy(self):
        return getattr(self.policy, 'neural', self.policy)

    def _score(self, state, player):
        winner=self.rules.outcome(state)
        if winner is not None:return 1e9*outcome_value(winner,player)
        network=getattr(self.policy,'network',None)
        if network is None:return self.value.evaluate(state,player,self.rules)/20000
        weight=(network.search_settings or {}).get('neural_value_weight',1.)
        n=self._policy().evaluate(state,player,self.rules)/20000 if weight else 0.
        h=math.tanh(HeuristicValue().evaluate(state,player,self.rules)/20000) if weight<1 else 0.
        return weight*n+(1-weight)*h

    def _options(self, prefix, context, branch=True):
        state=prefix.state;policy=self._policy()
        if self.rules.outcome(state) is not None:return []
        actors=[u.id for u in state.army(state.player) if not u.acted]
        if actors:
            proposals=policy.propose(state,context,self.rules,actors,limit=128 if branch else 1)
            if not branch:return [(p.action,None) for p in proposals[:1]]
            # Reserve branches for distinct actors and attacks/captures. This
            # prevents one actor's many destination variants filling the beam.
            selected=[];seen=set()
            def add(p):
                if p.action not in seen:selected.append((p.action,None));seen.add(p.action)
            for p in proposals[:3]:add(p)
            actor_seen=set();interactions=set()
            for p in proposals:
                a=p.action
                if a.actor not in actor_seen:
                    add(p);actor_seen.add(a.actor)
                key=(a.actor,a.kind,a.target if a.kind=='attack' else a.destination)
                if a.kind in ('attack','capture') and key not in interactions:
                    add(p);interactions.add(key)
                if len(selected)>=12:break
            return selected[:12]
        sites=[p for p in state.board.properties if state.owners.get(p)==state.player
               and p not in state.occupancy and p not in prefix.sites
               and self.rules.roster(state.board.tiles[p],state)]
        if not sites:return [(END,None)]
        site=sites[0];scores=policy.build_scores(state,self.rules,site)
        options=[]
        for kind,score in sorted(scores.items(),key=lambda kv:(-kv[1],kv[0])):
            action=None if kind=='save' else Action('build',destination=site,build=kind)
            if math.isfinite(score) and (action is None or self.rules.is_legal(state,action)):
                options.append((action,site))
        return options[:4 if branch else 1] or [(None,site)]

    def _step(self,prefix,option):
        action,site=option
        state=prefix.state.clone()
        if action is not None:self.rules.apply_inplace(state,action)
        return Prefix(state,prefix.actions+((action,) if action is not None else ()),
                      prefix.sites+((site,) if site is not None else ()))

    def _complete(self,prefix,context,budget=None,until=None):
        start_player=prefix.state.player
        while prefix.state.player==start_player and self.rules.outcome(prefix.state) is None:
            if budget is not None and not budget.take('order_rollout_steps',until):return None
            options=self._options(prefix,context,False)
            if not options:options=[(END,None)]
            prefix=self._step(prefix,options[0])
        return Candidate(prefix.actions,prefix.state,0.,'ordered rollout')

    def analyze(self,state,config=None):
        if config is not None:return OrderPlanner(self.rules,self.policy,self.value,config).analyze(state)
        state.validate();player=state.player;budget=Budget(self.config.seconds,self.config.nodes)
        context=self.strategist.plan(state,self.rules,budget,remember=True)
        if self.rules.outcome(state) is not None:
            return Analysis((),self._score(state,player),[],context.objectives,context.regions,budget.finish(),state.key(),player)
        # An atomic legal baseline, just like the existing neural backend.
        baseline=self._complete(Prefix(state.clone()),context)
        baseline.reason='greedy baseline';baseline.score=self._score(baseline.state,player)
        candidates={baseline.state.key():baseline}
        frontier=[Prefix(state.clone())];seen=set();until=budget.fraction(.55)
        while frontier and budget.available(until):
            expanded=[]
            for prefix in frontier:
                if not budget.available(until):break
                for option in self._options(prefix,context):
                    if not budget.take('order_expansions',until):break
                    child=self._step(prefix,option)
                    key=(child.state.key(),child.sites)
                    if key in seen:
                        budget.metrics.counts['order_transpositions']+=1;continue
                    seen.add(key)
                    terminal=self.rules.outcome(child.state) is not None or child.state.player!=player
                    complete=Candidate(child.actions,child.state,0.,'ordered beam') if terminal else self._complete(child,context,budget,until)
                    if complete is None:continue
                    complete.score=self._score(complete.state,player)
                    candidates.setdefault(complete.state.key(),complete)
                    if not terminal:expanded.append((complete.score,child))
            # Keep distinct acted-unit sets first, then fill by rollout value.
            ranked=sorted(expanded,key=lambda x:-x[0]);frontier=[];groups=set()
            for _,child in ranked:
                signature=tuple(sorted(u.id for u in child.state.army(player) if u.acted))
                if signature not in groups and len(frontier)<self.config.beam:
                    groups.add(signature);frontier.append(child)
            for _,child in ranked:
                if len(frontier)>=self.config.beam:break
                if not any(child is p for p in frontier):frontier.append(child)
        # Compare completed plans at equal depth. Finish the baseline reply
        # first, then admit each challenger only after its reply completes.
        roots=[baseline]+[c for c in sorted(candidates.values(),key=lambda c:-c.score) if c is not baseline][:7]
        pool=[]
        for c in roots:
            if self.rules.outcome(c.state) is not None:
                c.verified=True;pool.append(c);continue
            if not budget.available():continue
            ctx=self.strategist.plan(c.state,self.rules,budget)
            reply=self._complete(Prefix(c.state.clone()),ctx,budget)
            if reply is None:continue
            c.score=self._score(reply.state,player);c.reply=reply.actions;c.verified=True;pool.append(c)
            budget.metrics.counts['order_checked_roots']+=1
        if not baseline.verified:
            pool=[baseline]+[c for c in roots if self.rules.outcome(c.state)==player]
        best=max(pool,key=lambda c:c.score)
        budget.metrics.counts['order_complete_candidates']=len(candidates)
        budget.metrics.counts['order_baseline_selected']=int(best is baseline)
        return Analysis(best.actions,best.score,pool,context.objectives,context.regions,budget.finish(),state.key(),player)
