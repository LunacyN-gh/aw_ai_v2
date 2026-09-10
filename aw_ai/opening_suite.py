"""Fixed opening behavior checks, evaluated separately from replay sampling."""
from pathlib import Path
import random
from .model import Action, END
from .rules import Rules
from .scenarios import load
from .planner import Planner, Config


def fixtures():
    rules = Rules()
    path = Path(__file__).resolve().parents[1]/'maps/v1/test_cities_6x6.json'
    for side, base, city, uid in ((0,0,7,'human_infantry_1'),(1,35,28,'ai_infantry_1')):
        state = load(path)
        if side: state = rules.apply(state,END)
        yield f'opening_{side}', state, base, None
        state = rules.apply(state,Action('capture',uid,city))
        state = rules.apply(rules.apply(state,END),END)
        yield f'finish_capture_{side}', state, base, city


def opening_checks(network):
    from .guidance import deployment_agent
    from .neural_training import learned_turn
    result = {}
    for mode in ('policy','beam'):
        checks = {}
        for name,state,base,city in fixtures():
            rules = Rules(); player = state.player
            if mode == 'policy':
                after,_ = learned_turn(network,state,rules,random.Random(0),record=False,greedy=True)
            else:
                agent = deployment_agent(network)
                planner = Planner(rules=rules,policy=agent,value=agent,config=Config(seconds=2,nodes=1000))
                after = planner.execute(state,planner.analyze(state))
            if city is not None:
                checks[name] = after.owners.get(city) == player
            else:
                built = after.units.get(after.occupancy.get(base))
                expanded = any(after.units[uid].owner == player for uid,_ in after.captures.values())
                checks[name] = bool(built and built.owner == player and built.kind == 'infantry' and expanded)
        result[mode] = checks
    return result
