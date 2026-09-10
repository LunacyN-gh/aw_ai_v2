"""Conservative acceptance gate, not a statistical strength certificate."""
import math


def promotion_decision(candidate, reference, win_rate=.55):
    gate = candidate['summary'].get('gate')
    if gate is None:
        return False,'no head-to-head evaluation'
    games = sum(gate.values())
    # Censored games cannot supply evidence of a win.
    if not games or gate['win'] < math.ceil(games*win_rate) or gate['win'] <= gate['loss']:
        return False,'insufficient wins against the incumbent'
    for mode in ('policy','beam'):
        a,b = candidate['summary'][mode],reference['summary'][mode]
        if a['win'] < b['win'] or a['loss'] > b['loss']:
            return False,f'{mode} regressed against the teacher'
    checks = candidate.get('openings',{})
    if not checks.get('beam') or not all(checks['beam'].values()):
        return False,'opening regression in deployed search'
    if sum(checks.get('policy',{}).values()) < sum(reference.get('openings',{}).get('policy',{}).values()):
        return False,'opening regression in raw policy'
    return True,'head-to-head threshold passed without teacher/opening regression'
