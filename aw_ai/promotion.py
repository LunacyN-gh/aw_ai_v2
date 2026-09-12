"""Weighted match-score promotion; not a statistical significance test."""
import math

def match_score(counts):
    values=[counts.get(k,0) for k in ('win','loss','censored','draw')]
    if any(not isinstance(v,int) or v<0 for v in values) or not sum(values):
        raise ValueError('invalid match counts')
    return (values[0]+.5*(values[2]+values[3]))/sum(values)

def promotion_scores(candidate, teacher_weight=.5):
    if not math.isfinite(teacher_weight) or not 0<=teacher_weight<=1:
        raise ValueError('teacher weight must be in [0,1]')
    s=candidate['summary']
    c=match_score(s['beam']);i=match_score(s['incumbent']);h=match_score(s['gate'])
    return dict(challenger_teacher_score=c,incumbent_teacher_score=i,
                challenger_head_to_head_score=h,incumbent_head_to_head_score=1-h,
                teacher_weight=teacher_weight,combined_delta=teacher_weight*(c-i)+(1-teacher_weight)*(2*h-1))

def promotion_decision(candidate, reference=None, win_rate=None, *, margin=.05, teacher_weight=.5):
    # Legacy threshold .55 is accepted as an alias for margin .05.
    if win_rate is not None:margin=win_rate-.5
    if not math.isfinite(margin) or not 0<=margin<=1:
        raise ValueError('promotion margin must be in [0,1]')
    if 'gate' not in candidate['summary']:return False,'no head-to-head evaluation'
    if 'incumbent' not in candidate['summary']:return False,'no current incumbent-versus-teacher evaluation'
    delta=promotion_scores(candidate,teacher_weight)['combined_delta']
    accepted=delta>margin+1e-12
    return accepted,f'combined score delta {delta:.4f} '+('exceeds' if accepted else 'does not exceed')+f' margin {margin:.4f}'
