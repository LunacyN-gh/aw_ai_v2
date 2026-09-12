"""AWPGN v1: coordinate actions, player-turn numbering, embedded initial state."""
import json
import re
from .model import Action, END, DRAW
from .rules import Rules
from .scenarios import to_data, from_data

CODES={'infantry':'i','mech':'m','recon':'r','tank':'t','md_tank':'mt',
       'artillery':'a','rocket':'rk','anti_air':'aa','b_copter':'bc','fighter':'f','bomber':'b'}
KINDS={v:k for k,v in CODES.items()}
RESULTS={0:'1-0',1:'0-1',DRAW:'1/2-1/2',None:'*'}

def square(board,pos):
    x,y=board.xy(pos); label=''; x+=1
    while x:
        x,r=divmod(x-1,26); label=chr(97+r)+label
    return f'{label}{board.height-y}'

def position(board,text):
    m=re.fullmatch(r'([a-z]+)([1-9][0-9]*)',text)
    if not m: raise ValueError(f'Invalid square: {text}')
    x=0
    for ch in m[1]: x=x*26+ord(ch)-96
    y=board.height-int(m[2]); x-=1
    if not 0<=x<board.width or not 0<=y<board.height: raise ValueError(f'Square outside board: {text}')
    return board.pos(x,y)

def unit_name(state,uid):
    u=state.units[uid]
    return f'{CODES[u.kind]}:{square(state.board,u.pos)}'

def action_text(state,action):
    b=state.board
    if action.kind=='end': return '--'
    if action.kind=='build':return f'{square(b,action.destination)}+{CODES[action.build]}'
    u=state.units[action.actor]
    text=f'{CODES[u.kind]}: {square(b,u.pos)}'
    if action.destination!=u.pos:text+='-'+square(b,action.destination)
    if action.kind=='capture':text+='x'
    if action.kind=='attack':text+='x'+square(b,state.units[action.target].pos)
    return text

def parse_action(state,text):
    text=re.sub(r'\s+','',text.lower())
    if text=='--':return END
    build=re.fullmatch(r'([a-z]+[1-9][0-9]*)\+([a-z]+)',text)
    if build:
        if build[2] not in KINDS:raise ValueError(f'Unknown unit code: {build[2]}')
        return Action('build',destination=position(state.board,build[1]),build=KINDS[build[2]])
    move=re.fullmatch(r'(?:([a-z]+):)?([a-z]+[1-9][0-9]*)(?:-([a-z]+[1-9][0-9]*))?(x([a-z]+[1-9][0-9]*)?)?',text)
    if not move:raise ValueError(f'Invalid action: {text}')
    code,origin,destination,capture,target=move.groups()
    uid=state.occupancy.get(position(state.board,origin))
    if uid is None:raise ValueError(f'No unit at {origin}')
    if code and KINDS.get(code)!=state.units[uid].kind:raise ValueError(f'Unit code does not match {origin}')
    dest=position(state.board,destination or origin)
    if target:
        enemy=state.occupancy.get(position(state.board,target))
        if enemy is None:raise ValueError(f'No attack target at {target}')
        return Action('attack',uid,dest,enemy)
    return Action('capture' if capture else 'wait',uid,dest)

class GameRecord:
    def __init__(self,initial):
        self.initial=initial.clone()
        self.states=[initial.clone()]
        self.actions=[]

    def append(self,action):
        after=Rules().apply(self.states[-1],action)
        self.actions.append(action); self.states.append(after)

    def dumps(self):
        rules=Rules(); final=self.states[-1]
        open_turn=bool(self.actions and self.actions[-1]!=END and rules.outcome(final) is None)
        headers=['[AWPGN 1]','[InitialState '+json.dumps(to_data(self.initial),separators=(',',':'))+']',
                 '[OpenTurn '+json.dumps(open_turn)+']']
        lines=[]; moves=[]; turn=self.initial.turn+1
        for before,action in zip(self.states,self.actions):
            if action==END:
                lines.append(f'{turn}. '+', '.join(moves or ['--'])); moves=[]; turn+=1
            else:moves.append(action_text(before,action))
        if moves:lines.append(f'{turn}. '+', '.join(moves))
        return '\n'.join(headers+['']+lines+[RESULTS[rules.outcome(final)]])

    def turn_boundaries(self):
        return [0]+[i+1 for i,a in enumerate(self.actions) if a==END]

def loads(text,initial=None):
    headers={}; body=[]
    for line in text.splitlines():
        line=line.strip()
        if not line:continue
        if line.startswith('['):
            m=re.fullmatch(r'\[(\w+) (.+)\]',line)
            if not m or m[1] in headers:raise ValueError('Malformed or duplicate game header')
            try:headers[m[1]]=json.loads(m[2])
            except json.JSONDecodeError as e:raise ValueError('Invalid game header JSON') from e
        else:body.append(line)
    if headers.get('AWPGN',1)!=1:raise ValueError('Unsupported AWPGN version')
    if 'InitialState' in headers:initial=from_data(headers['InitialState'])
    if initial is None:raise ValueError('Game needs an InitialState header or a starting map')
    if type(headers.get('OpenTurn',False)) is not bool:raise ValueError('OpenTurn must be true or false')
    joined=' '.join(body)
    result=re.search(r'(1/2-1/2|1-0|0-1|\*)\s*$',joined)
    declared=result[1] if result else '*'
    moves=joined[:result.start()].strip() if result else joined.strip()
    matches=list(re.finditer(r'(\d+)\.\s*',moves))
    if moves and (not matches or moves[:matches[0].start()].strip()):raise ValueError('Expected numbered player turns')
    record=GameRecord(initial); rules=Rules()
    for index,m in enumerate(matches):
        state=record.states[-1]
        if int(m[1])!=state.turn+1:raise ValueError(f'Expected ply {state.turn+1}, got {m[1]}')
        segment=moves[m.end():matches[index+1].start() if index+1<len(matches) else len(moves)].strip()
        if not segment:raise ValueError('Empty ply; use -- for an empty turn')
        tokens=[x.strip() for x in segment.split(',')]
        for token in tokens:
            if token=='--':
                if len(tokens)!=1:raise ValueError('-- must be the only action in an empty turn')
                continue
            try:record.append(parse_action(record.states[-1],token))
            except (ValueError,KeyError) as e:raise ValueError(f'Ply {m[1]}, {token}: {e}') from e
        last=index==len(matches)-1
        if rules.outcome(record.states[-1]) is None and not (last and headers.get('OpenTurn',False)):
            record.append(END)
    actual=RESULTS[rules.outcome(record.states[-1])]
    if actual!=declared:raise ValueError(f'Result mismatch: notation says {declared}, rules give {actual}')
    return record
