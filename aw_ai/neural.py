"""Optional shared, variable-board policy/value network. No map identifiers.

Minibatches group boards by native dimensions and mask padded entity tokens,
avoiding padding artifacts at board edges. Legal actions are enumerated by Rules.
"""
from dataclasses import asdict, dataclass
from collections import OrderedDict
import math
import numpy as np
import torch
from torch import nn
from .model import DEFENSE, SPECS, PROFILE, outcome_value
from .production import production_features, KINDS
from .policy import Proposal, action_key

TERRAINS = tuple(DEFENSE)
COMMANDS = ('wait', 'attack', 'capture', 'end')
CHANNELS = len(TERRAINS)+2*len(KINDS)+8
ENTITY = len(KINDS)+8
FRONT = 3*len(KINDS)+4


@dataclass
class NetworkConfig:
    channels: int = 64
    blocks: int = 4
    hidden: int = 128
    layers: int = 3
    heads: int = 4


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(width,width,3,padding=1), nn.ReLU(),
                                 nn.Conv2d(width,width,3,padding=1))
    def forward(self, x):
        return torch.relu(x+self.net(x))


class Network(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or NetworkConfig()
        self.allowed_builds = KINDS
        self.search_settings = None
        c, h = self.config.channels, self.config.hidden
        self.spatial = nn.Sequential(nn.Conv2d(CHANNELS,c,3,padding=1), nn.ReLU(),
                                    *[Residual(c) for _ in range(self.config.blocks)])
        self.entity = nn.Linear(ENTITY+c,h)
        self.global_token = nn.Linear(c+10,h)
        layer = nn.TransformerEncoderLayer(h,self.config.heads,h*2,dropout=0.,batch_first=True)
        self.transformer = nn.TransformerEncoder(layer,self.config.layers,enable_nested_tensor=False)
        self.tile = nn.Linear(c,h)
        self.actor = nn.Linear(h,1)
        self.command = nn.Linear(h,len(COMMANDS))
        self.destination = nn.Linear(h,h)
        self.target = nn.Linear(h,h)
        self.value = nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1),nn.Tanh())
        self.production = nn.Sequential(nn.Linear(h*2+FRONT,h),nn.ReLU(),nn.Linear(h,len(KINDS)+1))

    def inputs(self, state, rules):
        """Prepare features on CPU; never launch a GPU operation per tile."""
        b, player = state.board, state.player
        grid = np.zeros((CHANNELS,b.height,b.width), dtype=np.float32)
        off = len(TERRAINS)+2*len(KINDS)
        for p,tile in enumerate(b.tiles):
            x,y=b.xy(p); grid[TERRAINS.index(tile),y,x]=1
            owner=state.owners.get(p)
            if owner is not None: grid[off+(owner!=player),y,x]=1
            grid[off+6,y,x]=x/max(1,b.width-1)
            grid[off+7,y,x]=y/max(1,b.height-1)
        ids, positions, features = [],[],[]
        for u in sorted(state.units.values(),key=lambda u:u.id):
            x,y=b.xy(u.pos); own=u.owner==player
            grid[len(TERRAINS)+(0 if own else len(KINDS))+KINDS.index(u.kind),y,x]=u.hp/100
            grid[off+2+(not own),y,x]=float(u.acted)
            f=[0.]*ENTITY; f[KINDS.index(u.kind)]=1
            f[len(KINDS):]=[float(own),u.hp/100,float(u.acted),u.spec.cost/22000,
                           u.spec.move/9,u.spec.maximum/5,x/max(1,b.width-1),y/max(1,b.height-1)]
            ids.append(u.id); positions.append(u.pos); features.append(f)
        for p in b.properties:
            x,y=b.xy(p)
            progress=state.captures.get(p)
            if progress:
                grid[off+4,y,x]=progress[1]/20
                grid[off+5,y,x]=1 if state.units[progress[0]].owner==player else -1
            f=[0.]*ENTITY
            f[len(KINDS):]=[float(state.owners.get(p)==player),0,0,0,0,0,
                           x/max(1,b.width-1),y/max(1,b.height-1)]
            ids.append(('site',p)); positions.append(p); features.append(f)
        economy=torch.tensor([state.funds[player]/20000,state.funds[1-player]/20000,
                              rules.income(state,player)/10000,rules.income(state,1-player)/10000,
                              float(state.income_capture_limit is not None),
                              rules.income(state,player)/(1000*state.income_capture_limit) if state.income_capture_limit else 0.,
                              rules.income(state,1-player)/(1000*state.income_capture_limit) if state.income_capture_limit else 0.,
                              float(state.turn_limit is not None),
                              max(0,state.turn_limit-state.turn)/100 if state.turn_limit else 0.,
                              state.turn_limit/100 if state.turn_limit else 0.],dtype=torch.float32)
        return torch.from_numpy(grid), ids, positions, torch.tensor(features, dtype=torch.float32).reshape(-1, ENTITY), economy

    def encode(self, state, rules):
        return self.encode_batch([state], rules)[0]

    def encode_batch(self, states, rules):
        """One CNN/attention pass per board shape; padding cannot affect real tokens."""
        device = next(self.parameters()).device
        groups = {}
        for i, state in enumerate(states):
            groups.setdefault((state.board.height, state.board.width), []).append(i)
        result = [None]*len(states)
        for indices in groups.values():
            inputs = [self.inputs(states[i], rules) for i in indices]
            grid = torch.stack([x[0] for x in inputs]).to(device)
            spatial = self.spatial(grid).flatten(2).transpose(1, 2)
            economy = torch.stack([x[4] for x in inputs]).to(device)
            global_tokens = self.global_token(torch.cat((spatial.mean(1), economy), 1))
            count = max(len(x[1]) for x in inputs)
            features = torch.zeros(len(indices), count, ENTITY)
            positions = torch.zeros(len(indices), count, dtype=torch.long)
            padding = torch.ones(len(indices), count+1, dtype=torch.bool)
            padding[:, 0] = False
            for row, (_, ids, pos, feats, _) in enumerate(inputs):
                n = len(ids)
                features[row, :n] = feats
                positions[row, :n] = torch.tensor(pos, dtype=torch.long)
                padding[row, 1:n+1] = False
            local = spatial.gather(1, positions.to(device)[:, :, None].expand(-1, -1, spatial.shape[2]))
            entities = self.entity(torch.cat((features.to(device), local), 2))
            tokens = torch.cat((global_tokens[:, None], entities), 1)
            tokens = self.transformer(tokens, src_key_padding_mask=padding.to(device))
            tiles = self.tile(spatial)
            for row, i in enumerate(indices):
                ids = inputs[row][1]
                result[i] = tokens[row, 0], dict(zip(ids, tokens[row, 1:len(ids)+1])), tiles[row]
        return result

    def action_logits(self, encoded, actions):
        global_token, entities, tiles=encoded
        actors=torch.stack([entities[a.actor]+global_token if a.actor else global_token for a in actions])
        commands=torch.tensor([COMMANDS.index(a.kind) for a in actions],device=actors.device)
        destinations=torch.tensor([max(0,a.destination) for a in actions],device=actors.device)
        active=torch.tensor([a.kind!='end' for a in actions],device=actors.device)
        targets=torch.stack([entities[a.target] if a.target else torch.zeros_like(global_token) for a in actions])
        score=self.command(actors).gather(1,commands[:,None]).squeeze(1)
        score=score+active*(self.actor(actors).squeeze(1)
              +(self.destination(actors)*tiles[destinations]).sum(1)/math.sqrt(actors.shape[1]))
        return score+(self.target(actors)*targets).sum(1)/math.sqrt(actors.shape[1])

    def decision_batch(self, examples, rules):
        """Flatten ragged actions into bulk gathers, avoiding a GPU kernel per action."""
        from .neural_training import unit_actions, build_choices
        encoded = self.encode_batch([e.state for e in examples], rules)
        device = next(self.parameters()).device
        choices, logits = [], [None]*len(examples)
        tokens, tiles = [], []
        actors, targets, destinations, commands, action_rows, globals_for_actions = [], [], [], [], [], []
        global_ids, build_rows, build_sites, fronts = [], [], [], []
        tile_offset = 0
        for i, (example, (global_token, entities, board_tiles)) in enumerate(zip(examples, encoded)):
            global_id = len(tokens)
            global_ids.append(global_id)
            tokens.append(global_token)
            ids = {key:len(tokens)+j for j,key in enumerate(entities)}
            tokens.extend(entities.values())
            tiles.append(board_tiles)
            if example.site is None:
                actions = unit_actions(example.state, rules)
                choices.append(actions)
                action_rows.append((i,len(actions)))
                for a in actions:
                    actors.append(ids[a.actor])
                    globals_for_actions.append(global_id)
                    targets.append(ids[a.target] if a.target else -1)
                    destinations.append(tile_offset+a.destination)
                    commands.append(COMMANDS.index(a.kind))
            else:
                choices.append(build_choices(example.state, rules, example.site, self.allowed_builds))
                fronts.append(production_features(example.state, rules, example.site))
                build_sites.append(ids[('site',example.site)])
                build_rows.append(i)
            tile_offset += len(board_tiles)
        tokens = torch.stack(tokens)
        globals = tokens[torch.tensor(global_ids,device=device)]
        if actors:
            actors = tokens[torch.tensor(actors,device=device)]+tokens[torch.tensor(globals_for_actions,device=device)]
            target_ids = torch.tensor(targets,device=device)
            target_tokens = tokens[target_ids.clamp_min(0)]*(target_ids >= 0)[:,None]
            destination_tiles = torch.cat(tiles)[torch.tensor(destinations,device=device)]
            scores = self.command(actors).gather(1,torch.tensor(commands,device=device)[:,None]).squeeze(1)
            scores = scores+self.actor(actors).squeeze(1)
            scores = scores+(self.destination(actors)*destination_tiles).sum(1)/math.sqrt(actors.shape[1])
            scores = scores+(self.target(actors)*target_tokens).sum(1)/math.sqrt(actors.shape[1])
            for (i, _), score in zip(action_rows, scores.split([n for _,n in action_rows])):
                logits[i] = score
        if build_rows:
            production = torch.cat((globals[build_rows],tokens[torch.tensor(build_sites,device=device)],
                                    torch.tensor(fronts,device=device)),1)
            scores = self.production(production)
            for i, score in zip(build_rows, scores):
                logits[i] = score[choices[i]]
        return choices, logits, self.value(globals).squeeze(1)

    def build_logits(self, encoded, state, rules, site):
        global_token, entities, _=encoded
        front=torch.tensor(production_features(state,rules,site),device=global_token.device)
        return self.production(torch.cat((global_token,entities[('site',site)],front)))

    def save(self,path,metadata=None):
        torch.save({'version':3,'profile':PROFILE,'kinds':KINDS,'terrains':TERRAINS,
                    'config':asdict(self.config),'weights':self.state_dict(),'metadata':metadata or {},
                    'allowed_builds':tuple(self.allowed_builds), 'search_settings':self.search_settings},path)

    @classmethod
    def load(cls,path):
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data['version'] not in (2,3) or (data['profile'],tuple(data['kinds']),tuple(data['terrains']))!=(PROFILE,KINDS,TERRAINS):
            raise ValueError('incompatible neural checkpoint schema/profile')
        model=cls(NetworkConfig(**data['config']))
        weights=dict(data['weights'])
        if data['version']==2:
            old=weights['global_token.weight']
            expected=(model.config.hidden,model.config.channels+7)
            if tuple(old.shape)!=expected: raise ValueError('invalid legacy global feature shape')
            weights['global_token.weight']=torch.cat((old,old.new_zeros((old.shape[0],3))),dim=1)
        model.load_state_dict(weights); model.eval()
        allowed = tuple(data.get('allowed_builds', KINDS))
        if not allowed or any(k not in KINDS for k in allowed):
            raise ValueError('invalid checkpoint production roster')
        model.allowed_builds = allowed
        settings = data.get('search_settings')
        if settings is not None:
            weight = settings.get('neural_value_weight')
            if settings.get('mode') != 'guided' or not isinstance(weight,(int,float)) or not math.isfinite(weight) or not 0 <= weight <= 1:
                raise ValueError('invalid checkpoint search settings')
            model.search_settings = dict(settings)
            if settings.get('planner','beam') not in ('bundle','beam','mcts'):
                raise ValueError('invalid checkpoint planner')
            if settings.get('bundle_version',1) not in (1,2):
                raise ValueError('invalid bundle version')
            for key,default,positive in (('proposal_temperature',.7,True),('policy_prior_weight',.1,False)):
                x=settings.get(key,default)
                if not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 or (positive and x==0):
                    raise ValueError('invalid policy search setting')
            if settings.get('planner') == 'mcts':
                from .bundle_mcts import validate
                validate(settings)
            if settings.get('planner') in ('bundle','mcts'):
                for key,default in (('target_temperature',.15),('heuristic_scale',20000.)):
                    x=settings.get(key,default)
                    if not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0:
                        raise ValueError('invalid bundle scale')
                for key in ('exploration_fraction','execution_exploration'):
                    x=settings.get(key,.1)
                    if not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<=x<=1:
                        raise ValueError('invalid exploration setting')
                x=settings.get('bundle_candidates',16)
                if not isinstance(x,int) or x<2:
                    raise ValueError('invalid bundle count')
        return model


def migrate_checkpoint(source, destination):
    """Preserve metadata/weights and write the current schema to a new file."""
    from pathlib import Path
    source, destination = Path(source), Path(destination)
    if destination.exists() or source.resolve() == destination.resolve():
        raise ValueError('migration output must be a new file; source is never overwritten')
    data = torch.load(source,map_location='cpu',weights_only=True)
    network = Network.load(source)
    metadata = dict(data.get('metadata') or {})
    metadata['clock_migration'] = {'source':str(source.resolve()),'source_version':data['version'],
                                 'clock_columns_initialized_to_zero':data['version']==2}
    destination.parent.mkdir(parents=True,exist_ok=True)
    network.save(destination,metadata)


class NeuralAgent:
    """Shared inference cache for both Planner.policy and Planner.value."""
    selects_actors=True
    def __init__(self, network, threads=1):
        torch.set_num_threads(threads)
        self.network=network.eval(); self.cache=OrderedDict()

    def encoded(self,state,rules):
        key=state.key()
        if key not in self.cache:
            with torch.no_grad(): self.cache[key]=self.network.encode(state,rules)
            if len(self.cache)>128: self.cache.popitem(last=False)
        return self.cache[key]

    def prefetch(self,states,rules):
        missing = {s.key():s for s in states if s.key() not in self.cache}
        if missing:
            with torch.no_grad(): encoded = self.network.encode_batch(list(missing.values()),rules)
            self.cache.update(zip(missing,encoded))
            while len(self.cache)>128: self.cache.popitem(last=False)

    def propose(self,state,context,rules,actors=None,limit=12,quiet=False):
        allowed=set(actors) if actors is not None else None
        actions=[a for a in rules.legal_actions(state) if a.kind not in ('end','build')
                 and (allowed is None or a.actor in allowed) and (not quiet or a.kind!='attack')]
        if not actions: return []
        with torch.no_grad(): scores=self.network.action_logits(self.encoded(state,rules),actions).log_softmax(0).tolist()
        proposals=[Proposal(a,s*1000,'neural legal-action policy') for a,s in zip(actions,scores)]
        return sorted(proposals,key=lambda p:(-p.score,action_key(p.action)))[:limit]

    def build_scores(self,state,rules,site):
        with torch.no_grad(): logits=self.network.build_logits(self.encoded(state,rules),state,rules,site).tolist()
        return {k: x*1000 if k == 'save' or k in self.network.allowed_builds else float('-inf')
                for k,x in zip((*KINDS,'save'),logits)}

    def evaluate(self,state,player,rules):
        outcome=rules.outcome(state)
        if outcome is not None: return 1e9*outcome_value(outcome,player)
        with torch.no_grad(): value=self.network.value(self.encoded(state,rules)[0]).item()
        return value*20000*(1 if player==state.player else -1)
