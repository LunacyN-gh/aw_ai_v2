"""Read-only game playback with asynchronous one-action model evaluation."""
import queue
import threading
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from pathlib import Path
from .notation import loads, GameRecord, action_text, square, CODES
from .model import DRAW, displayed_army_value
from .rules import Rules
from . import gui_art

class ReviewWindow:
    def __init__(self,parent,record=None):
        self.window=tk.Toplevel(parent); self.window.title('Game playback / model analysis')
        self.window.geometry('1140x800')
        self.record=record; self.index=0; self.version=0; self.closed=False
        self.model_path=None; self.jobs=queue.Queue(maxsize=1); self.results=queue.Queue()
        self.k=tk.StringVar(value='3'); self.status=tk.StringVar(value='Load a game file or paste notation below.')
        self.mode=tk.StringVar(value='Full search')
        self.seconds=tk.StringVar(value='1.0')
        self.neural_weight=tk.StringVar(value='100')
        self.explanation=tk.StringVar()
        self.model_label=tk.StringVar(value='No model loaded')
        bar=ttk.Frame(self.window,padding=8); bar.pack(fill='x')
        for title,callback in [('Load gameâ€¦',self.load),('Load modelâ€¦',self.load_model),('Use pasted text',self.use_text)]:
            ttk.Button(bar,text=title,command=callback).pack(side='left',padx=2)
        ttk.Label(bar,text='Top actions').pack(side='left',padx=(15,4))
        selector=ttk.Combobox(bar,textvariable=self.k,values=('1','2','3','4','5'),width=3,state='readonly')
        selector.pack(side='left'); selector.bind('<<ComboboxSelected>>',lambda e:self.request_analysis())
        options=ttk.Frame(self.window,padding=(8,0));options.pack(fill='x')
        mode=ttk.Combobox(options,textvariable=self.mode,values=('Full search','MCTS','One-action value'),state='readonly',width=20)
        mode.pack(side='left');mode.bind('<<ComboboxSelected>>',lambda e:self.request_analysis())
        ttk.Label(options,text='Search seconds').pack(side='left',padx=6)
        seconds=ttk.Entry(options,textvariable=self.seconds,width=6);seconds.pack(side='left')
        seconds.bind('<Return>',lambda e:self.request_analysis())
        ttk.Label(options,text='Search neural value %').pack(side='left',padx=6)
        weight=ttk.Combobox(options,textvariable=self.neural_weight,values=tuple(str(n) for n in range(101)),state='readonly',width=5)
        weight.pack(side='left');weight.bind('<<ComboboxSelected>>',lambda e:self.request_analysis())
        ttk.Button(options,text='Analyze',command=self.request_analysis).pack(side='left',padx=6)
        middle=ttk.Frame(self.window,padding=8); middle.pack(fill='both',expand=True)
        self.canvas=tk.Canvas(middle,background='#f4f3ee',highlightthickness=0)
        self.canvas.pack(side='left',fill='both',expand=True)
        self.canvas.bind('<Configure>',lambda e:self.draw())
        right=ttk.Frame(middle,width=350,padding=8);right.pack(side='right',fill='y')
        ttk.Label(right,textvariable=self.model_label,wraplength=320).pack(anchor='w')
        self.economy=tk.StringVar()
        ttk.Label(right,textvariable=self.economy,wraplength=330).pack(anchor='w',pady=8)
        ttk.Label(right,textvariable=self.explanation,wraplength=320).pack(anchor='w',pady=8)
        self.ranking=tk.Text(right,width=42,height=14,wrap='word',state='disabled');self.ranking.pack(fill='both',expand=True)
        nav=ttk.Frame(self.window);nav.pack()
        for title,command in [('|<','start'),('<<','prev_turn'),('<','prev'),('>','next'),('>>','next_turn'),('>|','end')]:
            ttk.Button(nav,text=title,width=7,command=lambda c=command:self.navigate(c)).pack(side='left',padx=3)
        ttk.Label(self.window,textvariable=self.status,wraplength=1080).pack(fill='x',padx=10,pady=8)
        self.text=tk.Text(self.window,height=7,wrap='none');self.text.pack(fill='x',padx=10,pady=8)
        if record:self.text.insert('1.0',record.dumps())
        self.window.protocol('WM_DELETE_WINDOW',self.close)
        threading.Thread(target=self.worker,daemon=True).start()
        self.poll_id=self.window.after(60,self.poll)
        self.show()

    def close(self):
        self.closed=True;self.version+=1
        try:self.jobs.get_nowait()
        except queue.Empty:pass
        self.jobs.put_nowait(None)
        self.window.after_cancel(self.poll_id)
        self.window.destroy()

    def set_ranking(self,text):
        self.ranking.configure(state='normal');self.ranking.delete('1.0','end')
        self.ranking.insert('1.0',text);self.ranking.configure(state='disabled')

    def load(self):
        path=filedialog.askopenfilename(parent=self.window,filetypes=[('AW game','*.awpgn *.pgn *.txt'),('All files','*.*')])
        if not path:return
        try:
            text=Path(path).read_text(encoding='utf-8-sig');record=loads(text)
        except (ValueError,KeyError,TypeError,OSError) as e:
            messagebox.showerror('Cannot load game',str(e),parent=self.window);return
        self.record=record;self.index=0;self.text.delete('1.0','end');self.text.insert('1.0',text);self.show()

    def use_text(self):
        try:
            # A headerless fragment may use the currently loaded initial map.
            record=loads(self.text.get('1.0','end'),self.record.initial if self.record else None)
        except (ValueError,KeyError,TypeError) as e:
            messagebox.showerror('Cannot parse game',str(e),parent=self.window);return
        self.record=record;self.index=0;self.show()

    def load_model(self):
        path=filedialog.askopenfilename(parent=self.window,filetypes=[('Neural checkpoint','*.pt')])
        if path:
            self.model_path=path;self.model_label.set(Path(path).name);self.request_analysis()

    def navigate(self,command):
        if not self.record:return
        size=len(self.record.actions);boundaries=self.record.turn_boundaries()
        if command=='start':self.index=0
        elif command=='end':self.index=size
        elif command=='prev':self.index=max(0,self.index-1)
        elif command=='next':self.index=min(size,self.index+1)
        elif command=='prev_turn':self.index=max([b for b in boundaries if b<self.index],default=0)
        elif command=='next_turn':self.index=min([b for b in boundaries if b>self.index],default=size)
        self.show()

    def show(self):
        self.draw()
        if self.record:
            state=self.record.states[self.index]
            rules=Rules()
            self.economy.set(f'Army value: Blue ${displayed_army_value(state,0):,} Â· Red ${displayed_army_value(state,1):,}\n'
                             f'Blue funds ${state.funds[0]:,} Â· income ${rules.income(state,0):,}\n'
                             f'Red funds ${state.funds[1]:,} Â· income ${rules.income(state,1):,}\n'
                             +(f'{max(0,state.turn_limit-state.turn)} player turns remaining' if state.turn_limit else 'No turn deadline'))
            last='Start' if not self.index else action_text(self.record.states[self.index-1],self.record.actions[self.index-1])
            winner=Rules().outcome(state)
            ending=' Â· Draw' if winner==DRAW else f' Â· {"Blue" if winner==0 else "Red"} wins' if winner is not None else ''
            self.status.set(f'Action {self.index}/{len(self.record.actions)} Â· Ply {state.turn+1} Â· {"Blue" if state.player==0 else "Red"} Â· {last}{ending}')
        self.request_analysis()

    def draw(self):
        from .gui import COLORS,SIDES
        c=self.canvas;c.delete('all')
        if not self.record:return
        state=self.record.states[self.index];b=state.board
        cell=max(8,min((max(240,c.winfo_width())-36)/b.width,(max(200,c.winfo_height())-36)/b.height,70));offset=24
        for p,t in enumerate(b.tiles):
            x,y=b.xy(p);left,top=offset+x*cell,offset+y*cell
            c.create_rectangle(left,top,left+cell,top+cell,fill=COLORS[t],outline='#afbaaa')
            links=[(b.xy(q)[0]-x,b.xy(q)[1]-y) for q in b.neighbors[p] if b.tiles[q] in ('road','bridge','factory','city','hq','airport')]
            gui_art.terrain(c,t,left,top,cell,state.owners.get(p),links)
        for u in state.units.values():
            x,y=b.xy(u.pos);left,top=offset+x*cell,offset+y*cell
            gui_art.unit(c,u.kind,left,top,cell,u.owner,u.acted)
            c.create_rectangle(left+cell*.04,top+cell*.73,left+cell*.97,top+cell*.97,fill=SIDES[u.owner],outline='')
            c.create_text(left+cell*.5,top+cell*.85,text=f'{CODES[u.kind]}  {u.displayed_hp}',fill='white',font=('Segoe UI',max(6,int(cell/6)),'bold'))
        for p,(_,remaining) in state.captures.items():
            x,y=b.xy(p);c.create_text(offset+(x+1)*cell-2,offset+y*cell+2,text=remaining,anchor='ne',font=('Segoe UI',max(7,int(cell/4)),'bold'))
        for x in range(b.width):c.create_text(offset+(x+.5)*cell,12,text=square(b,b.pos(x,0)).rstrip('0123456789'))
        for y in range(b.height):c.create_text(12,offset+(y+.5)*cell,text=b.height-y)

    def request_analysis(self):
        self.version+=1
        self.explanation.set('Full AI search: first actions ranked by their best sampled turn continuation. Fewer than k may be found. Scores are not win probabilities.' if self.mode.get()!='One-action value' else 'Pure neural one-action successor value; search weight does not apply. Positive favors the player to move. Scores are not win probabilities.')
        if not self.record or not self.model_path:
            self.set_ranking('Load a model to evaluate legal actions.');return
        try:
            from .planner import Config
            seconds=float(self.seconds.get());Config(seconds=seconds)
        except ValueError:
            self.set_ranking('Enter a finite, nonnegative search time.');return
        self.set_ranking('Evaluatingâ€¦')
        job=(self.version,self.model_path,self.record.states[self.index].clone(),int(self.k.get()),self.mode.get(),seconds,float(self.neural_weight.get())/100)
        try:self.jobs.get_nowait()
        except queue.Empty:pass
        self.jobs.put_nowait(job)

    def worker(self):
        network=None;loaded=None
        while True:
            job=self.jobs.get()
            if job is None:return
            version,path,state,k,mode,seconds,weight=job
            try:
                import torch
                from .neural import Network
                from .review import rank_actions,rank_search_actions
                torch.set_num_threads(1)
                if path!=loaded:network=Network.load(path);loaded=path
                from .gui_model import set_value_weight
                set_value_weight(network,weight)
                if mode in ('Full search','MCTS'):
                    rows=rank_search_actions(network,state,k,seconds,backend='mcts' if mode=='MCTS' else None)
                    lines=[]
                    for i,r in enumerate(rows):
                        score=f'Exact terminal value {r["terminal_value"]:+.0f}' if r['terminal'] else f'Search score {r["score"]:+.3f}'
                        if r.get('visits') is not None:score=f'MCTS Q {r["score"]:+.3f} Â· {r["visits"]} visits'
                        coverage='tree evaluated' if r.get('visits') is not None else 'reply checked' if r['verified'] else 'fallback / no common reply coverage'
                        if not r['complete']:coverage+='; partial plan'
                        lines.append(f'{i+1}. {r["text"]}'+(' Â· recommended' if r['recommended'] else '')+f'\n{score} Â· {coverage}\nPlan: {r["plan"]}')
                    text='\n\n'.join(lines) or 'Terminal position or no candidate plans.'
                else:
                    rows=rank_actions(network,state,k)
                    text='\n\n'.join(f'{i+1}. {r["text"]}\nValue {r["value"]:+.3f}'+(' Â· exact terminal result' if r['terminal'] else '') for i,r in enumerate(rows)) or 'Terminal position: no legal actions.'
                self.results.put((version,text))
            except Exception as e:self.results.put((version,f'Analysis failed: {e}'))

    def poll(self):
        try:
            while True:
                version,text=self.results.get_nowait()
                if version==self.version:self.set_ranking(text)
        except queue.Empty:pass
        if not self.closed:self.poll_id=self.window.after(60,self.poll)
