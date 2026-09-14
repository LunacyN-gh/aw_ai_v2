from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from .notation import GameRecord, unit_name, action_text as notation_action
from .model import displayed_army_value
from . import gui_art
from .animation import movement_path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .model import Action, Board, END, PROPERTIES, SPECS, Unit, DRAW
from .planner import Config, Planner
from .rules import Rules
from .scenarios import action_text, league_scale, load, save
from .tree import SampledPlanner


COLORS = {"plain": "#c4d8ad", "road": "#d5cdbb", "bridge": "#d5cdbb",
          "forest": "#73956b", "mountain": "#9fa59e", "water": "#7baecb",
          "river": "#7baecb", "shoal": "#ddd6b5", "city": "#d4d5cb",
          "factory": "#b7bcc1", "airport": "#b9c8cc", "tower": "#d2c3a6", "hq": "#c8b9a8"}
SYMBOL = {"infantry": "I", "mech": "M", "recon": "R", "tank": "T", "md_tank": "MT",
          "artillery": "A", "rocket": "RK", "anti_air": "AA", "b_copter": "BC",
          "fighter": "F", "bomber": "B"}
PROPERTY_SYMBOL = {"city": "C", "factory": "B", "airport": "A", "tower": "▲", "hq": "HQ"}
SIDES = ("#285c9d", "#b54145")


class App:
    def __init__(self, root, turn_limit=None):
        self.root = root
        root.title("Advance Wars AI v2 — deterministic research profile")
        root.geometry("1240x880")
        root.minsize(900, 640)
        self.rules = Rules()
        self.state = league_scale()
        self.state.turn_limit = turn_limit
        self.initial = self.state.clone()
        self.record = GameRecord(self.state)
        self.game_text = self.record.dumps()
        self.record_printed = False
        self.selected = None
        self.destination = None
        self.analysis = None
        self.busy = False
        self.auto = False
        self.results = queue.Queue()
        self.action_menu = tk.Menu(root, tearoff=False)
        self.planners = {}
        self.neural_agent = None
        self.heuristic_backend = "ordered"
        self.neural_weight = tk.StringVar(value="100")
        self.applied_neural_weight = "100"
        self.generation = 0
        self.cell = 32
        self.offset = 25
        self.animation_speed = tk.StringVar(value="1×")
        self.motion = {}
        self.flashing = set()
        self.animation_token = 0
        self.seconds = tk.StringVar(value="1.0")
        self.search = tk.StringVar(value="beam")
        self.size = tk.StringVar(value="20")
        self.count = tk.StringVar(value="12")
        self.human_side = 0
        self.human_label = tk.StringVar(value="Human Blue / automatic Red")
        self.human = tk.BooleanVar(value=True)
        self.editing = tk.BooleanVar(value=False)
        self.paint_tile = tk.StringVar(value="plain")
        self.paint_owner = tk.StringVar(value="neutral")
        self.paint_unit = tk.StringVar(value="none")
        self.status = tk.StringVar(value="Blue to move. Select a unit, then a destination or target.")
        self.summary = tk.StringVar()
        self._layout()
        self.draw()
        root.after(60, self.poll)

    def _layout(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        left = ttk.Frame(outer)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, background="#f4f3ee", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.draw())
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Button-3>", self.right_click)
        ttk.Label(left, textvariable=self.status, wraplength=750).pack(fill="x", pady=(6, 0))
        side = ttk.Frame(outer, width=330, padding=(12, 0, 0, 0))
        side.pack(side="right", fill="y")
        ttk.Label(side, text="ADVANCE WARS / V2", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(side, text="Full information · no luck · experimental rules", wraplength=310).pack(anchor="w", pady=(0, 8))
        ttk.Label(side, textvariable=self.summary, font=("Consolas", 10), justify="left").pack(anchor="w", pady=6)
        files = ttk.Frame(side)
        files.pack(fill="x")
        for label, command in (("Load…", self.load), ("Save…", self.save), ("Restart", self.restart)):
            ttk.Button(files, text=label, command=command, width=9).pack(side="left", padx=2)
        games = ttk.Frame(side)
        games.pack(fill="x", pady=3)
        ttk.Button(games,text="Save game…",command=self.save_game).pack(side="left",padx=2)
        ttk.Button(games,text="Review game…",command=self.review_game).pack(side="left",padx=2)
        settings = ttk.Frame(side)
        settings.pack(fill="x", pady=8)
        ttk.Label(settings, text="Seconds").pack(side="left")
        ttk.Entry(settings, textvariable=self.seconds, width=6).pack(side="left", padx=5)
        ttk.Combobox(settings, textvariable=self.search, values=("beam", "tree", "mcts"), state="readonly", width=8).pack(side="left")
        playback = ttk.Frame(side)
        playback.pack(fill="x", pady=(0,6))
        ttk.Label(playback, text="Animation speed").pack(side="left")
        ttk.Combobox(playback, textvariable=self.animation_speed, values=("0.25×","0.5×","1×","2×","4×","8×","Instant"), state="readonly", width=8).pack(side="left", padx=5)
        models = ttk.Frame(side)
        models.pack(fill="x")
        ttk.Button(models, text="Load neural model…", command=self.load_neural).pack(side="left")
        teachers = ttk.Frame(side)
        teachers.pack(fill="x", pady=3)
        ttk.Button(teachers, text="Use heuristic (ordered)", command=self.clear_neural).pack(side="left")
        ttk.Button(teachers, text="Old heuristic", command=lambda: self.clear_neural("old")).pack(side="left", padx=3)
        weights = ttk.Frame(side)
        weights.pack(fill="x", pady=4)
        ttk.Label(weights, text="Neural value %").pack(side="left")
        weight = ttk.Combobox(weights, textvariable=self.neural_weight,
                              values=tuple(str(n) for n in range(101)), state="readonly", width=5)
        weight.pack(side="left", padx=5)
        weight.bind('<<ComboboxSelected>>', lambda e: self.change_neural_weight())
        for label, command in (("Analyze position", lambda: self.start(False)),
                               ("Play recommended turn", self.play_recommendation),
                               ("AI vs AI", self.start_auto), ("Pause after turn", self.pause)):
            ttk.Button(side, text=label, command=command).pack(fill="x", pady=2)
        ttk.Checkbutton(side, textvariable=self.human_label, variable=self.human).pack(anchor="w", pady=5)
        ttk.Button(side, text="Flip sides", command=self.flip_sides).pack(fill="x", pady=2)
        moves = ttk.Frame(side)
        moves.pack(fill="x", pady=5)
        ttk.Button(moves, text="Wait / move", command=lambda: self.commit("wait")).pack(side="left")
        ttk.Button(moves, text="Capture", command=lambda: self.commit("capture")).pack(side="left", padx=3)
        ttk.Button(moves, text="End turn", command=self.end_turn).pack(side="left")
        self.build_kind = tk.StringVar(value="infantry")
        build = ttk.Frame(side)
        build.pack(fill="x")
        self.build_menu = ttk.Combobox(build, textvariable=self.build_kind, values=tuple(SPECS), state="readonly", width=17)
        self.build_menu.pack(side="left")
        ttk.Button(build, text="Build", command=self.build).pack(side="left", padx=3)
        ttk.Label(side, text="Select an empty owned base or airport to build.", wraplength=310).pack(anchor="w", pady=4)
        ttk.Separator(side).pack(fill="x", pady=8)
        ttk.Label(side, text="Alternatives / analysis").pack(anchor="w")
        self.alternatives = tk.Listbox(side, height=4, width=42, exportselection=False)
        self.alternatives.pack(fill="x")
        self.alternatives.bind("<<ListboxSelect>>", self.show_alternative)
        self.log = tk.Text(side, width=42, height=11, wrap="word", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, pady=5)
        ttk.Separator(side).pack(fill="x", pady=5)
        generate = ttk.Frame(side)
        generate.pack(fill="x")
        ttk.Label(generate, text="Size").pack(side="left")
        ttk.Entry(generate, textvariable=self.size, width=4).pack(side="left", padx=3)
        ttk.Label(generate, text="Units/side").pack(side="left")
        ttk.Entry(generate, textvariable=self.count, width=4).pack(side="left", padx=3)
        ttk.Button(generate, text="Generate", command=self.generate).pack(side="left")
        ttk.Checkbutton(side, text="Edit map (left paint / right erase unit)", variable=self.editing).pack(anchor="w", pady=5)
        edit = ttk.Frame(side)
        edit.pack(fill="x")
        for variable, values, width in ((self.paint_tile, tuple(COLORS), 9),
                                         (self.paint_owner, ("neutral", "blue", "red"), 7),
                                         (self.paint_unit, ("none",)+tuple(SPECS), 12)):
            ttk.Combobox(edit, textvariable=variable, values=values, state="readonly", width=width).pack(side="left")

    def write(self, text):
        self.log.insert("end", text+"\n")
        self.log.see("end")

    def game_over_message(self):
        winner = self.rules.outcome(self.state)
        if winner is None:
            return None
        if winner == DRAW:
            return "Draw — equal income at the player-turn deadline. Restart or load another position."
        name = "Blue" if winner == 0 else "Red"
        limit = self.state.income_capture_limit
        count = self.rules.income(self.state, winner)//1000
        reason = f" — income capture limit ({count}/{limit} properties)" if limit and count >= limit else ""
        if self.state.turn_limit is not None and self.state.turn >= self.state.turn_limit:
            reason = " — higher income at the player-turn deadline"
        return f"{name} wins{reason}. The game is over; restart or load another position."

    def draw(self):
        roster = self.state.allowed_builds or tuple(SPECS)
        self.build_menu.configure(values=roster)
        if self.build_kind.get() not in roster:
            self.build_kind.set(roster[0])
        canvas, board = self.canvas, self.state.board
        canvas.delete("all")
        self.cell = max(10, min((max(300, canvas.winfo_width())-45)//board.width,
                               (max(300, canvas.winfo_height())-45)//board.height, 65))
        cell, offset = self.cell, self.offset
        reach = self.rules.reachable(self.state, self.selected) if self.selected in self.state.units else {}
        for p, tile in enumerate(board.tiles):
            x, y = board.xy(p)
            x0, y0 = offset+x*cell, offset+y*cell
            canvas.create_rectangle(x0, y0, x0+cell, y0+cell, fill=COLORS[tile], outline="#afbaaa")
            links=[]
            if tile in ("road","bridge"):
                links=[(board.xy(q)[0]-x,board.xy(q)[1]-y) for q in board.neighbors[p]
                       if board.tiles[q] in ("road","bridge") or board.tiles[q] in PROPERTIES]
            gui_art.terrain(canvas,tile,x0,y0,cell,self.state.owners.get(p),links)
            if p in reach:
                canvas.create_rectangle(x0+2, y0+2, x0+cell-2, y0+cell-2, outline="#e9cf45", width=2)
            if p == self.destination:
                canvas.create_rectangle(x0+1, y0+1, x0+cell-1, y0+cell-1, outline="#fff7d6", width=3)
            owner = self.state.owners.get(p)
            if tile in PROPERTIES:
                color = SIDES[owner] if owner is not None else "#636663"
                canvas.create_text(x0+cell*.5, y0+cell*.22, text=PROPERTY_SYMBOL[tile], fill=color,
                                   font=("Segoe UI", max(6, cell//4), "bold"))
            if p in self.state.captures:
                canvas.create_text(x0+cell-2, y0+2, text=str(self.state.captures[p][1]), anchor="ne",
                                   fill="#383222", font=("Segoe UI", max(7, cell//4), "bold"))
        # Units are a separate layer so moving sprites cover every terrain tile.
        for unit in sorted(self.state.units.values(), key=lambda u:u.id in self.motion):
            x,y=self.motion.get(unit.id,board.xy(unit.pos))
            x0,y0=offset+x*cell,offset+y*cell
            gui_art.unit(canvas,unit.kind,x0,y0,cell,unit.owner,unit.acted,unit.id in self.flashing)
            if unit.id == self.selected:
                canvas.create_rectangle(x0+2,y0+2,x0+cell-2,y0+cell-2,outline="#fff9ce",width=2)
            canvas.create_rectangle(x0+cell*.05,y0+cell*.73,x0+cell*.31,y0+cell*.96,fill=SIDES[unit.owner],outline="")
            canvas.create_text(x0+cell*.18,y0+cell*.84,text=SYMBOL[unit.kind],fill="white",font=("Segoe UI",max(6,cell//7),"bold"))
            canvas.create_rectangle(x0+cell*.73,y0+cell*.73,x0+cell*.95,y0+cell*.96,fill="#263e47",outline="")
            canvas.create_text(x0+cell*.83,y0+cell*.84,text=unit.displayed_hp,fill="white",font=("Segoe UI",max(7,cell//5),"bold"))
        for x in range(board.width):
            label = chr(97+x) if x < 26 else str(x+1)
            canvas.create_text(offset+(x+.5)*cell, offset-10, text=label, fill="#555")
        for y in range(board.height):
            canvas.create_text(offset-12, offset+(y+.5)*cell, text=str(board.height-y), fill="#555")
        self.summary.set(f"Turn {self.state.turn+1} · {'Blue' if self.state.player == 0 else 'Red'}\n"
                         f"Army value  Blue ${displayed_army_value(self.state,0):,} · Red ${displayed_army_value(self.state,1):,}\n"
                         f"Blue  ${self.state.funds[0]:,}  +${self.rules.income(self.state, 0):,}  {len(self.state.army(0))} units\n"
                         f"Red   ${self.state.funds[1]:,}  +${self.rules.income(self.state, 1):,}  {len(self.state.army(1))} units"
                         + (f"\nCapture victory: {self.state.income_capture_limit} income properties"
                            if self.state.income_capture_limit else "")
                         + (f"\nDeadline: {self.state.turn_limit} player turns; {max(0,self.state.turn_limit-self.state.turn)} remaining"
                            if self.state.turn_limit is not None else ""))

        message = self.game_over_message()
        if message:
            self.summary.set(self.summary.get()+"\nGAME OVER")
            self.status.set(message)

    def tile_at(self, event):
        x, y = (event.x-self.offset)//self.cell, (event.y-self.offset)//self.cell
        return self.state.board.pos(x, y) if 0 <= x < self.state.board.width and 0 <= y < self.state.board.height else None

    def click(self, event):
        p = self.tile_at(event)
        if p is None:
            return
        if self.busy:
            self.status.set("Wait for the AI turn to finish.")
            return
        if self.editing.get():
            self.paint(p)
            return
        message = self.game_over_message()
        if message:
            self.status.set(message)
            return
        uid = self.state.occupancy.get(p)
        if uid and self.state.units[uid].owner == self.state.player:
            self.destination = None
            if self.state.units[uid].acted:
                self.selected = None
                self.status.set("This unit has already acted or was built this turn. It becomes available next turn.")
            else:
                self.selected = uid
                self.status.set("Choose a highlighted destination, then Wait / move; or select an enemy to attack.")
        elif uid:
            if self.selected:
                attacks = [a for a in self.rules.attack_actions(self.state, self.selected) if a.target == uid]
                preferred = [a for a in attacks if a.destination == self.destination]
                if attacks:
                    action = (preferred or sorted(attacks, key=lambda a: self.state.board.distance(
                        self.state.units[self.selected].pos, a.destination)))[0]
                    self.play_action(action)
                    return
            self.status.set("No legal attack on this unit from the selected unit.")
        elif self.selected:
            if p in self.rules.reachable(self.state, self.selected):
                self.destination = p
                self.status.set("Choose Move / wait, Capture, or Attack from the destination menu.")
                self.draw()
                self.show_destination_menu(event)
                return
            else:
                self.status.set("That destination is not reachable by this unit.")
        else:
            self.destination = p
            self.status.set("Select an available unit to move, or an empty owned base/airport to build.")
        self.draw()

    def show_destination_menu(self, event):
        """Expose the action confirmation at the square the player selected."""
        uid, pos = self.selected, self.destination
        if uid not in self.state.units or pos is None:
            return
        self.action_menu.delete(0, "end")
        actions = [(Action("wait", uid, pos), "Move / wait here"),
                   (Action("capture", uid, pos), "Capture here")]
        actions.extend((a, f"Attack {unit_name(self.state,a.target)}")
                       for a in self.rules.attack_actions(self.state, uid) if a.destination == pos)
        for action, label in actions:
            if self.rules.is_legal(self.state, action):
                self.action_menu.add_command(label=label, command=lambda a=action: self.play_action(a))
        self.action_menu.add_separator()
        self.action_menu.add_command(label="Cancel / choose another square", command=self.action_menu.unpost)
        x = getattr(event, "x_root", self.canvas.winfo_rootx()+event.x)
        y = getattr(event, "y_root", self.canvas.winfo_rooty()+event.y)
        try:
            self.action_menu.tk_popup(x, y)
        finally:
            self.action_menu.grab_release()

    def right_click(self, event):
        p = self.tile_at(event)
        if p is None or self.busy:
            return
        if self.editing.get():
            uid = self.state.occupancy.get(p)
            if uid:
                self.state.remove(uid)
                self.state.winner = None
                self.invalidate()
                self.draw()
        elif self.selected:
            self.destination = p
            self.commit("capture")

    def commit(self, kind):
        if not self.busy and self.selected in self.state.units:
            destination = self.destination if self.destination is not None else self.state.units[self.selected].pos
            self.play_action(Action(kind, self.selected, destination))

    def build(self):
        if not self.busy and self.destination is not None:
            self.play_action(Action("build", destination=self.destination, build=self.build_kind.get()))

    def end_turn(self):
        if not self.busy:
            self.apply_action(END)
            if self.human.get() and self.state.player != self.human_side and self.rules.outcome(self.state) is None:
                self.start(True)

    def flip_sides(self):
        """Switch human control, without relabeling or advancing the game."""
        if self.busy:
            return
        if self.rules.outcome(self.state) is not None:
            self.status.set("Restart or load a position before switching sides.")
            return
        try:
            Config(seconds=float(self.seconds.get()))
        except ValueError:
            self.status.set("Enter a nonnegative thinking time in seconds.")
            return
        self.human_side = 1-self.human_side
        human_name = "Blue" if self.human_side == 0 else "Red"
        ai_name = "Red" if self.human_side == 0 else "Blue"
        self.human_label.set(f"Human {human_name} / automatic {ai_name}")
        self.auto = False
        self.human.set(True)
        self.editing.set(False)
        self.invalidate()
        self.alternatives.delete(0, "end")
        self.write(f"Human now controls {human_name}; AI controls {ai_name}. Position and turn unchanged.")
        self.draw()
        if self.state.player != self.human_side:
            self.start(True)
        else:
            self.status.set(f"Human {human_name} to move.")

    def apply_action(self, action):
        self.action_menu.unpost()
        message = self.game_over_message()
        if message:
            self.status.set(message)
            return False
        try:
            before = self.state
            after = self.rules.apply(before, action)
            record = self.current_record()
            record.append(action)
            self.state = after
            self.game_text = record.dumps()
            self.write("End turn" if action.kind=="end" else notation_action(before,action))
            self.selected = self.destination = None
            self.analysis = None
            self.draw()
            winner = self.rules.outcome(self.state)
            if winner is not None:
                self.auto = False
                self.status.set(self.game_over_message())
                if not self.record_printed:
                    self.write("\nGame notation:\n"+self.game_text)
                    self.record_printed=True
            else:
                self.status.set("Select a unit, destination, then an enemy or a command.")
            return True
        except ValueError as error:
            self.status.set(str(error))
            return False

    def load_neural(self):
        if self.busy: return
        path = filedialog.askopenfilename(filetypes=[("Neural checkpoint", "*.pt")])
        if not path: return
        try:
            from .neural import Network
            from .guidance import deployment_agent
            from .gui_model import set_value_weight
            network = Network.load(path)
            set_value_weight(network, float(self.neural_weight.get()) / 100)
            self.neural_agent = deployment_agent(network)
            self.applied_neural_weight = self.neural_weight.get()
            self.planners.clear()
            self.analysis = None
            settings = self.neural_agent.network.search_settings
            mode = f"Guided model (position value: {settings['neural_value_weight']:.0%} neural / {1-settings['neural_value_weight']:.0%} heuristic)" if settings else "Neural policy/value"
            self.status.set(f"{mode} loaded: {path}")
        except Exception as error:
            messagebox.showerror("Could not load model", str(error))

    def change_neural_weight(self):
        if self.busy:
            self.neural_weight.set(self.applied_neural_weight)
            self.status.set("Wait until the current AI turn finishes to change the neural value weight.")
            return
        if self.neural_agent is not None:
            from .gui_model import set_value_weight
            from .guidance import deployment_agent
            network = self.neural_agent.network
            set_value_weight(network, float(self.neural_weight.get()) / 100)
            self.neural_agent = deployment_agent(network)
        self.applied_neural_weight = self.neural_weight.get()
        self.planners.clear()
        self.analysis = None
        self.alternatives.delete(0, 'end')
        self.status.set(f"Position value: {self.neural_weight.get()}% neural; applies to neural search.")

    def clear_neural(self, backend="ordered"):
        if self.busy: return
        self.neural_agent = None
        self.heuristic_backend = backend
        self.search.set("beam")
        self.planners.clear()
        self.analysis = None
        self.alternatives.delete(0, "end")
        self.status.set("Using ordered heuristic teacher." if backend=="ordered" else "Using old heuristic teacher.")

    def start(self, execute):
        if self.busy or self.rules.outcome(self.state) is not None:
            return
        if self.search.get()=='mcts' and self.neural_agent is None:
            self.status.set('Load a neural model before selecting MCTS.')
            return
        try:
            config = Config(seconds=float(self.seconds.get()))
        except ValueError:
            self.status.set("Enter a nonnegative thinking time in seconds.")
            return
        self.action_menu.unpost()
        self.editing.set(False)
        self.busy = True
        self.status.set("Thinking… The board remains interactive for viewing.")
        self.selected = self.destination = None
        snapshot = self.state.clone()
        generation = self.generation
        kind = self.search.get()
        key = (snapshot.player, kind, config)
        if key not in self.planners:
            backend=SampledPlanner if kind=='tree' else Planner
            if kind=='beam' and self.neural_agent is None and self.heuristic_backend=='ordered':
                from .order_search import OrderPlanner
                backend=OrderPlanner
            if kind=='mcts':
                from .bundle_mcts import MCTSPlanner
                backend=MCTSPlanner
            self.planners[key] = backend(
                config=config, policy=self.neural_agent, value=self.neural_agent)
        planner = self.planners[key]

        def work():
            try:
                result = planner.analyze(snapshot)
                self.results.put((generation, result, execute, None))
            except Exception as error:
                self.results.put((generation, None, execute, str(error)))
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            generation, result, execute, error = self.results.get_nowait()
            if generation == self.generation:
                self.busy = False
                if error:
                    self.auto = False
                    self.status.set(error)
                else:
                    self.analysis = result
                    self.alternatives.delete(0, "end")
                    for i, c in enumerate(result.alternatives):
                        coverage = "reply searched" if c.verified else "estimate only"
                        if not c.complete:
                            coverage = "partial turn / budget limit"
                        self.alternatives.insert("end", f"{i+1}. {c.score:,.1f} · {coverage}")
                    self.write(f"Analysis: {result.metrics.elapsed:.3f}s; {result.metrics.counts['nodes']} nodes; "
                               f"{len(result.regions)} interacting regions.")
                    for uid, objective in result.objectives.items():
                        self.write(f"  {uid}: {objective.reason}")
                    self.status.set(f"Analysis ready. Score {result.score:,.1f}; not a win probability.")
                    if execute:
                        self.animate(result)
        except queue.Empty:
            pass
        self.root.after(60, self.poll)

    def speed_multiplier(self):
        value=self.animation_speed.get()
        return 0. if value=="Instant" else float(value.rstrip("×x"))

    def play_action(self, action, done=None):
        """Animate the old state, then commit once. No game changes per frame."""
        if done is None and self.busy:
            return
        self.action_menu.unpost()
        if not self.rules.is_legal(self.state,action):
            self.status.set(self.game_over_message() or "That action is not legal.")
            if done: done(False)
            return
        token=self.animation_token
        generation=self.generation
        self.busy=True
        self.selected=self.destination=None
        speed=self.speed_multiplier()
        path=[]
        if action.actor:
            path=movement_path(self.state,action.actor,action.destination,self.rules)
        def alive():
            return token==self.animation_token and generation==self.generation
        def commit():
            if not alive(): return
            self.motion.clear(); self.flashing.clear()
            ok=self.apply_action(action)
            if done: done(ok)
            else: self.busy=False
        def combat():
            if not alive(): return
            if action.kind!='attack' or not speed:
                commit(); return
            started=perf_counter()
            self.status.set("Combat…")
            def flash():
                if not alive(): return
                elapsed=(perf_counter()-started)*speed
                if elapsed>=.5:
                    commit(); return
                self.flashing={action.actor,action.target} if int(elapsed/.08)%2==0 else set()
                self.draw(); self.root.after(25,flash)
            flash()
        if not speed or len(path)<2:
            combat(); return
        started=perf_counter()
        self.status.set("Moving…")
        def frame():
            if not alive(): return
            distance=(perf_counter()-started)*speed/.18
            if distance>=len(path)-1:
                self.motion[action.actor]=self.state.board.xy(path[-1])
                self.draw(); combat(); return
            index=int(distance); fraction=distance-index
            x,y=self.state.board.xy(path[index]); nx,ny=self.state.board.xy(path[index+1])
            self.motion[action.actor]=(x+(nx-x)*fraction,y+(ny-y)*fraction)
            self.draw(); self.root.after(25,frame)
        frame()

    def animate(self, result):
        if result.root_key != self.state.key():
            self.status.set("Position changed; analyze again.")
            return
        self.busy=True
        generation=self.generation
        token=self.animation_token
        actions=iter(result.actions)
        def step(ok=True):
            if generation!=self.generation or token!=self.animation_token: return
            if not ok:
                self.busy=False; self.auto=False; return
            action=next(actions,None)
            if action is None:
                self.busy=False
                if self.auto and self.rules.outcome(self.state) is None:
                    self.root.after(120,lambda: self.start(True) if generation==self.generation else None)
                return
            def next_action(success):
                speed=self.speed_multiplier()
                self.root.after(max(1,int(100/speed)) if speed else 1,lambda: step(success))
            self.play_action(action,next_action)
        step()

    def play_recommendation(self):
        if self.busy:
            return
        if self.analysis and self.analysis.root_key == self.state.key():
            self.animate(self.analysis)
        else:
            self.start(True)

    def show_alternative(self, event):
        selected = self.alternatives.curselection()
        if not selected or not self.analysis:
            return
        candidate = self.analysis.alternatives[selected[0]]
        self.write(candidate.reason)
        for action in candidate.actions:
            self.write("  "+action_text(action, self.state.board))
        if candidate.reply:
            self.write("Opponent reply:")
            for action in candidate.reply:
                self.write("  "+action_text(action, self.state.board))

    def start_auto(self):
        self.auto = True
        self.start(True)

    def pause(self):
        self.auto = False
        self.status.set("Will pause after the current turn.")

    def invalidate(self):
        self.animation_token += 1
        self.motion.clear()
        self.flashing.clear()
        self.action_menu.unpost()
        self.analysis = None
        self.selected = self.destination = None
        self.planners.clear()
        self.generation += 1

    def reset(self, state):
        self.record=GameRecord(state)
        self.game_text=self.record.dumps()
        self.record_printed=False
        self.busy = False
        self.state = state
        self.initial = state.clone()
        self.auto = False
        self.invalidate()
        self.log.delete("1.0", "end")
        self.alternatives.delete(0, "end")
        self.draw()

    def generate(self):
        if self.busy:
            return
        try:
            self.reset(league_scale(int(self.size.get()), int(self.count.get())))
        except ValueError as error:
            self.status.set(str(error))

    def restart(self):
        if not self.busy:
            self.reset(self.initial.clone())

    def load(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(filetypes=[("Map / state JSON", "*.json")])
        if path:
            try:
                self.reset(load(path))
            except (ValueError, KeyError, TypeError, OSError) as error:
                messagebox.showerror("Cannot load map", str(error))

    def current_record(self):
        # Editor changes start a new record from that exact position.
        if self.record.states[-1].key()!=self.state.key():
            self.record=GameRecord(self.state)
            self.record_printed=False
        self.game_text=self.record.dumps()
        return self.record

    def save_game(self):
        path=filedialog.asksaveasfilename(defaultextension=".awpgn",filetypes=[("AW game notation","*.awpgn"),("Text","*.txt")])
        if path:
            from pathlib import Path
            try:Path(path).write_text(self.current_record().dumps()+"\n",encoding="utf-8")
            except OSError as error:messagebox.showerror("Cannot save game",str(error))

    def review_game(self):
        from .review_ui import ReviewWindow
        from .notation import loads
        ReviewWindow(self.root,loads(self.current_record().dumps()))

    def save(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Map / state JSON", "*.json")])
        if path:
            save(self.state, path)

    def paint(self, p):
        state = self.state.clone()
        tiles = list(state.board.tiles)
        tiles[p] = self.paint_tile.get()
        state.board = Board(state.board.width, state.board.height, tuple(tiles))
        state.owners.pop(p, None)
        state.captures.pop(p, None)
        owner = {"neutral": None, "blue": 0, "red": 1}[self.paint_owner.get()]
        if owner is not None and tiles[p] in PROPERTIES:
            state.owners[p] = owner
        kind = self.paint_unit.get()
        if kind != "none":
            if owner is None:
                self.status.set("Choose Blue or Red for a unit.")
                return
            if p in state.occupancy:
                state.remove(state.occupancy[p])
            uid = f"editor_{state.next_id}"
            while uid in state.units:
                state.next_id += 1
                uid = f"editor_{state.next_id}"
            state.next_id += 1
            state.units[uid] = Unit(uid, owner, kind, p)
            state.occupancy[p] = uid
        state.winner = None
        try:
            from .scenarios import from_data, to_data
            state = from_data(to_data(state))
        except ValueError as error:
            self.status.set(str(error))
            return
        self.state = state
        self.invalidate()
        self.draw()


def main(turn_limit=None):
    root = tk.Tk()
    App(root,turn_limit=turn_limit)
    root.mainloop()


if __name__ == "__main__":
    main()
