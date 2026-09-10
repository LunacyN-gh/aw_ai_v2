# Advance Wars AI v2

An independent implementation of the architecture proposal, located at `C:\Users\apple\OneDrive\Desktop\AW_AI_v2`. V1 remains in the sibling `AW_AI` directory. V2 has no runtime imports from v1; required legacy map fixtures are copied into `maps/v1`. The heuristic engine uses Python's standard library; optional neural training/inference uses PyTorch.

The default is a **heuristic hierarchical planner with budgeted local sequence search**. A shared multi-map neural policy/value model, self-play trainer, sampled whole-turn tree search, and linear value baseline are also included. The supplied neural checkpoint is a pipeline smoke test, not a strong agent. The simulator is not yet fully AWBW-compatible.

See [the neural and tactical implementation report](reports/NEURAL_TACTICAL.md) for architecture, commands, measured results, and remaining gaps.

The default trainer now uses protected expert replay, teacher corrections from learner positions, and heuristic-assisted search. New checkpoints default to heuristic value scoring while the neural value head continues learning. Promotion into the accepted opponent pool requires head-to-head wins without teacher/opening regression. See [teacher-guided training](reports/GUIDED_TRAINING.md) for the recommended recovery command and tests. The heuristic teacher excludes mechs when the enemy has no ground vehicles.

Neural training now batches the shared encoder and action/production heads, uses CUDA automatically when available, and defaults to four CPU game workers in the CLI. Both training games and evaluations run in parallel. Use `--workers 1 --device cpu` for serial CPU execution. See [training performance](reports/TRAINING_PERFORMANCE.md) for measurements, timing fields, and the bounded delay between collecting games and updating their model.

## Run

Python 3.11+ is required. For the desktop interface, use **`run_gui.bat`** on Windows, or run these commands from this directory:

```powershell
python -m aw_ai gui
python -m aw_ai analyze --map maps/league_scale_20.json --seconds 1
python -m aw_ai play --map maps/league_scale_20.json --seconds 1 --turns 100 --output replay.json
python -m aw_ai replay --map replay.json
python -m unittest discover -s tests -v
```

The Windows launcher first uses the local `.venv`, then Python on PATH, the existing Codex bundled runtime, or the Python launcher. A normal Python installation with Tk support is sufficient. Some restricted execution sandboxes prevent Tcl initialization even when its files are installed; the interface was smoke-tested outside that sandbox. Headless commands do not require Tk.

Installation is optional: `python -m pip install -e .` adds the `aw-v2` and `aw-v2-gui` commands.

## Desktop controls

To play against a trained model, start `run_gui.bat`, use **Load…** to open `maps/v1/test_cities_6x6.json`, then **Load neural model…** to select `models/cities_6x6.pt`. Leave **Human Blue / automatic Red** checked. Make Blue's moves and builds, then click **End turn** for Red to respond. The loaded checkpoint supplies its saved search configuration to the selected beam/tree planner; guided checkpoints combine neural and heuristic proposals and use their saved value blend; GUI play and the default neural trainer both search around the network. **Use heuristic** switches back. This map uses a seven-property capture limit, counting bases. Opening funds are 0/2000 before income, yielding 1000 for Blue and 3000 for Red on their respective first turns.

- Select an available unit, then left-click an empty highlighted destination. A menu offers **Move / wait here**, **Capture here**, and legal attacks from that square. Choose an action to execute it; moving/waiting ends that unit's action, so use the attack menu choice to move and attack together. You can also select an enemy directly; the UI uses the selected legal endpoint or the nearest legal attack endpoint.
- **Wait / move** and **Capture** are separate commands. Right-click a reachable property to capture.
- Select an empty owned base or airport, choose a unit, and click **Build**.
- **Analyze position** leaves the board unchanged. Select an alternative to inspect its ordered actions and opponent reply. **Play recommended turn** executes the recommendation.
- **Human Blue / automatic Red** invokes the AI after the human ends a turn. **AI vs AI** runs both sides; **Pause after turn** finishes the currently executing turn.
- **Load** accepts v2 snapshots and v1 JSON maps. **Save** preserves partial-turn action availability, funds, captures, and the turn counter. Opening income is credited when a fresh legacy map is loaded. Income is not added again when a saved v2 decision state is loaded. Legacy snapshots can set `turn_started: true` to opt out of opening income.
- **Generate** creates a symmetric synthetic map. **Edit map** enables terrain/ownership/unit painting; right-click removes a unit. Save edits to retain them. Restart restores the last loaded/generated state.

Analysis scores are heuristic quantities, not calibrated win probabilities. Tree scores use a different bounded scale. The interface identifies whether a candidate received an opponent reply search.

## Map scale

The initial target is the scale described by the project owner and illustrated by [AWBW's league map page](https://awbw.amarriner.com/newleaguesetups.php): approximately 20×20, 2–3 bases per side, 1–2 airports and towers per side, and 20–50 other cities.

The included `league_scale_20.json` has **three bases, one airport, one tower, and one HQ per side, plus 32 cities**. `contact_20.json` and `dense_24.json` stress contested positions with 24 and 40 units per side. These are generated fixtures, **not downloaded league maps** or claims of balanced competitive layouts. Copied v1 regression maps are in `maps/v1/`.

```powershell
python -m aw_ai generate --size 20 --units 24 --contact --seed 7 --output custom.json
```

## Implemented architecture

| Layer | Implementation |
| --- | --- |
| Rules and state | Ordered actions; checked public execution; independent full-information observation; headless victory handling; immutable shared board/unit records; state cloning; bounded transit and attack-coverage caches. |
| Strategy | Reverse Dijkstra travel costs by movement class; sparse property bidding; persistent capture assignments; repair/front objectives; conservative interaction regions. |
| Policy | Selects both actor and action. Retains attack, capture and quiet proposals; threat-aware endpoint ranking; supports temporary material losses in local search. |
| Local search | Bounded beam over interleaved primitive actions, regenerated after each transition. Full-board simulation retains outside support. Coordinate descent carries improvements across regions with legality checks and dynamic region expansion. |
| Production | Shared funds across bases/airports; bounded frontier; an explicit save option; matchup, delivery-time and composition utility. No factory Cartesian product. |
| Reply search | Opponent uses the same policy and local sequence machinery, including movement, capture and production. Incomplete replies are not marked verified. |
| Continuation | Extends all retained reply lines into our next turn only if the common horizon can be completed; otherwise retains the two-turn scores. |
| Experimental tree | Sampled ordered complete turns, progressive widening and UCT. Player perspective changes at actual turn boundaries. Seeded proposal perturbations explore alternatives. |
| Learning | Pluggable Policy/ValueModel interfaces, state features, terminal-outcome collection, trainable linear value weights, versioned model files. |
| Analysis | Ordered recommendation, alternatives, searched replies, objective reasons, coverage counters, node counts and elapsed time. Stale plans are rejected. |
| Arena | Headless play, hash-checked action replays, side-swapped comparison against a policy-only ablation, explicit turn-limit censoring. |

All search phases share one node/deadline budget. A legal end-turn incumbent is available immediately. The deadline is cooperative: one in-flight policy call or distance-field calculation can finish shortly after it. At very short budgets the AI may leave units unused; `complete` and coverage metrics expose this. Completed candidates take precedence over interrupted ones. More time is required to compare several full replies for a dense army.

The interaction graph is deliberately conservative and currently uses pairwise unit checks. A long connected front can become one region. Cloning still copies unit/property dictionaries; the immutable terrain and unit records are shared. These are measured optimization opportunities, not claims of constant-time or perfectly linear scaling.

## Rules profile and compatibility boundary

Every snapshot identifies `deterministic-v2`. The profile supports infantry, mech, recon, tank, medium tank, artillery, rockets, anti-air, battle copter, fighter and bomber. Terrain includes cities, bases, airports, towers, HQs, roads, woods, mountains, water, rivers, bridges and shoals.

It implements movement classes, direct attacks and counters, stationary indirect fire, explicit capture, tower attack bonuses, production rosters, income and property-class repairs. Unit costs/movement and the supported base-damage entries were taken from the [AWBW unit chart](https://awbw.amarriner.com/units.php), [damage chart](https://awbw.amarriner.com/damage.php), and [terrain chart](https://awbw.amarriner.com/terrain.php).

Deliberate remaining differences:

- No COs/powers, luck, fuel/ammo, transports, joining/resupply, naval units, weather or fog.
- Combat and repairs use the documented internal 1–100 HP arithmetic in `rules.py`; AWBW rounding/parity has not been established.
- Enemy units of every domain conservatively block transit. Air/ground transit interactions are not asserted to match AWBW.
- Towers give attack bonuses and no income or repairs. Income-producing properties grant 1,000 per turn.
- Capturing an enemy-owned HQ wins immediately. An army with no units remains alive while it owns a base or airport. This production-aware elimination convention is a sandbox rule.
- There is no hard unit cap or game-specific draw rule. An optional income-property capture limit ends the game immediately; towers do not count. Arena turn limits are censored outcomes, not training draws.

V1 maps import as states under the **v2** rules profile; importing a map does not preserve all v1 combat/evaluation semantics. A second game's adapter needs its own rules, tactical proposals and training; changing a label is insufficient.

## Experiments and learning

```powershell
# Tactical dependencies
python -m aw_ai analyze --scenario screen --seconds 1
python -m aw_ai analyze --scenario blocker --seconds 1

# Compare search implementations; neither is declared stronger yet
python -m aw_ai analyze --map maps/contact_20.json --seconds 3 --search tree
python -m aw_ai benchmark --seconds 1 --repeats 5 --output benchmark.json

# Matched per-turn budgets, swapped agent sides, policy-only ablation opponent
python -m aw_ai compare --size 20 --units 12 --seconds 1 --games 10 --turns 150 --output comparison.json

# End-to-end training smoke test on a tiny terminal fixture
python -m aw_ai train --scenario blocker --games 4 --turns 10 --seconds .2 --output value.json
python -m aw_ai analyze --scenario blocker --model value.json
```

Search controls include `--nodes`, `--beam`, `--depth` (local primitive actions), `--candidates`, `--replies`, and `--no-continuation`. The production frontier has a minimum width of two so both saving and buying can survive. To run reproducibility tests, give a generous time budget and a fixed node budget; wall-clock cutoffs naturally vary with system load.

Complete-turn policy rollouts compare two candidate actors at a time; local sequence search considers up to six. These are separate budgets: cheap rollouts complete the army's turn while detailed search explores more action ordering in contested regions. Both can be configured through `Config` in Python.

The linear learner fits terminal game outcomes and refuses to train when every game reaches the turn limit. The separate `neural-train` command learns policy targets from searched turns and value targets from completed games, including bootstrap games. It defaults to a two-thirds income-property capture limit. `models/smoke.pt` demonstrates the earlier pipeline, not the current trainer's strength. There is no human-replay dataset or established playing-strength result.

Reports from implementation checks are in `reports/`; small timing samples are diagnostic, not a strength benchmark. `compare` currently compares against the v2 policy-only ablation, **not the original v1 agent**. A reliable v1-versus-v2 strength comparison requires a common rules profile and a controlled time budget for v1.

## Validation and next decision gates

The default suite checks tactical move/attack dependencies, captures, legal production, cache invalidation, tower/air behavior, exact state/replay round trips, source-map imports, global budgets, local regions, model fitting and both search implementations. Optional hidden-window Tk integration checks are separate:

```powershell
python -m unittest discover -s tests_gui -v
```

The next gates are sustained training, held-out map-family tournaments, and AWBW mechanics conformance. The spatial/entity network now uses the existing policy/value interfaces. Keep the heuristic beam baseline when evaluating strength per second. Fog beliefs, full AWBW mechanics, and adjacent-game adapters remain later stages.

## Files

```text
aw_ai/model.py        compact state, actions, supported unit data
aw_ai/rules.py        authoritative deterministic simulator and caches
aw_ai/strategy.py     objectives, travel times, interaction graph
aw_ai/policy.py       action proposals, threat ranking, production
aw_ai/planner.py      budgeted local search and full-turn comparison
aw_ai/tree.py         experimental sampled whole-turn UCT
aw_ai/evaluation.py   features and heuristic/trainable value models
aw_ai/arena.py        self-play, replay checks, outcome samples
aw_ai/scenarios.py    snapshots, v1 map importer, scale fixtures
aw_ai/gui.py          Tk desktop board, editor and analysis interface
aw_ai/cli.py          headless commands and experiment entry points
```
