# Neural, production, and tactical evaluation stage — September 7, 2026

V2 now includes composition-conditioned production, a shared neural policy/value model, a working bootstrap/self-play trainer, configurable income-property capture victory, and an evaluation-only tactical suite. V1 was not edited during this stage. Heuristic search remains the default.

## Production

Each factory/airport receives a soft responsibility share for nearby enemy and friendly forces, using terrain-aware travel times relative to other production sites. Scores combine damage matchups, friendly counter coverage, delivery time, capture/screen demand, cost, and duplicate-unit penalties. Buying a unit updates the position before the next site's choices are scored. A bounded shared-budget beam retains saving as an option; alternative production plans now reach reply/value comparison.

Tests cover different enemies on opposite fronts, existing reinforcements reducing demand, shared funds, saving, and actual planner purchases. In the controlled 8,000-fund test, the planner buys anti-air against a copter and a tank against anti-air. These are useful priors, not optimal economic play or a forced build quota.

## One shared network

Default architecture: 64-channel spatial convolution, four residual blocks, 128-dimensional unit/property tokens, three transformer layers with four heads, and separate action, production, and value outputs. It has 835,474 parameters. The optional tiny configuration has approximately 29,000 parameters and is intended for pipeline checks.

Inputs include terrain, ownership, per-type unit HP, acted flags, capture progress, coordinates, unit capabilities, money/income, and income capture-limit progress. Ownership is relative to the player to move. Units and properties read features from the spatial encoder; a global economy token lets all entities exchange strategic context. There is no map ID or map-specific output layer.

The action head combines actor, command, destination, and target scores. The rules engine enumerates legal actions; invalid actions never enter the policy distribution. The neural proposer considers all available actors rather than inheriting the heuristic's shortlist. The production head combines the factory token, global token, and local composition/arrival features, and outputs all unit types plus save. Roster, money, and occupancy determine legal purchases.

The value head predicts a bounded result from the player-to-move perspective. Planner integration rescales it to the existing heuristic score range; it is not calibrated. Search still adds strategic objective/screening priors. A shared bounded inference cache serves policy and value. Checkpoints validate the rules profile, roster, terrain schema, and version; capture-limit-aware checkpoints use version 2.

Boards are encoded at their native dimensions. Mixed-size training accumulates per-state gradients, so padding cannot influence a smaller board. This is a straightforward CPU implementation, not yet a high-throughput batched trainer. Grid work grows with area, entity attention grows quadratically with unit/property count, and legal-action scoring grows with legal actions. Bounded search time does not guarantee unchanged playing strength on larger boards.

## Training

The trainer first distills actions and purchases from heuristic search. It then samples complete turns with legal exploration and performs terminal-only Monte Carlo actor-critic updates, with a value baseline, entropy regularization, and gradient clipping. Every legal purchase, including saving, retains exploration support. Opponents cycle through current self-play, a rolling pool of older network snapshots, and the heuristic teacher. Learner sides alternate.

Each game samples a synthetic terrain seed and one of 8, half the requested maximum size, or the requested maximum size. Initial army composition varies. Validation uses disjoint seeds at or above 1,000,000 and alternates sides against the teacher. These are held-out seeds from the same generator, not held-out competitive map families. Tactical evaluation generators are never imported by training.

Self-play uses a capture limit of `ceil(2 * total_income_properties / 3)` by default. Every supported income tile counts: city, factory, airport, and HQ; towers do not. The denominator includes neutral properties. A completed capture reaching the limit ends the game immediately. Existing snapshots omit the optional limit and retain their previous rules. Limits are saved in snapshots/replays and affect state/cache keys. This is the requested income-control variant, not a claim of exact AWBW capture-limit rules parity.

Turn-limited games are censored, not draws. They produce no reinforcement update; teacher policy labels can still be used without inventing a result. Each update samples at most 64 decisions. Checkpoint loading currently resumes weights with a fresh optimizer and opponent pool, not a bit-exact interrupted training run.

From `AW_AI_v2`, using the installed isolated environment:

```powershell
# Small end-to-end run; the supplied models/smoke.pt was made this way
.\.venv\Scripts\python.exe -m aw_ai neural-train --tiny --size 8 --units 3 --games 3 --bootstrap 2 --turns 30 --output models/smoke.pt

# Example longer experiment: full network and varied map sizes
.\.venv\Scripts\python.exe -m aw_ai neural-train --size 20 --units 12 --bootstrap 20 --games 100 --turns 150 --output models/shared.pt

# Use a checkpoint in search; GUI also has Load neural model / Use heuristic
.\.venv\Scripts\python.exe -m aw_ai analyze --map maps/league_scale_20.json --checkpoint models/smoke.pt --seconds 2

# Disable the training capture limit, or set it on a map for ordinary play
.\.venv\Scripts\python.exe -m aw_ai neural-train --tiny --capture-fraction 0 --games 3 --size 8 --units 3 --turns 30 --output models/no_limit.pt
.\.venv\Scripts\python.exe -m aw_ai generate --size 20 --capture-limit 28 --output maps/league_with_limit.json
```

For another Python installation, install the optional dependency with `python -m pip install -e ".[neural]"`. `run_gui.bat` prefers the local `.venv` when present.

The supplied smoke run completed two bootstrap games and three self-play games. Only one self-play game terminated within 30 turns and produced a reinforcement update; the other two were censored. This demonstrates the path, not convergence. See `models/smoke.json` for validation outcomes and run parameters. Sustained training, learning curves, more training map families, and meaningful tournaments remain necessary before promoting a learned model over the heuristic.

## Tactical tests

Sixteen fixtures cover three protected-target breakthroughs (tank→artillery, tank→anti-air, anti-air→copter), plus HQ capture screening. Four rotations and alternating player ownership check orientation sensitivity. Snapshots for GUI inspection are in `maps/tactical_eval/`.

The exhaustive oracle verifies exactly one successful solution layout per fixture. Independent action permutations count as the same layout. For HQ protection it checks all legal opponent action sequences, including screen destruction followed by another attack, and requires the capturing infantry to remain undamaged after advancing its capture. An oracle budget overrun raises an error rather than claiming proof.

The heuristic now rewards screen kills that unlock new follow-up attacks, proposes HQ screens, and prefers safer HQ captures when continuation values tie. These are generic rules without fixture IDs. Because these tests informed engineering, they are development regressions, not an untouched strength benchmark. They remain excluded from neural training examples.

```powershell
.\.venv\Scripts\python.exe -m aw_ai tactical-eval --prove --output reports/tactical_current.json
.\.venv\Scripts\python.exe -m aw_ai tactical-eval --checkpoint models/smoke.pt --output reports/tactical_smoke_neural.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

At the default heuristic budget, all 16 solutions appear in the candidate lists and 12 are selected. The four copter variants expose a valuation disagreement: the breakthrough is found but another exchange receives a higher continuation score. The evaluator reports both discovery and selection, so more search is not mistaken for fixing that valuation issue. Forcing a puzzle objective is deliberately not part of normal game search.

The capture-limit-aware tiny smoke checkpoint selected 0/16 solutions and found 4/16 at that budget. It is substantially worse than the heuristic on these regressions and should not replace it. More training and validation are required; merely connecting a neural network does not improve the agent.

## Validation and limits

52 core/optional-neural tests pass, covering simulator changes, unique tactical solutions, production behavior, legal masks, variable dimensions, side-relative encoding, head gradients, policy learning, terminal value updates, checkpoint round trips, and replayed capture-limit wins. Four separate Tk integration tests pass outside this host's Tcl-restricted sandbox.

The one-seed heuristic scale check took approximately 0.30, 0.46, 0.96, and 0.90 seconds for 8×8/4 units, 20×20/12, 20×20/24, and 24×24/40 per side, respectively, at a one-second budget. These timings are diagnostics, not reliable latency percentiles. Neural timings and coverage are recorded separately in `neural_smoke_benchmark.json`; neural calls can consume the reply budget, and dense-board search coverage needs further optimization. Inspect incomplete-turn and verified-candidate counters, not latency alone.

Remaining work: evaluate richer tactical families with multiple preparations and sacrificial walls; calibrate the discovered copter exchange; train on substantially broader map structures and run fixed-budget tournaments; batch/cache neural work more efficiently; establish full AWBW mechanics conformance. Fog, CO powers, luck, and adjacent-game adapters remain outside this stage.
