# Bundle MCTS

This backend is actual PUCT: tree nodes are turn-boundary states, edges are
complete turns, newly reached states receive neural value estimates, and the
result is backed up along the selected path. Opponent nodes minimize the root
player's value. Terminal results are exact. Trees are rebuilt each turn.

Defaults (saved in checkpoints and configurable for neural training):

| Flag | Default | Meaning |
|---|---:|---|
| `--search-mode mcts` | opt-in | Use MCTS rather than old bundle search |
| `--mcts-simulations` | 128 | Maximum simulations per turn |
| `--mcts-depth` | 6 | Maximum player turns down the tree |
| `--mcts-c-puct` | 1.5 | Prior exploration coefficient |
| `--mcts-widening` | 2 | Children limit = ceil(coefficient * sqrt(visits + 1)) |
| `--mcts-max-children` | 16 | Maximum distinct turn outcomes per node |
| `--mcts-visit-temperature` | 1 | Exponent 1/temperature for root visit targets |
| `--teacher-search` | ordered | Stronger heuristic in training, corrections and teacher evaluation |

The existing `--train-seconds`, `--eval-seconds`, `--train-nodes` caps still
apply. Default CLI times remain 0.5/1 seconds; use 1/2 explicitly for the next
experiment. Atomic fallback completion and individual inference operations can
overrun a deadline. Nodes/work units are not synonymous with simulations.

Every root retains a complete greedy neural turn. Widening proposes greedy
neural, heuristic, sampled neural turns and one root ordered-beam proposal.
The ordered proposal gets at most 12% of the requested time and 300 work units,
charged to the tree budget (its atomic fallback can overrun that slice).
Sampling includes actor/action ordering; 25% of sampled proposals use elevated
temperature by default. Equivalent child states are merged within a node.
After eight consecutive duplicate proposals a node stops widening; this is a
bounded practical proposal mechanism, not an exhaustive action generator.

Priors use normalized mean action log-probability with a 10% uniform floor,
not a product of action probabilities. The teacher and ordered proposal are
scored under that same policy. There is no extra additive policy score in Q.
Q is the average backed-up root-perspective result, not a calibrated win rate.
Deployment selects the most visited root edge (Q breaks ties); discovered
immediate wins take precedence. Without simulations it returns the greedy turn.
Unvisited alternatives are excluded from training targets.

Training uses root visit mass conditioned on the executed action prefix,
including production/save decisions. Terminal outcomes still train the value
head. The existing 10% uniform execution exploration remains; there is no root
Dirichlet noise in this version. Teacher examples remain expert action targets.
The `beam` key in historical evaluation JSON now means deployed search and can
refer to MCTS; the config and checkpoint identify the actual backend.

Example warm-start command from AW_AI_v2 (change games/output as desired):

```powershell
$mapArgs = Get-ChildItem .\maps\6x6_ita\*.json | Sort-Object Name | ForEach-Object { '--train-map'; $_.FullName }
.\.venv\Scripts\python.exe -m aw_ai neural-train @mapArgs `
  --checkpoint models/eight_maps_ita_clock40_neural100.best.pt `
  --output models/eight_maps_ita_mcts.pt `
  --roster ita --bootstrap 0 --games 1200 --turns 40 `
  --search-mode mcts --neural-value-weight 1.0 --teacher-search ordered `
  --train-seconds 1 --eval-seconds 2 --workers 4 --device cuda `
  --eval-games 16 --eval-every 25
```

This resets optimizer/replay and promotion history, as before. Teacher benchmark
scores are not directly comparable with runs using the old heuristic. Use
`--teacher-search old` to retain that benchmark. Source weights are unchanged.

MCTS-trained checkpoints dispatch automatically in the play GUI and full-search
review. To test an existing checkpoint without changing it, select `mcts` in
the play GUI, `MCTS` in review, or `--search mcts --checkpoint ...` in CLI analysis.
Review ranks MCTS alternatives by visits and displays visits/Q. One-action
review remains pure neural value. Tournament entries accept `search: "mcts"`
to evaluate an existing checkpoint with MCTS without modifying its source file.

Counters include completed simulations, expansions, duplicate proposals, root
children/visits, maximum depth, summed simulation depth, baseline selections,
and zero-simulation fallbacks. Depth maximum is aggregated as a maximum across
turns; other counters are sums. Evaluation includes per-side search counters.

This is an initial implementation, not a proven strength improvement. There is
no cross-turn tree reuse, global transposition DAG, batched leaf-inference server
or solved-node propagation yet. The within-turn generator is still selective.
