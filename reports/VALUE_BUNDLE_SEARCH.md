# Value-guided bundle search — September 9, 2026

This iteration implements complete-turn candidate evaluation, not MCTS. The
learned value head now influences which turns are executed and imitated. The
existing experimental UCT tree has not been converted into a training backend.

## Search and learning

New neural training defaults: `--search-mode bundle`, `--train-seconds .5`,
`--train-nodes 4000`, `--bundle-candidates 16`, `--neural-value-weight .5`.
Candidate generation uses a heuristic turn, a raw neural turn, and sampled
turns. `--exploration-fraction .25` reserves part of the sampled set for raw
teacher/neural proposals; the remaining samples use mixed proposals. Sampling
includes production, respects the checkpoint roster, and merges identical
successor states. This fraction is not the total fraction of randomized turns.

Generation uses 55% of the time budget; remaining time compares immediate
opponent replies. Both teacher and neural replies are attempted, subject to
budget. Complete reply-checked candidates are preferred when available. The
opponent chooses the lowest value from the root player's perspective. Heuristic
evaluation guides construction; learned value ranks completed turns/replies.
Root neural evaluations are batched within a worker; game workers still use CPU.
CUDA batches the learner's gradient updates.

Nonterminal scoring is
`(1-w)*tanh(heuristic/heuristic_scale) + w*V(root_player)`.
Defaults are `w=.5`, `heuristic_scale=20000`. Both terms are bounded outcome-like
scores; the heuristic term is not a calibrated win probability. The mixing
weight is experimental, not a measured optimal value. Exact root terminal wins
and losses override estimates. Values are signed by player, not blindly by
number of individual actions. No game-length discount or weighting is applied.

Bundle targets use `softmax(score / target_temperature)` with default `.15`,
not the legacy beam temperature of 2000 heuristic units. These are conditional
candidate probabilities, not MCTS visits. Individual policy targets condition
on the executed action prefix and production history. Logs include soft-target
counts to expose overly sparse distillation. Self-play mixes these probabilities
with `--execution-exploration .1` uniform candidate sampling. Evaluation chooses
the best candidate; it does not sample execution.

The objective is `mean(policy_CE - entropy_weight*entropy) +
value_loss_weight*mean(labeled_value_error^2)`, defaults `.01` and `.5`.
Unlabeled teacher corrections and unfinished games have no value target and no
longer dilute the labeled value gradient. Protected teacher replay is retained.

Settings persist with checkpoints and are automatically used by the normal
Planner, including GUI beam deployment. `--search tree` remains the earlier UCT
implementation. Old checkpoints without bundle settings retain their previous
deployment behavior. Evaluation's legacy `beam` report key now means the
checkpoint's searched deployment; inspect `config.search_settings.planner`.

## Validation

88 core tests and 11 GUI tests passed. New checks cover value-driven selection,
value-gradient invariance to extra unlabeled corrections, bounded-score target
temperature, legal distillation targets, checkpoint round trip, and timeout
fallback. Existing scalar/batched gradient equivalence was updated for the new
loss normalization.

A full-size checkpoint continuation used CUDA learning, four CPU workers,
two bootstrap games and four searched games, capped at 30 individual turns.
It completed in about 76 seconds, five games terminated, and no executed turns
were partial. All eight opening checks passed at both evaluations. This is a
pipeline smoke test, not evidence of improved playing strength. Validation
report: `bundle_smoke_validation.json`. User models were not overwritten.

`bundle_benchmark.json` compares weights 0/.5/1 and deadlines .15/.5/1 seconds on
three fixed same-map positions, using one frozen checkpoint and 4000 nodes.
At weight .5, .15 seconds produced 2–10 bundles and 0–6 replies; .5 seconds
produced 12–16 bundles and 11–24 replies; 1 second produced 12–16 bundles and
20–30 replies. The node cap did not bind. These are single timing measurements,
not a strength tournament or statistically stable throughput benchmark. Full
turn construction and inference are atomic operations, so deadlines can overrun
slightly. Larger maps remain unvalidated for this new backend.

## Suggested continuation

Run from `C:\Users\apple\OneDrive\Desktop\AW_AI_v2`:

```powershell
.\.venv\Scripts\python.exe -m aw_ai neural-train --map maps/test_cities_6x6_ita.json --checkpoint models/cities_ita.pt --output models/cities_ita_value.pt --roster ita --bootstrap 50 --games 200 --turns 100 --workers 4 --device cuda --search-mode bundle --train-seconds .5 --train-nodes 4000 --eval-seconds 1 --eval-games 4 --eval-every 25 --neural-value-weight .5
```

Resumption restores weights and roster, not optimizer or replay. Bootstrap
repopulates expert anchors. New output names preserve the existing baseline.
For a controlled zero-value ablation use the same command/seed with
`--neural-value-weight 0 --output models/cities_ita_bundle_control.pt`.
This keeps candidate generation and loss changes identical. For the old beam
backend use `--search-mode beam --neural-value-weight 0`.

Next decisions should use strength per hour, reply coverage, and target
diversity. A small repeated fixed-map evaluation remains noisy. Progressive
widening, PUCT, explicit visit-count targets, and tactical search coverage are
subsequent iterations; this change does not claim to implement them.
