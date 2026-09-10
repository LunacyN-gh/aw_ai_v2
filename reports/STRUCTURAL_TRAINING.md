# Search-based training and restricted-roster baseline

**Updated default:** see [teacher-guided training](GUIDED_TRAINING.md). The original search-distillation design described below has been strengthened with protected teacher replay, corrective teacher rollouts, heuristic-assisted search, and opponent promotion gates.

The default `neural-train` command now uses beam-candidate policy distillation plus terminal value regression. The old raw actor-critic experiment is retained as the Python function `neural_training.run_legacy`; the CLI uses the new trainer. Existing version-2 neural checkpoints load without changing network dimensions.

## Teacher rule

The heuristic teacher never proposes a mech purchase when the opposing army contains no treaded or wheeled units. Infantry, mechs, and aircraft alone do not enable a mech purchase. Once an enemy ground vehicle exists, ordinary production scoring applies again. This restriction filters the production candidate frontier, including alternatives searched by beam/tree. It is a teacher rule, not a simulator legality restriction or a permanent constraint on an unrestricted learned policy.

## Targets, losses, and updates

Both bootstrap and self-play now execute searched turns. Complete, reply-verified candidates are preferred when available. Candidate scores are converted to weights, and the policy target for each executed decision is the conditional distribution over candidates consistent with the preceding actions. Purchases include explicit save decisions and condition on earlier purchases sharing the same funds. Truncated turns do not manufacture save labels for unvisited factories. Targets are derived from beam scores, not MCTS visit counts or proven optimal moves.

Training explores the retained candidate turns using 90% score-derived weights and 10% uniform choice. This cannot discover an action that never reaches the candidate set. Bootstrap uses the heuristic teacher on both sides; reinforcement-phase opponents cycle through the current searched network, searched historical snapshots, the heuristic teacher, and an optional frozen baseline checkpoint. Both sides enter replay in current-self games. Actual heuristic-opponent decisions now enter expert replay; neural-opponent decisions remain excluded. Corrective teacher rollouts are queried from learner positions and carry policy labels only.

After a terminal game, each recorded state receives a +1/-1 target from its acting player's perspective. Bootstrap therefore teaches value as well as policy. For turn-limited games, search-policy labels remain usable but value targets are absent. No draw labels are invented.

Loss per sampled decision is:

`cross_entropy(search_target, policy) + 0.5 * terminal_value_MSE - 0.01 * policy_entropy`

The value term is omitted when no terminal label exists. Logging separates `policy_loss`, `production_loss`, `value_mse`, `entropy`, and `total_loss`, with sample counts. CE and MSE are nonnegative; entropy regularization can make the combined loss negative. A row with `value_mse: null` performed no supervised value update. Replay can supply previous terminal labels even when the newest game times out; `new_value_labels` distinguishes these cases.

The default replay buffer holds 8,192 decisions, with four optimizer updates of up to 64 decisions per game. Batches balance unit decisions and production decisions when both exist, so purchases are not overwhelmed by movement examples. The optimizer remains AdamW with learning rate 0.0003, weight decay 0.0001, and gradient norm clipping at 1.0.

## Evaluation and checkpoints

After bootstrap, every 25 self-play games, and at the end, the trainer evaluates against a fresh fixed heuristic teacher, with the learner on both sides. It evaluates the greedy raw policy and deployed beam separately. Four games per mode are the default; `--eval-games` must be even. Fixed-map runs reuse that map; generated-map runs use fixed held-out seeds. Evaluation games never enter replay. Both beam players use `--eval-seconds` and the configured node cap, so wall-clock noise can still affect outcomes.

Wins, losses, censored games, and search coverage are recorded. A small evaluation suite is a progress diagnostic, not statistical proof of superiority. `*.best.pt` now retains the initial reference or latest accepted promotion. Candidates must pass incumbent head-to-head, teacher non-regression, and opening checks; improving teacher wins alone is insufficient. It is not automatically promoted to the GUI default.

The requested output path is updated atomically every ten games, after bootstrap/evaluations, and at completion. Its JSON report includes progress and a `complete` flag. This replaces the old save-at-end-only behavior. `--checkpoint` still resumes weights and the stored roster with a fresh optimizer, replay buffer, random sequence, and historical opponent pool; it is not an exact interrupted-run resume.

## Infantry/tank/artillery baseline

Run from `C:\Users\apple\OneDrive\Desktop\AW_AI_v2`:

```powershell
.\.venv\Scripts\python.exe -m aw_ai neural-train `
  --map maps/v1/test_cities_6x6.json --capture-limit 7 `
  --roster ita --bootstrap 50 --games 1000 --turns 150 `
  --device auto --workers 4 --batch-size 64 `
  --train-seconds .15 --train-nodes 1000 `
  --eval-every 25 --eval-games 4 --eval-seconds 1 `
  --output models/cities_ita.pt
```

`ita` restricts initial units and teacher/network purchases to infantry, tank, and artillery. Fixed maps containing other unit types are rejected, not silently converted. Generated training armies use the selected roster. The full network architecture/output dimensions remain unchanged: masks enforce the restriction. It is stored in the checkpoint, so GUI/search and raw-policy use continue to respect it after loading. Existing full-roster checkpoints default to the full roster.

Once the restricted model shows useful evaluation results, include it as a frozen opponent during general training:

```powershell
.\.venv\Scripts\python.exe -m aw_ai neural-train `
  --map maps/v1/test_cities_6x6.json --capture-limit 7 `
  --roster all --bootstrap 50 --games 1000 --turns 150 `
  --opponent-checkpoint models/cities_ita.best.pt `
  --output models/cities_general.pt
```

To initialize the learner from that baseline too, additionally use `--checkpoint models/cities_ita.best.pt`; explicit `--roster all` expands its purchase mask. This transfers weights, not evidence that its value estimates generalize to aircraft and other new units. When facing the baseline, the current learner may build the full roster while the frozen baseline retains its restricted purchase mask.

Useful controls: `--updates-per-game`, `--batch-size`, `--replay-size`, `--checkpoint-every`, `--eval-every`, `--eval-games`, `--eval-seconds`, `--train-seconds`, and `--train-nodes`. Training now runs search for each turn and will take longer than raw-policy sampling. Inspect win/loss and coverage trends before increasing run length. No claim of improved playing strength follows merely from implementing this pipeline.

## Throughput and timing

The learner uses real minibatches through the CNN, masked entity transformer, and variable-length action/production heads. Boards are grouped by exact dimensions. `--device auto` selects CUDA when available, otherwise CPU; `--device cuda` fails clearly if CUDA is unavailable. The CLI defaults to four CPU game workers with one PyTorch thread each (`--workers 1` is serial). The Python `run()` API defaults to one worker so scripts/tests must opt into multiprocessing from a guarded main function.

Games and policy/beam evaluations use persistent spawned workers. A round shares frozen learner weights and is capped at the worker count, with additional boundaries for bootstrap, evaluations, and checkpoint writes. Historical opponents update only on accepted promotions. Completed games are ingested in seed order, with the same number of optimizer updates per game. Training can overlap outstanding workers. With four workers, the collection policy is at most three games behind the updated learner. `policy_after_game` and `policy_lag_games` expose this delay. Changing worker count changes learning trajectories; wall-clock search cutoffs also depend on CPU load. No search budgets or learning targets are reduced by this optimization.

Game logs record `play_seconds` (worker setup and searched play), `training_seconds`, and their sum in `elapsed_seconds`. These per-game durations overlap across workers and must not be added to estimate wall time. The new `round` log gives actual round wall time including waiting, training, and in-round saves. Evaluation logs have actual elapsed time; all logs also expose cumulative `run_elapsed_seconds` where applicable. The final report has `total_elapsed_seconds` measured before its last write. Initial CUDA and process startup are one-time costs. See [performance measurements](TRAINING_PERFORMANCE.md).
