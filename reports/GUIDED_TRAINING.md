# Teacher-guided training after the ITA regression

The previous run learned useful bootstrap behavior and then regressed as its own searched decisions replaced expert supervision. The new default retains expert examples, obtains fresh teacher corrections, and uses heuristic-assisted search while the value head learns. This is an implementation and regression-tested recovery path; it has not yet demonstrated sustained learning over hundreds of games.

## Replay and labels

The ordinary search replay buffer remains bounded by `--replay-size` (8,192 by default). An independent teacher buffer is controlled by `--teacher-replay-size` (also 8,192):

- Half its capacity is a reservoir sampled from all bootstrap decisions. After bootstrap this reservoir is never overwritten by self-play or new corrections.
- The other half holds recent teacher-opponent decisions and corrective teacher rollouts.
- `--teacher-fraction .5` reserves half the requested minibatch for expert examples. When both expert pools exist, those slots divide equally between bootstrap and recent examples. Unit and production decisions are balanced within each source. Small expert pools sample with replacement to maintain the quota; scarce ordinary replay can make the actual batch smaller and the expert fraction larger.

Every `--teacher-every 2` learner turns, counted separately for each player, the worker searches a teacher continuation from the learner's actual current position. The first label is at that encountered position; later labels describe the teacher's alternative continuation. This is bounded teacher rollout supervision, not a teacher search at every primitive learner decision. These counterfactual examples **never receive the actual game's terminal value label**. Actual bootstrap, learner, and teacher-opponent trajectories retain terminal +/-1 labels; turn-limited trajectories remain censored.

Teacher turns choose their recommended candidate without exploratory candidate sampling, and expert policy labels are one-hot recommendations. This avoids teaching a mixture of weaker alternatives as the teacher's answer. Learner turns retain exploratory candidate selection and soft search-policy targets. Fixed-map bootstrap therefore has limited trajectory diversity; corrective rollouts from learner positions supply additional situations.

Production has a separate completion marker: exhausting the search budget must not manufacture a "save" target at an unsearched factory. A partial production plan still supplies labels for purchases it actually made. A zero-budget end-turn fallback supplies no purchase/save supervision.

The loss coefficients, network dimensions, AdamW settings, and GPU minibatching remain unchanged. Logs now distinguish `teacher_ce` from `search_ce`, with sample counts, so a declining self-imitation loss cannot hide poor teacher agreement. Protected replay lasts for the current run. As before, `--checkpoint` restores weights and roster, not optimizer/replay state; rebootstrap when starting a new run to establish fresh anchors.

## Search and deployment

`GuidedAgent` interleaves bounded heuristic and neural proposal lists. A neural logit's magnitude cannot remove all heuristic proposals. Production retains the teacher's utility scale and applies a bounded neural rank adjustment. A complete heuristic turn is attempted before neural expansion, within the same search deadline/node budget. Opponent reply search uses the same mixture and heuristic fallback. This is a bounded safety mechanism, not proof that the teacher's best move always survives all pruning or timeouts.

`--neural-value-weight 0` is the default: deployed search uses heuristic value, while the neural value head still trains from terminal results. A value between 0 and 1 enables an explicit linear blend; it does not increase automatically. This avoids trusting the regressed value head before evaluation supports doing so.

New checkpoints save the guided-search settings. GUI loading, CLI analysis/play, training, and evaluation use the same settings. The GUI reports the loaded neural value weight. Old checkpoints still load with their original raw-neural search behavior; starting a new training run applies the new guided defaults. The network weights and dimensions remain compatible. Beam is the training/evaluation path; the existing experimental tree can use mixed proposals/value but is not the promotion evaluation target.

Because guided beam uses a heuristic teacher at inference, its wins are **not evidence that the raw neural policy has learned the same strength**. Evaluation continues to report raw policy and deployed beam separately.

## Stable opponents and promotion

Training cycles through current-self, accepted snapshots, the heuristic teacher, and an immutable initial reference; an optional baseline checkpoint adds another opponent. Each opponent receives a Blue/Red pair, avoiding side bias from an even-length opponent cycle. Self games use the round's current learner on both sides. Only accepted checkpoints enter the historical pool; there is no automatic snapshot every four games.

After bootstrap (or before training when bootstrap is zero), the evaluated learner initializes the frozen reference and incumbent. The reference stays fixed throughout the run. Subsequent evaluations include head-to-head games against the latest incumbent. Promotion requires all of:

1. At least `ceil(--promotion-win-rate * games)` wins against the incumbent, and more wins than losses. The default rate is 0.55. Censored games do not count as wins.
2. At least as many wins and no more losses against the teacher, for **both** raw policy and beam, compared with the incumbent's recorded evaluation.
3. Passing all deployed-search opening checks, with no drop in the number of raw-policy opening checks passed.

The small fixed evaluation suite is a conservative regression gate, not a statistical strength certificate. Fixed-map games repeat the map with swapped sides; search cutoffs depend on machine load. More evaluation games and held-out maps are needed for strength claims. Promotion does not roll back the learner or discard its updates; it controls trusted opponents and the recommended checkpoint.

Files for an output such as `models/cities_ita_guided.pt`:

- `cities_ita_guided.pt`: latest learner, even if promotion failed.
- `cities_ita_guided.reference.pt`: immutable initial reference for this run.
- `cities_ita_guided.best.pt`: initial incumbent or most recent accepted promotion.
- Matching JSON files, including evaluation and promotion decisions. The latest JSON has complete run history; reference/best JSON files describe their save point.

## Opening checks and telemetry

Evaluation checks both sides for opening expansion plus infantry production, and for completing an uncontested capture on the following turn. These fixed behavior checks are not explicitly inserted into replay; normal training can naturally encounter the same opening positions. Tests also use deliberately overconfident bad policy/build priors to verify that guidance preserves these behaviors.

Training and evaluation game rows now include `played_turns`, counting individual player turns, and `termination` (`income_capture_limit`, `hq_capture`, `elimination`, or `turn_limit`). Training also reports final income, protected-anchor count, correction count, and `teacher_query_seconds`. Existing per-game, round, and cumulative timers remain. Expert queries cost additional CPU time, and promotion adds head-to-head games to evaluation; the earlier throughput benchmark did not include these new operations.

## Suggested recovery run

Start from the previous bootstrap/best checkpoint, rather than continuing the regressed latest weights. Use a new output name. From `C:\Users\apple\OneDrive\Desktop\AW_AI_v2`:

```powershell
.\.venv\Scripts\python.exe -m aw_ai neural-train `
  --map maps/test_cities_6x6_ita.json --capture-limit 7 --roster ita `
  --checkpoint models/cities_ita.best.pt `
  --bootstrap 25 --games 100 --turns 100 `
  --device cuda --workers 4 --batch-size 64 `
  --teacher-fraction .5 --teacher-every 2 --neural-value-weight 0 `
  --train-seconds .15 --train-nodes 1000 `
  --eval-every 25 --eval-games 8 --eval-seconds 1 `
  --output models/cities_ita_guided.pt
```

The longer turn limit gives competitive games more opportunity to produce real terminal labels. Review raw policy, guided beam, teacher CE, censoring, opening results, and promotion reasons before extending to 1,000 games. For human play, load `cities_ita_guided.best.pt` on the ITA-only map.

## Validation

Focused tests cover protected-anchor retention and sampling quotas, legal mixed proposals, opening production/capture completion under bad neural priors, teacher queries for both players, collection of actual teacher moves, absence of fabricated counterfactual value labels, promotion acceptance/rejection, preservation of initial/best weights after rejection, and guided checkpoint round trips including the GUI loader.

A full-size checkpoint was used in a four-worker CUDA smoke run with two bootstrap plus eight self-play games. It exercised every standard opponent type, periodic saves, teacher sampling, correction collection, and rejection of promotions when games were censored. A separate longer run exercised real capture-limit wins and value updates. These tests verify the pipeline, not sustained improvement. Small-run summaries are saved in `guided_training_validation.json`.
