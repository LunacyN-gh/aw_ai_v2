# Policy-informed bundle search (version 2)

Policy proposes coherent alternatives; value compares their consequences.
This is complementary use of the two heads, not a claim that value-guided
exploration is inherently inferior. A weak value estimate can mistakenly favor
the same unusual states both during generation and during selection. Using
policy proposals reduces that feedback; subsequent value-based comparison can
still overturn the policy. Uncertainty-guided exploration or PUCT remains a
possible later extension when the evaluator is better validated.

## Changes

* Preserve the exact greedy raw-policy turn, including sequential purchases and
  save choices, with the same legal ordering and argmax as raw evaluation.
* Default shortlist: at most six distinct resulting states, including the raw
  baseline, a teacher candidate when distinct, and policy-sampled turns.
* Policy sampling uses temperature .7; 25% of proposal attempts use at least
  temperature 1.2. Generation reserves 40% of the total deadline. Teacher
  variants cannot consume the entire shortlist before policy samples arrive.
* Complete a greedy policy reply for every live root before committing that
  comparison. Then attempt a teacher reply for every root. If any candidate's
  reply times out, discard that entire comparison round. All compared roots
  therefore face identical completed reply methods.
* With no complete common reply round, keep the raw baseline (except an exact
  terminal win). Do not distill unevaluated alternatives as improvements.
* Rank by `Q + policy_prior_weight * mean_log_policy_probability`, default .1.
  Q is the worst value over the common reply methods, blending neural and
  bounded heuristic values as before. Mean log probability includes save/build
  choices and avoids penalizing plans simply for containing more actions. It
  is a length-normalized policy preference, not a joint turn probability.
* Exact root terminal outcomes dominate nonterminal preferences. No arbitrary
  "always capture" rule was added. Entropy coefficient stays .01.
* Training targets remain score-softmax distributions conditional on action
  prefixes, not visit counts. This is prior-regularized candidate evaluation,
  not PUCT or full MCTS.
* Logs count raw retention/selection, common reply rounds and discarded rounds.
  Analysis additionally exposes per-candidate root value, reply value, policy
  preference and combined selection score in `bundle_diagnostics`.

The exact raw baseline is completed atomically. It can exceed a tiny deadline,
especially on larger boards; optional proposals/replies are budgeted. This
tradeoff deliberately prevents timeout-based exclusion of the baseline. Larger
map performance has not been validated for this backend.

## Compatibility and use

New training uses `--bundle-version 2 --bundle-candidates 6` by default. Existing
checkpoints without a version retain version-1 deployment. Resuming training
uses the new CLI defaults and writes them into the new checkpoint. The earlier
backend remains selectable using `--bundle-version 1 --bundle-candidates 16`.

For immediate GUI testing, load `models/cities_ita_policy_search.pt` and use the
normal beam deployment. This is a copy of the completed game-250 model with
unchanged neural weights and version-2 settings. It is not a newly trained or
promoted model. Original checkpoints are unchanged. Explicit tree mode remains
the older experimental UCT backend.

To continue training from the completed model:

```powershell
.\.venv\Scripts\python.exe -m aw_ai neural-train --map maps/test_cities_6x6_ita.json --checkpoint models/cities_ita_value_fresh.pt --output models/cities_ita_policy_trained.pt --roster ita --bootstrap 50 --games 200 --turns 50 --workers 4 --device cuda
```

For a fresh run, omit `--checkpoint`. Budgets remain .5 seconds in training and
1 second in evaluation. Replay/optimizer are not restored from checkpoints;
bootstrap rebuilds protected teacher examples.

When the second map is available, append:

```powershell
--eval-map maps/test_cities_6x6_ita.json --eval-map maps/YOUR_SECOND_MAP.json --eval-games 4
```

The explicit list replaces the default evaluation map source. Each map receives
both learner sides; the evaluation count must be divisible by twice the map
count. Per-game map hashes are recorded. These evaluation maps are not
automatically added to training. Different maps increase coverage, but four
games still provide a small strength sample.

## Validation

93 core tests and 11 GUI tests passed. Added tests cover exact greedy-turn
equivalence, legal/reproducible policy sampling, preservation under zero reply
budget, whole-round discard on an interrupted reply, distinct evaluation maps,
and version-2 checkpoint persistence.

Using the frozen final game-250 checkpoint on the four opening/capture fixtures:
version 1 passed 2/4; version 2 passed 4/4, choosing the raw baseline after two
complete common reply rounds in every fixture. No weights were updated. The
diagnostics show that policy preference can counter a small estimated value
advantage assigned to a less-preferred capture-abandoning plan.

A six-game frozen comparison used .5-second/4000-node search, three CPU workers,
50-turn limits, one map and both sides per opponent. New search scored 2–0
against version 1, 2–0 against raw policy, and 1–1 against the heuristic teacher.
This is preliminary evidence, not a stable win-rate estimate or a multi-map
strength claim. Timing-sensitive search and the small sample limit conclusions.

See `policy_frozen_positions.json`, `policy_frozen_matches.json`, and
`policy_smoke_validation.json` for the recorded results. The smoke report is a
short CUDA/four-worker training pipeline check, not a strength experiment.
