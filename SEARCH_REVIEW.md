# Full AI game review

Restart the GUI and open **Review game…**. Review now defaults to **Full search**,
using the same Planner + checkpoint deployment adapter as main-window beam play.
The checkpoint selects its bundle-search backend and saved value/prior settings.
Set **Search seconds** (default 1.0), press Enter or Analyze, and choose top 1–5.

The display groups sampled turn plans by their first action, keeping the best
continuation for each. It shows the first action, score, complete continuation,
reply-coverage status, and the actual recommendation. Search may find fewer than
k distinct first actions. It does not exhaustively score all legal first moves.
**One-action value** retains the earlier direct neural successor ranking for
comparison. Both modes analyze asynchronously and discard stale results.

## Current clock36 checkpoint / training search settings

- Training: 0.5 seconds per player turn; evaluation: 1.0 second.
- Shared work/node budget: 4,000; not 4,000 MCTS tree nodes.
- Bundle backend version 2; maximum six distinct complete-turn candidates.
- First 40% of the time budget is for candidate generation. Always retain a
  complete greedy-policy turn, even if this slightly overruns the budget.
- Add a heuristic-teacher turn when possible, then policy-sampled turns.
- Sampling temperature 0.7; 25% of sampled proposal attempts use at least 1.2.
- A bundle orders the remaining unit actions, chooses build/save at available
  production sites with shared funds, then ends the turn. Terminal wins can end
  it earlier. Generation rejects interrupted proposals.
- Compare one greedy-neural opponent reply and one heuristic opponent reply.
  A reply method is retained only if it finishes for every live root candidate.
  Retain the worst value across completed reply methods. With no common reply
  coverage, keep the greedy baseline except for discovered immediate wins.
- Nonterminal leaf value: `0.5 * neural_value + 0.5 * tanh(heuristic_value/20000)`.
  Values use the original player's perspective; terminal outcomes are exact.
- Rank with `reply_value + 0.1 * mean_log_policy_probability`. The policy term
  includes unit and production decisions; it is length-normalized.
- Training target temperature 0.15; explored execution mixes 90% of that score
  distribution with 10% uniform candidate selection. Evaluation/play uses the
  highest-ranking candidate rather than this training sampling.

This is bounded whole-turn candidate search, not a full PUCT/MCTS tree. Two
reply methods do not mean two further plies: both are alternative responses
on the opponent's next turn.

The GUI's 50% neural / 50% heuristic label describes the position-value blend.
It does not mean half the moves come from each agent and is not the value-loss
coefficient (which independently happens to be 0.5).

## Next map pool

All eight `maps/6x6_ita` maps now use **40 player turns / 20 full days**. Revised
funds and layouts were preserved. Four re-exported maps had lost their roster
and capture-limit metadata; their ITA roster and two-thirds capture thresholds
were restored. Use **`--turns 40`** for the next training command: that flag
overrides map deadlines. Existing game records retain their embedded old rules.
