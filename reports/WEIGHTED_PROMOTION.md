# Weighted promotion

Match score is `(wins + 0.5 * turn_limit_games) / games`. Unfinished games
remain reported as `censored`; the half point is an evaluation convention, not
a terminal training target. Value training still omits unfinished outcomes.

Let C be the challenger, I the incumbent and T the teacher. Promotion requires:

`teacher_weight * (S(C,T) - S(I,T)) +
(1-teacher_weight) * (2*S(C,I) - 1) > promotion_margin`.

Defaults: `--promotion-teacher-weight .5 --promotion-margin .05`. Equality with
the margin is not sufficient. This is a practical acceptance rule, not a
statistical significance claim. The incumbent head-to-head score is the
complement of the challenger score from the same games, not a separate match.

Teacher comparisons use deployed search (`beam` in legacy report keys).
Raw policy results and opening checks remain visible diagnostics; neither can
veto promotion. The incumbent is re-evaluated against the teacher every cycle
using the same current maps, player sides, seeds, turn limit and search budget.
Historical incumbent measurements do not enter the calculation.

Reports/logs include the component scores, combined_delta, margin, decision and
reason. No promotions or model artifacts from past runs are rewritten.
The old `--promotion-win-rate .55` flag is retained as a deprecated alias for
margin `.05`; use the new flags for new commands.

After bootstrap, `--eval-games 16` schedules four sets of 16: raw challenger vs
teacher, searched challenger vs teacher, searched incumbent vs teacher, and
challenger vs incumbent. Bootstrap has only the two challenger diagnostics.
This increases evaluation cost but removes dependence on a lucky historical
incumbent teacher score. No automatic extra confirmation games are scheduled.

The learner continues updating whether or not it is promoted; acceptance
updates `.best.pt` and the accepted opponent pool. The original reference
remains frozen. Continue the next map-pool run from the existing two-map model
under a new output name, retaining bootstrap to rebuild expert replay (weights
are restored, optimizer/replay are not).
