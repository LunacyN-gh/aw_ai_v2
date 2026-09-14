# Order-sensitive search experiment

`aw_ai.order_search.OrderPlanner` implements a shared partial-turn beam for
heuristic and neural agents. The existing `Planner` and training defaults are
unchanged. This backend is not MCTS and does not change training targets.

It retains a complete greedy fallback, branches across actors and attacks,
captures and purchases, and merges identical states including acted flags and
production-site decisions. Partial prefixes are ranked by completed greedy
rollouts. Beam width defaults to four, with diversity across acted-unit sets.
Action candidates are policy-pruned; it does not exhaustively solve combinations.
Production occurs after unit actions and branches on purchases/save.

Generation uses up to 55% of a soft deadline. Up to eight distinct complete
plans receive a single greedy opponent reply using the entrant's policy. The
baseline reply is completed first; other plans enter comparison only when their
own reply finishes. If the baseline reply cannot finish, retain the baseline
plus discovered immediate wins. Work cap is shared; baseline completion is
atomic. Additional time does not currently deepen beyond one opponent turn.

Neural actors and completions use raw neural policy; heuristic actors use
heuristic proposals. Final neural values respect the configured blend; exact
terminal results take priority. Heuristic values use the existing evaluator.

Tournament entrants accept `search: "old"` (default) or `search: "ordered"`.
Run from AW_AI_v2:

```powershell
.\.venv\Scripts\python.exe -B -m aw_ai.tournament --config tournaments/order_search.json --output reports/order_search_round_robin --workers 4
```

The experiment uses 2 seconds and 4,000 work units per turn, all eight maps,
40-turn deadlines, both sides per pairing (96 games). Both neural entrants
use the same frozen game-275 best checkpoint of the 100%-value run. This
compares complete search backends, including proposal and reply differences;
it is not an isolated ablation of action ordering alone. Inputs, effective
settings, per-game counters and portable replays are saved in the report folder.
