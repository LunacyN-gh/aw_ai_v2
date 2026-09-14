# Reusable round-robin tournaments

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m aw_ai.tournament --config tournaments/value_blend.json --output reports/value_blend_round_robin --workers 4
```

Every unordered pair plays every listed map twice, swapping entrant colors.
Four entrants and eight maps produce 96 games, 48 per entrant. The heuristic
entrant uses the same heuristic Planner as the training teacher. Neural entrants
use the checkpoint's complete deployment search; `neural_value_weight` overrides
only its value blend. Omit that field to retain the checkpoint's setting.

Copy/edit the JSON config for future tournaments. Each entrant has a unique
`name`, a `type` (`heuristic` or `neural`), and for neural entrants a `checkpoint`.
Paths resolve relative to the config file. `seconds` and `nodes` apply equally
to both sides. `max_turns` is a safety cap; map deadlines still determine real
income wins/draws. An optional `turn_limit` sets remaining player turns on every
initial position. An unfinished game is reported separately, not scored a draw.

The output contains:

- `manifest.json`: exact initial states, original file hashes, frozen checkpoint
  hashes, engine code hashes and search budgets.
- `inputs/`: frozen checkpoint copies (one copy per unique hash).
- `games/`: individual results, effective entrant settings and search counters.
- `replays/`: portable `.awpgn` games, loadable in **Review game…**.
- `summary.json` and `results.md`: continually updated standings and pair results.

Rerunning the same command resumes missing/failed games. Changed maps,
checkpoints, configuration or engine code require a new output directory.
Never overwrite a running tournament. A `.running` file prevents concurrent
writers; after a hard crash, verify its PID is no longer running before removing
that stale lock. Graceful completion/failure removes the lock automatically.

Game scheduling uses CPU worker processes with one PyTorch thread each. Neural
entrants share the same model weights but independent per-game search state.
There is no training or modification of source checkpoints. Wall-clock search
is not perfectly deterministic under changing CPU load. Paired map/color games
are useful comparisons but are not independent samples of all possible maps.
