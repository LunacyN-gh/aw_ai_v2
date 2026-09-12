# Clock-aware ITA training

The eight maps in `maps/6x6_ita` now have an income deadline of **36 player turns**:
18 turns for Blue and 18 for Red (18 full days). After Red ends the last turn,
higher recurring income wins; equal income is a draw, regardless of stored funds
or army value. Capture-limit, HQ and elimination victories can finish earlier.
No next-turn income or repairs occur after the deadline.

Neural training uses `--turns` as the remaining player-turn deadline for every
training and evaluation position, overriding the map's deadline. For example,
`--turns 36` starts a fresh map with 36 remaining player turns. Generic arena
`--turns` remains an execution safeguard; use a map's `turn_limit` or explicit
`--turn-limit` for its game deadline. Maps without `turn_limit` remain unlimited.

## Weights

`models/eight_maps_ita.clock.pt` is the game-1025 best checkpoint migrated to
schema 3. All old weights, roster, search settings and metadata are retained.
Three global inputs were added: deadline enabled, remaining player turns / 100,
and total player-turn limit / 100. Their 384 incoming weights start at zero.
This is a compatible starting point, not a model already trained on deadlines.

Old schema-2 checkpoints also migrate automatically when loaded. To write a
separate migrated file yourself:

```powershell
.\.venv\Scripts\python.exe -m aw_ai neural-migrate --checkpoint models/eight_maps_ita.best.pt --output models/another_clock_copy.pt
```

Migration refuses to overwrite existing output files. Starting a training run
restores model weights and settings, not the previous optimizer/replay buffers.

## Next cycle

Run from `C:\Users\apple\OneDrive\Desktop\AW_AI_v2`:

```powershell
$mapArgs = Get-ChildItem .\maps\6x6_ita\*.json | Sort-Object Name | ForEach-Object { '--train-map'; $_.FullName }
.\.venv\Scripts\python.exe -m aw_ai neural-train @mapArgs `
  --checkpoint models/eight_maps_ita.clock.pt `
  --output models/eight_maps_ita_clock36.pt `
  --roster ita --bootstrap 50 --games 1000 --turns 36 `
  --workers 4 --device cuda --eval-games 16 `
  --train-seconds 0.5 --eval-seconds 0.5
```

Choose a new output stem if that run already exists. The bootstrap exposes the
existing weights to the new outcomes before self-play. Both training and
evaluation above use 0.5 seconds per player turn (the previous run evaluated at
1.0 second). Results are not directly comparable with that run because both
the game deadline and evaluation budget changed.

For human play, load one of the eight maps and the model in the GUI. The summary
shows the deadline and remaining player turns. Save/load and restart preserve
the limit. Deadline wins and draws stop both human moves and automatic AI play.

## Results and labels

True wins/losses produce full-weight value targets +1/-1; equal-income deadline
draws produce 0. Counterfactual teacher corrections remain policy-only.
`winner: -1` denotes a terminal draw, `winner: null` means unfinished.
Evaluation summaries separately count `win`, `loss`, `draw`, and `censored`.
Termination reasons include `deadline_income`, `deadline_draw`, and the existing
early victory reasons. Promotion counts draws as half a point, independently
of the legacy half-point convention for unfinished evaluation games.

Clock values are included in inference/search cache keys. Search treats deadline
results as exact terminal outcomes, not neural or heuristic estimates.
