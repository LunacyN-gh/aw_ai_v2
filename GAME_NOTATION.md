# Recording and reviewing games

The main GUI records every committed action, including both human and AI moves.
**Save game…** writes a `.awpgn` file, including an unfinished game. When a game
ends, the complete notation is printed to the log once. Restarting/loading a
map starts a new record; editing a position starts a new initial snapshot when
the next action is recorded or the game is saved. Switching human control does
not alter the record. Games played before this feature cannot be reconstructed
from a final position alone.

**Review game…** opens a separate, read-only playback window with the current
game. Use **Load game…** for a saved file, or paste notation into the lower text
box and click **Use pasted text**. Headerless pasted moves use the current
record's initial position. Navigation does not change the game in the main GUI.

| Button | Operation |
|---|---|
| `<` / `>` | Previous / next action, including an end-turn transition |
| `<<` / `>>` | Previous / next player-turn boundary |
| `|<` / `>|` | Initial / final recorded position |

Load a neural checkpoint in the review window and choose **Top actions** from
1–5. Every shown position triggers a background evaluation. Candidate actions
are legal individual actions (builds also respect the checkpoint's roster).
The model values each resulting position; End turn correctly reverses the
value perspective. Terminal results are exact. Scores range from -1 to +1,
positive for the player to move in the displayed position. These are raw
one-action value estimates, not searched plans or calibrated win probabilities.
No model search settings or policy logits are used to rank these actions.

Both windows show army purchase value scaled by **displayed HP** above funds:
a tank showing 8 HP is worth 5,600, including internal HP from 71 through 80.

## AWPGN version 1

Each number is a player turn (ply), rather than a pair of turns. Moves in that
turn are separated by commas. Files include an `InitialState` JSON header with
the complete starting position, funds, current player, turn, capture progress,
roster and deadline. This makes them portable without a separate map file.
The `OpenTurn` header preserves an unfinished final ply. Result `*` means the
game is still ongoing; `1-0`, `0-1`, and `1/2-1/2` mean Blue win, Red win, draw.
Results and actions are validated against the rules on import.

| Notation | Meaning |
|---|---|
| `a6+i` | Build infantry on a6 |
| `i: a6-b5x` | Move infantry a6 to b5 and capture |
| `i: a1x` | Capture without moving |
| `t: b1-c5xc6` | Move tank b1 to c5 and attack c6 |
| `i: a6-b5` | Move and wait |
| `i: a6` | Wait without moving |
| `--` | Empty turn (no unit actions or builds) |

Unit prefixes are optional when reading: the origin square identifies the unit.
Exporter prefixes are `i` infantry, `m` mech, `r` recon, `t` tank, `mt` medium
tank, `a` artillery, `rk` rocket, `aa` anti-air, `bc` battle copter, `f` fighter,
and `b` bomber. Squares use the same orientation as the GUI: a1 is bottom left.
An empty turn is written, for example, `4. --`. An End turn is otherwise implicit
at the end of each numbered line; the importer adds income/repairs exactly once.

This is a project-specific PGN-like format, not standard chess PGN or an AWBW
replay format. The initial header may be long; the numbered move text beneath
it remains human-readable.
