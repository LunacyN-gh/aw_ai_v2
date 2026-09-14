# Ordered-search experiment: interpretation

All 96 games completed, with 48 games per entrant and no errors or unfinished
games. All portable replay final states matched the recorded game hashes.
Budget: 2 seconds / 4,000 work units per turn, four CPU workers. Both neural
entrants used the same frozen game-275 best checkpoint from the 100% neural run.
No training or GUI defaults were changed.

| Entrant | W | L | D | Points / 48 |
|---|---:|---:|---:|---:|
| Heuristic ordered | 32 | 13 | 3 | 33.5 |
| Neural old | 26 | 20 | 2 | 27 |
| Heuristic old | 17 | 26 | 5 | 19.5 |
| Neural ordered | 16 | 32 | 0 | 16 |

New heuristic beats old heuristic 9–4–3 and old neural 10–6. New versus old
neural ties 8–8, with a strong map split: old wins 7/8 on standard maps;
new wins 7/8 on predeployed/two-base maps. New neural nonetheless loses 5–11
to old heuristic and 3–13 to new heuristic. Do not replace neural defaults.

This validates the backend as a promising heuristic opponent on this map pool,
not as a universally stronger neural search. The experiment changes proposal
generation and reply evaluation together: ordered neural uses one greedy neural
reply and no policy-preference penalty, whereas old bundle search can test both
neural and teacher replies and retains its policy preference. Thus the result
does not isolate action-order search as the cause of neural regressions.
The next useful ablation would put ordered proposals into the old neural
reply/scoring mechanism before training on their targets.

The 16-case tactical suite (same 2-second limit, after the tournament) selected
solutions in 12 cases for each heuristic, 1 for old neural and 4 for ordered
neural. Ordered heuristic selects all 12 breakthroughs but none of the four
HQ screens. Finding some correct plans without selecting them remains a
limitation. See tactical_results.json for case-level coverage and timing.

Eight implementation/tournament unit tests passed. These tactical fixtures
are small and deliberately constrained; success is not a general tactical
strength estimate. Wall-clock scheduling also makes tournament results variable.

Rerun instructions and algorithm scope: ../../ORDER_SEARCH.md.
