# Implementation validation — September 7, 2026

This file records the initial heuristic implementation. The later production, neural, tactical, and capture-limit stage is documented in [NEURAL_TACTICAL.md](NEURAL_TACTICAL.md); its results supersede the earlier stage's status and remaining-work list below.

This is an engineering smoke test and regression report, not evidence of competitive AWBW strength.

- **31 engine/search/persistence tests passed.** Includes randomized checked-versus-trusted transitions, ordered-action dependencies, capture continuation, budget limits, deterministic fixed-node search, state/replay serialization, v1 map import, learning and sampled tree search.
- **4 Tk integration tests passed.** Constructed the board/sidebar, executed human moves and an attack, generated/edited a map, and completed background analysis without mutating the board. Test windows were withdrawn. Tk initialization is blocked by the execution sandbox on this host; these checks passed outside it.
- **12-turn, 20×20 self-play replay verified** from the recorded initial state through every per-turn SHA-256 digest. The game reached its configured turn limit and was not labeled as a training draw.
- **Training smoke test passed:** two tiny tactical games produced terminal labels, fitted a linear value model, and saved a reloadable file. This is a plumbing check, not a generally useful trained model.
- **Comparison harness smoke test passed:** four side-swapped tactical games against the policy-only ablation, with replay verification. The initially advantaged board side won all four; the agents split wins 2–2. This does not distinguish playing strength.
- Original v1 sources, tests, maps and examples matched their pre-implementation SHA-256 hashes. Pre-existing working-tree modifications were left intact. No commit was created.

## Final scale check

One-second nominal budget, 4,000-node maximum, default beam planner, three generated terrain seeds per fixture. Each returned plan was executed through checked rules and validated. Runs were performed sequentially on this host. Times vary with machine load; the 95th-percentile column is just the maximum of three samples and is not a statistically reliable tail estimate.

| Board | Units per side | Contact | Median time | Observed p95 | Completed candidates / searched replies |
| --- | ---: | --- | ---: | ---: | --- |
| 8×8 | 4 | No | 0.085 s | 0.091 s | Complete own turn; three-turn comparison |
| 20×20 | 12 | No | 0.136 s | 0.146 s | Complete own turn; three-turn comparison |
| 20×20 | 24 | Yes | 1.007 s | 1.008 s | Two verified candidate turns; two-turn comparison |
| 24×24 | 40 | Yes | 1.013 s | 1.016 s | One verified candidate turn; two-turn comparison |

The final runs retained no incomplete own-turn candidates. The densest fixture did not have budget left for local beam expansion, so it used the policy-generated complete turn and an opponent reply. Meeting the latency budget does not imply equal search quality across scales. Earlier diagnostic reports show this explicitly: before reducing repeated threat calculations and separating rollout actor breadth from tactical search breadth, some dense runs returned incomplete turns or had no completed replies.

Raw final results are in `benchmark_final.json`. The other benchmark JSON files are earlier development diagnostics, not the final measurements.

## Remaining validation

Full AWBW damage/rounding and mechanics conformance, held-out map-family tournaments, calibrated value estimates, direct v1 comparisons under a common rules profile, neural policy/value training, stochastic play and fog remain outstanding. The supplied map fixtures match the requested scale but are synthetic rather than downloaded league maps.
