# Batched learning and parallel play

Implemented 2026-09-08. The learner now batches the CNN, entity transformer, action head, production head, and loss/backward pass. Feature arrays are prepared on CPU and transferred in bulk; action indexing uses bulk gathers. Boards are grouped by native height/width, and padded entity tokens are masked. The network dimensions, checkpoint schema, loss coefficients, and legal-action masks are unchanged.

Independent searched games and policy/beam evaluations run in persistent CPU processes, with one PyTorch thread per worker. The learner uses CUDA when available. The CLI defaults to `--device auto --workers 4 --batch-size 64`; Python `run()` defaults to one worker. CPU-only installations remain supported, using `--device cpu` if desired. NumPy is included in the neural optional dependencies for feature preparation.

## Measurements on this machine

RTX 4060 Laptop GPU, 8 GB VRAM; 16 logical CPU cores. Installed and verified PyTorch 2.10.0+cu126 in the v2 virtual environment; the previous installation was CPU-only. The existing NVIDIA driver supports this build. All comparisons below used the same installed PyTorch version.

The 835,474-parameter `cities_ita.pt` model and `test_cities_6x6.json` were used. These are component timings, not a forecast for an entire training run or evidence of stronger play.

| Work | Before/serial | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| Four optimizer updates, batch size 64, original scalar CPU versus batched CUDA | 4.00 s | 0.54 s | 7.4× |
| Same learner work, original scalar CPU versus batched CPU | 4.00 s | 1.11 s | 3.6× |
| Eight searched self-play games, serial versus four CPU workers | 92.14 s | 24.63 s | 3.7× |

Learner results are medians of three warm measurements, including CPU feature preparation, legal choices, losses, backward, clipping, and AdamW updates. Replay contained 161 searched teacher decisions from a 40-turn fixture; both implementations used identical sampling and optimizer settings. The original encoder and scalar fit were preserved temporarily for comparison. Encoder outputs, aggregate losses, and an SGD update matched within numerical tolerance. The new tests also cover mixed-size boards, entity padding, soft targets, gradients, and censored value-head behavior on CUDA.

The searched-game comparison used eight fixed seeds, twelve turns per game, 300 nodes per turn, and a generous 30-second deadline to compare equal search work. Both modes used the same frozen model and new CPU inference path. Decision counts and verified-candidate counts matched for every seed. Worker startup took another 3.91 seconds and is excluded from the warm parallel measurement. Separate spawn regression tests compare complete returned targets and state hashes with serial execution. Raw measurements are in `training_performance.json`.

At normal short wall-clock limits, CPU contention can change how many candidates finish; inspect `verified_candidates` and `partial_turns` as well as time. The implementation does not reduce search budgets to gain speed. Large boards or larger batches require more memory; reduce `--batch-size` if necessary.

## Running

From `C:\Users\apple\OneDrive\Desktop\AW_AI_v2`, a new ITA run:

```powershell
.\.venv\Scripts\python.exe -m aw_ai neural-train `
  --map maps/v1/test_cities_6x6.json --capture-limit 7 `
  --roster ita --bootstrap 50 --games 1000 --turns 150 `
  --device cuda --workers 4 --batch-size 64 `
  --train-seconds .15 --train-nodes 1000 `
  --eval-every 25 --eval-games 4 --eval-seconds 1 `
  --output models/cities_ita_fast.pt
```

To continue the existing model instead, add `--checkpoint models/cities_ita.pt` and choose the desired bootstrap count (often zero for continued self-play). Checkpoint loading restores weights and roster; optimizer state, replay, and the snapshot pool still start fresh, as before. Existing models remain compatible with GUI play.

## Scheduling and logs

Each round collects at most one game per worker using frozen learner weights. Results are ingested in seed order, and each completed game still receives the configured optimizer updates. Training can overlap outstanding games. Rounds stop at bootstrap, snapshot, evaluation, and checkpoint boundaries. With four workers, a collection policy can be up to three game updates behind the learner; `policy_after_game` and `policy_lag_games` record this. Different worker counts therefore need not produce identical training trajectories.

`play_seconds` includes worker model setup and searched play. Per-game `elapsed_seconds` is play plus training, which overlaps with other games. Use the `round` log's `elapsed_seconds` or cumulative `run_elapsed_seconds` to assess wall throughput. Evaluations log their own wall duration. Startup logs identify learner device, GPU, worker count, and batch size. Worker pools are closed on normal completion and exceptions.

The end-to-end smoke test loaded the existing ITA checkpoint, ran two bootstrap and five self-play games with four CPU workers and a CUDA learner, evaluated after games 2, 6, and 7, and saved both current and best checkpoints. Its deliberately short eight-turn games were censored; this checks the pipeline, not strength. Tests separately verify terminal value learning and preservation of the seven-property capture rule across processes.
