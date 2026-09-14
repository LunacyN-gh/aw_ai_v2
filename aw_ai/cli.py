from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics

from .arena import play, replay, save_replay, terminal_samples
from .evaluation import LinearValue
from .planner import Config, Planner, PolicyOnlyPlanner
from .scenarios import action_text, league_scale, load, save, tactical


def parser():
    p = argparse.ArgumentParser(description="Advance Wars AI v2 — deterministic research profile")
    p.add_argument("command", choices=("analyze", "play", "compare", "benchmark", "generate", "replay", "train", "gui", "neural-train", "neural-migrate", "tactical-eval"))
    p.add_argument("--map", type=Path)
    p.add_argument("--train-map", type=Path, action="append", help="repeat for each neural training map; also used for evaluation by default")
    p.add_argument("--scenario", choices=("screen", "blocker", "capture"))
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--units", type=int, default=12, help="units per side")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--contact", action="store_true")
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--nodes", type=int, default=4000)
    p.add_argument("--turns", type=int, default=100, help="player turns; neural training sets an income deadline (36 = 18 turns each)")
    p.add_argument("--turn-limit", type=int, help="absolute player-turn income deadline for loaded/generated states")
    p.add_argument("--output", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--checkpoint", type=Path, help="shared neural policy/value checkpoint")
    p.add_argument("--bootstrap", type=int, default=2)
    p.add_argument("--roster", choices=("all", "ita"), help="production roster; defaults to loaded checkpoint roster or all")
    p.add_argument("--opponent-checkpoint", type=Path, help="frozen baseline included in the training opponent cycle")
    p.add_argument("--train-seconds", type=float, default=.5)
    p.add_argument("--train-nodes", type=int, default=4000)
    p.add_argument("--eval-seconds", type=float, default=1.)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-games", type=int, default=4, help="even number of games per policy/beam evaluation")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--updates-per-game", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4, help="parallel CPU game workers; use 1 for serial play")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="learner device; workers use CPU")
    p.add_argument("--teacher-fraction", type=float, default=.5, help="minibatch share reserved for expert replay")
    p.add_argument("--teacher-every", type=int, default=2, help="query teacher every N learner turns; 0 disables queries")
    p.add_argument("--teacher-replay-size", type=int, default=8192)
    p.add_argument("--neural-value-weight", type=float, default=.5, help="guided search value blend; 0 uses heuristic value")
    p.add_argument("--search-mode", choices=("bundle","beam","mcts"), default="bundle", help="neural training candidate evaluator")
    p.add_argument("--teacher-search", choices=("old","ordered"), default="ordered")
    p.add_argument("--mcts-simulations", type=int, default=128)
    p.add_argument("--mcts-depth", type=int, default=6)
    p.add_argument("--mcts-c-puct", type=float, default=1.5)
    p.add_argument("--mcts-widening", type=float, default=2.)
    p.add_argument("--mcts-max-children", type=int, default=16)
    p.add_argument("--mcts-visit-temperature", type=float, default=1.)
    p.add_argument("--bundle-candidates", type=int, default=6)
    p.add_argument("--bundle-version", type=int, choices=(1,2), default=2)
    p.add_argument("--proposal-temperature", type=float, default=.7)
    p.add_argument("--policy-prior-weight", type=float, default=.1)
    p.add_argument("--eval-map", type=Path, action="append", help="repeat for each evaluation map; pair games by side")
    p.add_argument("--target-temperature", type=float, default=.15)
    p.add_argument("--exploration-fraction", type=float, default=.25)
    p.add_argument("--execution-exploration", type=float, default=.1)
    p.add_argument("--heuristic-scale", type=float, default=20000.)
    p.add_argument("--value-loss-weight", type=float, default=.5)
    p.add_argument("--entropy-weight", type=float, default=.01)
    p.add_argument("--promotion-win-rate", type=float, default=None, help="deprecated: maps threshold to margin by subtracting .5")
    p.add_argument("--promotion-margin", type=float, default=.05)
    p.add_argument("--promotion-teacher-weight", type=float, default=.5)
    p.add_argument("--replay-size", type=int, default=8192)
    p.add_argument("--capture-fraction", type=float, default=2/3, help="neural training income-property win fraction; 0 disables")
    p.add_argument("--capture-limit", type=int, help="income-property win count for generated/loaded maps")
    p.add_argument("--tiny", action="store_true", help="small network for pipeline smoke tests")
    p.add_argument("--prove", action="store_true", help="exhaustively verify tactical fixture uniqueness")
    p.add_argument("--search", choices=("beam", "tree", "mcts"), default="beam")
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--beam", type=int, default=4)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--replies", type=int, default=2)
    p.add_argument("--no-continuation", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "neural-migrate":
        from .neural import migrate_checkpoint
        if not args.checkpoint or not args.output:
            raise SystemExit("neural-migrate requires --checkpoint and a new --output path")
        migrate_checkpoint(args.checkpoint,args.output)
        print(f"Saved clock-aware checkpoint: {args.output.resolve()}")
        return
    if args.turn_limit is not None and args.turn_limit < 1:
        raise SystemExit("--turn-limit must be positive")
    if args.command == "neural-train" and args.turn_limit is not None:
        raise SystemExit("For neural-train use --turns to set the player-turn income deadline")
    if args.command == "train" and args.checkpoint:
        raise SystemExit("Use neural-train for neural checkpoints; train fits the legacy linear value model")
    if args.command == "neural-train":
        from .neural_training import run
        run(args.output or Path("models/shared.pt"),args.games,args.turns,args.seed,args.size,args.units,
            args.bootstrap,args.checkpoint,args.tiny,args.capture_fraction,args.map,args.capture_limit,
            roster=args.roster,opponent_checkpoint=args.opponent_checkpoint,
            train_seconds=args.train_seconds,train_nodes=args.train_nodes,eval_seconds=args.eval_seconds,
            eval_every=args.eval_every,eval_games=args.eval_games,checkpoint_every=args.checkpoint_every,
            updates_per_game=args.updates_per_game,batch_size=args.batch_size,replay_size=args.replay_size,
            workers=args.workers,device=args.device,teacher_fraction=args.teacher_fraction,
            teacher_every=args.teacher_every,teacher_replay_size=args.teacher_replay_size,
            neural_value_weight=args.neural_value_weight,promotion_win_rate=args.promotion_win_rate,search_mode=args.search_mode,
            bundle_candidates=args.bundle_candidates,target_temperature=args.target_temperature,
            exploration_fraction=args.exploration_fraction,execution_exploration=args.execution_exploration,
            heuristic_scale=args.heuristic_scale,value_loss_weight=args.value_loss_weight,entropy_weight=args.entropy_weight,
            bundle_version=args.bundle_version,proposal_temperature=args.proposal_temperature,
            policy_prior_weight=args.policy_prior_weight,eval_maps=args.eval_map,train_maps=args.train_map,
            promotion_margin=args.promotion_margin,promotion_teacher_weight=args.promotion_teacher_weight,
            teacher_search=args.teacher_search,mcts_simulations=args.mcts_simulations,mcts_depth=args.mcts_depth,
            mcts_c_puct=args.mcts_c_puct,mcts_widening=args.mcts_widening,mcts_max_children=args.mcts_max_children,
            mcts_visit_temperature=args.mcts_visit_temperature)
        return
    if args.command == "gui":
        from .gui import main as gui_main
        gui_main(turn_limit=args.turn_limit)
        return
    if args.command == "replay":
        if not args.map:
            raise SystemExit("replay requires --map replay.json")
        state = replay(json.loads(args.map.read_text(encoding="utf-8")))
        print(f"Replay verified through turn {state.turn}; winner={state.winner}")
        return
    value = LinearValue.load(args.model) if args.model else None
    policy = None
    if args.checkpoint:
        if args.model:
            raise SystemExit("Choose either --model or --checkpoint")
        from .neural import Network
        from .guidance import deployment_agent
        policy = value = deployment_agent(Network.load(args.checkpoint))
    config = Config(seconds=args.seconds, nodes=args.nodes, beam=args.beam, local_depth=args.depth,
                    candidates=args.candidates, replies=args.replies, continuation=not args.no_continuation)
    if args.search == "mcts":
        if not args.checkpoint:raise SystemExit('--search mcts requires --checkpoint')
        from .bundle_mcts import MCTSPlanner
        planner_type = MCTSPlanner
    elif args.search == "tree":
        from .tree import SampledPlanner
        planner_type = SampledPlanner
    else:
        planner_type = Planner
    if args.command == "tactical-eval":
        from .tactical_suite import evaluate_suite
        report = evaluate_suite(lambda: planner_type(config=config,policy=policy,value=value),prove=args.prove)
        print(json.dumps(report,indent=2))
        if args.output: args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        return
    state = load(args.map) if args.map else tactical(args.scenario) if args.scenario else league_scale(
        args.size, args.units, args.seed, args.contact)
    if args.turn_limit is not None:
        state.turn_limit = args.turn_limit
        state.validate()
    if args.capture_limit is not None:
        state.income_capture_limit = args.capture_limit
        state.validate()
    if args.command == "generate":
        path = args.output or Path("map.json")
        save(state, path)
        print(f"Saved {path.resolve()}")
    elif args.command == "analyze":
        planner = planner_type(config=config, policy=policy, value=value)
        analysis = planner.analyze(state)
        planner.execute(state, analysis)  # Validate returned ordered plan.
        for action in analysis.actions:
            print(action_text(action, state.board))
        units='MCTS Q' if hasattr(analysis,'visit_weights') else 'bounded tree value' if args.search=='tree' else 'search score'
        print(f"Score: {analysis.score:.3f} ({units}; not win probability)")
        print(json.dumps(analysis.metrics.as_dict(), indent=2))
        if args.output:
            data = {"actions": [asdict(a) for a in analysis.actions], "score": analysis.score,
                    "metrics": analysis.metrics.as_dict(), "objectives": {k: asdict(v) for k, v in analysis.objectives.items()},
                    "alternatives": [{"actions": [asdict(a) for a in c.actions], "score": c.score,
                                      "reply": [asdict(a) for a in c.reply], "reason": c.reason,
                                      "verified": c.verified, "complete": c.complete} for c in analysis.alternatives]}
            args.output.write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    elif args.command == "play":
        record, final = play(state, [planner_type(config=config, policy=policy, value=value), planner_type(config=config, policy=policy, value=value)], args.turns)
        path = args.output or Path("replay.json")
        save_replay(record, path)
        replay(record)
        print(f"{len(record['turns'])} turns; {record['termination']}; winner={record['result']}; {path.resolve()}")
    elif args.command == "benchmark":
        rows = []
        for size, count, contact in ((8, 4, False), (20, 12, False), (20, 24, True), (24, 40, True)):
            times = []
            metrics = []
            for seed in range(args.repeats):
                s = league_scale(size, count, seed, contact)
                if args.capture_limit is not None:
                    s.income_capture_limit = args.capture_limit
                    s.validate()
                planner = planner_type(config=config, policy=policy, value=value)
                a = planner.analyze(s)
                planner.execute(s, a)
                times.append(a.metrics.elapsed)
                metrics.append(a.metrics.as_dict())
            if not times:
                raise SystemExit("--repeats must be positive")
            ordered = sorted(times)
            row = {"size": size, "units_per_side": count, "contact": contact,
                   "p50": statistics.median(times), "p95": ordered[min(len(ordered)-1, int(.95*len(ordered)))],
                   "runs": metrics}
            rows.append(row)
            print(f"{size}x{size}, {count}/side, contact={contact}: median={row['p50']:.3f}s, p95={row['p95']:.3f}s")
        if args.output:
            args.output.write_text(json.dumps(rows, indent=2)+"\n", encoding="utf-8")
    elif args.command == "compare":
        rows = []
        for game in range(args.games):
            initial = state if args.map or args.scenario else league_scale(args.size, args.units, args.seed+game//2, args.contact)
            if args.turn_limit is not None:
                initial.turn_limit = args.turn_limit
            if args.capture_limit is not None:
                initial.income_capture_limit = args.capture_limit
                initial.validate()
            planners = [planner_type(config=config, policy=policy, value=value), PolicyOnlyPlanner(config=config)]
            search_side = game % 2
            if search_side:
                planners.reverse()
            record, _ = play(initial, planners, args.turns)
            replay(record)
            row = {"game": game, "search_side": search_side, "winner": record["result"],
                   "termination": record["termination"], "turns": len(record["turns"])}
            rows.append(row)
            print(json.dumps(row), flush=True)
        result = {"search_wins": sum(r["termination"] == "victory" and r["winner"] == r["search_side"] for r in rows),
                  "policy_wins": sum(r["termination"] == "victory" and r["winner"] != r["search_side"] for r in rows),
                  "draws": sum(r["termination"] == "draw" for r in rows),
                  "censored": sum(r["termination"] not in ("victory","draw") for r in rows), "games": rows,
                  "config": asdict(config)}
        if args.output:
            args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    elif args.command == "train":
        records = []
        for game in range(args.games):
            initial = state if args.map or args.scenario else league_scale(args.size, args.units, args.seed+game, args.contact)
            if args.turn_limit is not None:
                initial.turn_limit = args.turn_limit
            if args.capture_limit is not None:
                initial.income_capture_limit = args.capture_limit
                initial.validate()
            record, _ = play(initial, [planner_type(config=config, value=value), Planner(config=config)], args.turns)
            records.append(record)
            print(f"game {game+1}: {record['termination']}", flush=True)
        samples = terminal_samples(records)
        if not samples:
            raise SystemExit("No terminal outcomes; refusing to train on turn-limit labels. Use tactical scenarios or more turns.")
        model = value or LinearValue()
        losses = model.fit(samples, epochs=args.epochs)
        path = args.output or Path("value.json")
        model.save(path)
        print(f"Fitted {len(samples)} positions; training MSE {losses[0]:.4f} -> {losses[-1]:.4f}. Saved {path}")


if __name__ == "__main__":
    main()
