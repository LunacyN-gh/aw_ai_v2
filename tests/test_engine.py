import json
import random
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from aw_ai.arena import play, replay, terminal_samples
from aw_ai.budget import Budget
from aw_ai.evaluation import HeuristicValue, LinearValue, features
from aw_ai.model import Action, Board, END, State, Unit
from aw_ai.planner import Config, Planner
from aw_ai.policy import production_plans
from aw_ai.rules import Rules
from aw_ai.scenarios import digest, from_data, league_scale, tactical, to_data
from aw_ai.strategy import Strategist, interaction_regions


class RulesTests(unittest.TestCase):
    def setUp(self):
        self.rules = Rules()

    def test_friendly_transit_and_endpoints(self):
        s = State(Board.plain(5, 1), {"a": Unit("a", 0, "infantry", 0), "b": Unit("b", 0, "infantry", 1),
                                     "e": Unit("e", 1, "infantry", 4)})
        reach = self.rules.reachable(s, "a")
        self.assertIn(2, reach)
        self.assertNotIn(1, reach)
        self.assertNotIn(4, reach)

    def test_move_then_attack_clears_friendly_endpoint(self):
        s = tactical("blocker")
        attack = Action("attack", "tank", 2, "enemy")
        self.assertFalse(self.rules.is_legal(s, attack))
        s = self.rules.apply(s, Action("wait", "friend", 7))
        self.assertTrue(self.rules.is_legal(s, attack))
        s = self.rules.apply(s, attack)
        self.assertEqual(self.rules.outcome(s), 0)

    def test_clear_enemy_screen_opens_attack(self):
        s = tactical("screen")
        attack = Action("attack", "tank", 4, "target")
        self.assertFalse(self.rules.is_legal(s, attack))
        s = self.rules.apply(s, Action("attack", "gun", 0, "screen"))
        self.assertTrue(self.rules.is_legal(s, attack))

    def test_cache_survives_hp_and_friendly_move(self):
        s = tactical("blocker")
        self.rules.reachable(s, "tank")
        changed = s.clone()
        changed.update("tank", hp=90)
        changed.update("friend", pos=7)
        self.assertIn(2, self.rules.reachable(changed, "tank"))
        self.assertEqual(self.rules.cache_hits, 1)
        changed.remove("enemy")
        self.rules.reachable(changed, "tank")
        self.assertEqual(self.rules.cache_misses, 2)

    def test_capture_is_explicit_and_hq_is_terminal(self):
        b = Board.plain(3, 2, {(1, 0): "hq"})
        s = State(b, {"a": Unit("a", 0, "infantry", 1), "b": Unit("b", 1, "tank", 5)},
                  owners={1: 1}, captures={1: ("a", 10)})
        waited = self.rules.apply(s, Action("wait", "a", 1))
        self.assertNotIn(1, waited.captures)
        self.assertEqual(waited.owners[1], 1)
        captured = self.rules.apply(s, Action("capture", "a", 1))
        self.assertEqual(self.rules.outcome(captured), 0)
        self.assertEqual(captured.owners[1], 0)

    def test_income_repairs_and_acted_reset_once_on_end(self):
        b = Board.plain(4, 2, {(3, 1): "city", (2, 1): "tower"})
        s = State(b, {"a": Unit("a", 0, "tank", 0), "b": Unit("b", 1, "infantry", 7, 50, True)},
                  owners={7: 1, 6: 1})
        s = self.rules.apply(s, END)
        self.assertEqual((s.player, s.turn), (1, 1))
        self.assertEqual(s.funds[1], 800)
        self.assertEqual(s.units["b"].hp, 70)
        self.assertFalse(s.units["b"].acted)

    def test_airports_rosters_and_air_defense(self):
        b = Board.plain(5, 2, {(0, 0): "airport", (4, 1): "factory", (2, 0): "mountain"})
        s = State(b, {"a": Unit("a", 0, "b_copter", 1), "e": Unit("e", 1, "b_copter", 2)},
                  funds=[20000, 0], owners={0: 0, 9: 1})
        self.assertFalse(self.rules.is_legal(s, Action("build", destination=0, build="tank")))
        self.assertTrue(self.rules.is_legal(s, Action("build", destination=0, build="fighter")))
        self.assertEqual(self.rules.damage(s, s.units["a"], s.units["e"]), 65)
        built = self.rules.apply(s, Action("build", destination=0, build="fighter"))
        self.assertTrue(built.units[built.occupancy[0]].acted)
        self.assertEqual(built.funds[0], 0)

    def test_tower_damage_bonus(self):
        s = tactical("blocker")
        b = Board.plain(5, 2, {(4, 1): "tower"})
        s.board = b
        target = replace(s.units["enemy"], hp=100)
        a = self.rules.damage(s, s.units["tank"], target)
        s.owners[9] = 0
        self.assertGreater(self.rules.damage(s, s.units["tank"], target), a)

    def test_indirect_cannot_move_and_fire_or_counter(self):
        s = tactical("screen")
        self.assertFalse(self.rules.is_legal(s, Action("attack", "gun", 2, "screen")))
        self.assertTrue(all(a.destination == 0 for a in self.rules.attack_actions(s, "gun")))

    def test_reverse_terrain_distances(self):
        b = Board.plain(4, 1, {(1, 0): "forest"})
        self.assertEqual(self.rules.distances(b, "wheel", 3)[0], 7)
        self.assertEqual(self.rules.distances(b, "foot", 3)[0], 3)

    def test_illegal_action_does_not_mutate(self):
        s = tactical("screen")
        key = s.key()
        with self.assertRaises(ValueError):
            self.rules.apply(s, Action("attack", "tank", 4, "target"))
        self.assertEqual(s.key(), key)

    def test_random_generated_actions_match_checked_and_trusted_paths(self):
        rng = random.Random(9)
        for seed in range(4):
            state = league_scale(8, 4, seed, contact=True)
            for _ in range(20):
                legal = list(self.rules.legal_actions(state))
                if not legal:
                    break
                action = rng.choice(legal)
                key = state.key()
                checked = self.rules.apply(state, action)
                trusted = state.clone()
                self.rules.apply_inplace(trusted, action)
                self.assertEqual(digest(checked), digest(trusted))
                self.assertEqual(state.key(), key)
                checked.validate()
                state = checked


class PlannerTests(unittest.TestCase):
    def test_zero_budget_returns_legal_incumbent_and_rejects_stale(self):
        p = Planner(config=Config(seconds=0, nodes=0))
        s = tactical("screen")
        a = p.analyze(s)
        self.assertEqual(a.actions, (END,))
        self.assertTrue(a.metrics.exhausted)
        result = p.execute(s, a)
        self.assertEqual(result.player, 1)
        with self.assertRaisesRegex(ValueError, "stale"):
            p.execute(result, a)

    def test_planner_screen_sequence_and_input_immutable(self):
        p = Planner(config=Config(seconds=2, nodes=2000, continuation=False))
        s = tactical("screen")
        key = s.key()
        a = p.analyze(s)
        self.assertEqual(s.key(), key)
        attacks = [x for x in a.actions if x.kind == "attack"]
        self.assertTrue(any(x.target == "screen" for x in attacks), a.actions)
        self.assertTrue(any(x.target == "target" for x in attacks), a.actions)
        p.execute(s, a).validate()

    def test_planner_moves_blocker_before_tank_attack(self):
        p = Planner(config=Config(seconds=2, nodes=3000, beam=8, continuation=False))
        s = tactical("blocker")
        a = p.analyze(s)
        self.assertTrue(any(x.actor == "tank" and x.kind == "attack" for x in a.actions), a.actions)
        self.assertEqual(p.rules.outcome(p.execute(s, a)), 0)

    def test_continue_capture(self):
        p = Planner(config=Config(seconds=.5, nodes=1000, continuation=False))
        s = tactical("capture")
        a = p.analyze(s)
        self.assertTrue(any(x.actor == "cap" and x.kind == "capture" for x in a.actions), a.actions)

    def test_global_budget_and_production_width(self):
        s = league_scale(20, 0)
        s.funds[0] = 100000
        budget = Budget(10, 12)
        plans = production_plans(s, Rules(), budget, width=4)
        self.assertLessEqual(len(plans), 4)
        self.assertLessEqual(budget.used, 12)
        self.assertTrue(any(not actions for _, actions, _ in plans))
        for state, _, _ in plans:
            state.validate()
            self.assertGreaterEqual(state.funds[0], 0)

    def test_capture_assignment_does_not_send_everyone_to_same_city(self):
        b = Board.plain(8, 3, {(3, 0): "city", (3, 2): "city"})
        s = State(b, {"a": Unit("a", 0, "infantry", 0), "b": Unit("b", 0, "infantry", 16),
                      "e": Unit("e", 1, "tank", 15)})
        c = Strategist().plan(s, Rules())
        self.assertNotEqual(c.objectives["a"].destination, c.objectives["b"].destination)

    def test_separated_battles_form_separate_regions(self):
        b = Board.plain(40, 2)
        units = [Unit("a", 0, "infantry", 0), Unit("b", 1, "infantry", 2),
                 Unit("c", 0, "infantry", 37), Unit("d", 1, "infantry", 39)]
        groups, _ = interaction_regions(State(b, {u.id: u for u in units}))
        self.assertEqual(len(groups), 2)

    def test_twenty_by_twenty_returns_executable_turn_with_node_cap(self):
        p = Planner(config=Config(seconds=.4, nodes=120))
        s = league_scale(20, 12)
        a = p.analyze(s)
        p.execute(s, a).validate()
        self.assertLessEqual(a.metrics.counts["nodes"], 120)
        self.assertTrue(a.actions)

    def test_fixed_node_budget_is_deterministic(self):
        s = league_scale(8, 4, contact=True)
        config = Config(seconds=30, nodes=140)
        first = Planner(config=config).analyze(s)
        second = Planner(config=config).analyze(s)
        self.assertEqual(first.actions, second.actions)
        self.assertEqual(first.metrics.counts["nodes"], second.metrics.counts["nodes"])

    def test_nonfinite_time_rejected(self):
        for seconds in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                Config(seconds=seconds)


class PersistenceTests(unittest.TestCase):
    def test_state_roundtrip_includes_partial_turn(self):
        rules = Rules()
        s = tactical("screen")
        s = rules.apply(s, Action("attack", "gun", 0, "screen"))
        restored = from_data(to_data(s))
        self.assertEqual(digest(s), digest(restored))
        self.assertTrue(restored.units["gun"].acted)

    def test_v1_maps_import_without_v1_dependency(self):
        root = Path(__file__).resolve().parents[1] / "maps" / "v1"
        self.assertTrue(list(root.glob("*.json")))
        for path in root.glob("*.json"):
            from_data(json.loads(path.read_text(encoding="utf-8"))).validate()

    def test_duplicate_units_rejected(self):
        data = to_data(tactical("screen"))
        data["units"].append(data["units"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            from_data(data)

    def test_scale_fixture_counts(self):
        s = league_scale()
        self.assertEqual((s.board.width, s.board.height), (20, 20))
        self.assertEqual(s.board.tiles.count("city"), 32)
        for owner in (0, 1):
            self.assertEqual(sum(o == owner and s.board.tiles[p] == "factory" for p, o in s.owners.items()), 3)
            self.assertEqual(sum(o == owner and s.board.tiles[p] == "airport" for p, o in s.owners.items()), 1)
            self.assertEqual(sum(o == owner and s.board.tiles[p] == "tower" for p, o in s.owners.items()), 1)

    def test_replay_hash_and_tamper_detection(self):
        s = tactical("screen")
        record, final = play(s, [Planner(config=Config(seconds=.1, nodes=80)) for _ in range(2)], 4)
        self.assertEqual(digest(replay(record)), digest(final))
        record["turns"][0]["hash"] = "bad"
        with self.assertRaisesRegex(ValueError, "diverged"):
            replay(record)

    def test_truncated_games_not_training_labels(self):
        self.assertEqual(terminal_samples([{"termination": "turn_limit", "samples": [1]}]), [])

    def test_linear_value_fit_and_roundtrip(self):
        s = tactical("blocker")
        rules = Rules()
        model = LinearValue()
        x = features(s, 0, rules)
        loss = model.fit([(x, 1.)], epochs=20)
        self.assertLess(loss[-1], loss[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"value.json"
            model.save(path)
            self.assertEqual(model.weights, LinearValue.load(path).weights)


if __name__ == "__main__":
    unittest.main()
