import unittest

from bench import agents, compare, stats, store
from bench.seeding import DiceStream, derive_seed
from bench.store import metric


class StatsTests(unittest.TestCase):
    def test_wilson_matches_reference_values(self):
        lo, hi = stats.wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        self.assertEqual(stats.wilson(0, 0), (0.0, 1.0))

    def test_percentile_interpolates(self):
        self.assertEqual(stats.percentile([1, 2, 3, 4], 50), 2.5)
        self.assertEqual(stats.percentile([5], 95), 5)

    def test_paired_diff_cancels_shared_noise(self):
        base = [0, 1, 0, 1, 1, 0, 1, 0]
        mean, half = stats.paired_diff(base, base)
        self.assertEqual((mean, half), (0.0, 0.0))


class SeedingTests(unittest.TestCase):
    def test_streams_repeat_and_doubles_have_four_dice(self):
        s1, s2 = DiceStream(7), DiceStream(7)
        rolls = [s1.next() for _ in range(200)]
        self.assertEqual(rolls, [s2.next() for _ in range(200)])
        for roll in rolls:
            self.assertIn(len(roll), (2, 4))
            if len(roll) == 4:
                self.assertEqual(len(set(roll)), 1)

    def test_derive_seed_is_stable(self):
        self.assertEqual(derive_seed("pair", 3), derive_seed("pair", 3))
        self.assertNotEqual(derive_seed("pair", 3), derive_seed("pair", 4))


class AgentSpecTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(agents.parse("random"), ("random", None, {}))
        self.assertEqual(agents.parse("nn"), ("nn", agents.DEFAULT_NN, {}))
        self.assertEqual(agents.parse("nn:models/x.pt"), ("nn", "models/x.pt", {}))
        self.assertEqual(agents.parse("mcts:time=0.5,mode=heuristic"),
                         ("mcts", None, {"time": 0.5, "mode": "heuristic"}))
        self.assertEqual(agents.parse("py:pkg.mod:Cls,depth=2"), ("py", "pkg.mod:Cls", {"depth": 2}))
        with self.assertRaises(agents.SpecError):
            agents.parse("gnubg")

    def test_section_ignores_dots_in_specs(self):
        m = metric("strength[mcts:time=0.05 vs heuristic].win_rate", "x", 0.5)
        self.assertEqual(m["section"], "strength")


def _run(ident, started, label=None, metrics=(), host="h"):
    return {"id": ident, "started_at": started, "label": label, "metrics": list(metrics),
            "env": {"host": {"fingerprint": host}, "git": {}}, "config": {}}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.runs = [
            _run("20260101-000000-aaaaaaa", "2026-01-01T00:00:00", "baseline"),
            _run("20260102-000000-bbbbbbb", "2026-01-02T00:00:00"),
            _run("20260103-000000-bbbbbbc", "2026-01-03T00:00:00"),
        ]

    def test_resolve(self):
        self.assertEqual(store.resolve("latest", self.runs)["id"], self.runs[2]["id"])
        self.assertEqual(store.resolve("latest~2", self.runs)["id"], self.runs[0]["id"])
        self.assertEqual(store.resolve("baseline", self.runs)["id"], self.runs[0]["id"])
        self.assertEqual(store.resolve("20260102", self.runs)["id"], self.runs[1]["id"])
        with self.assertRaises(store.RefError):
            store.resolve("2026010", self.runs)  # ambiguous
        with self.assertRaises(store.RefError):
            store.resolve("latest~3", self.runs)


def _timing(value, lo, hi, floor=0.15):
    return metric("speed.x", "x", value, unit="ops/s", kind="timing", better="higher",
                  noise_floor=floor, spread={"min": lo, "max": hi})


class CompareTests(unittest.TestCase):
    def test_timing_within_noise_is_unchanged(self):
        row = compare.compare_metric(_timing(100, 98, 100), _timing(90, 88, 90))
        self.assertEqual(row["status"], "unchanged")

    def test_timing_beyond_floor_is_flagged(self):
        self.assertEqual(compare.compare_metric(_timing(100, 99, 100), _timing(70, 69, 70))["status"], "regressed")
        self.assertEqual(compare.compare_metric(_timing(100, 99, 100), _timing(130, 129, 130))["status"], "improved")

    def test_timing_on_different_hosts_is_not_judged(self):
        row = compare.compare_metric(_timing(100, 99, 100), _timing(50, 49, 50), same_host=False)
        self.assertEqual(row["status"], "info")

    def test_counts_follow_direction(self):
        a = metric("rules.x", "x", 3, kind="count", better="lower")
        b = metric("rules.x", "x", 0, kind="count", better="lower")
        self.assertEqual(compare.compare_metric(a, b)["status"], "improved")
        self.assertEqual(compare.compare_metric(b, a)["status"], "regressed")
        self.assertEqual(compare.compare_metric(a, a)["status"], "unchanged")

    def _rate(self, outcomes, pairing="1:100"):
        wins = outcomes.count("1")
        return metric("strength[a vs b].win_rate", "x", wins / len(outcomes), unit="fraction",
                      kind="rate", better="higher", n=len(outcomes), outcomes=outcomes, pairing=pairing)

    def test_paired_rates_detect_consistent_change(self):
        base = self._rate("10" * 100)
        head = self._rate("11" * 60 + "10" * 40)
        row = compare.compare_metric(base, head)
        self.assertEqual(row["status"], "improved")
        self.assertIn("paired", row["test"])

    def test_identical_outcomes_are_unchanged(self):
        base = self._rate("1101" * 50)
        self.assertEqual(compare.compare_metric(base, self._rate("1101" * 50))["status"], "unchanged")

    def test_unpaired_rates_need_larger_differences(self):
        base = self._rate("10" * 100, pairing="1:100")
        head = self._rate("1" * 106 + "0" * 94, pairing="2:100")
        row = compare.compare_metric(base, head)
        self.assertIn("unpaired", row["test"])
        self.assertEqual(row["status"], "unchanged")

    def test_unpaired_perfect_score_on_few_games_is_not_significant(self):
        # A 100% rate has zero sample variance; a normal interval would call this an improvement.
        base = self._rate("1" * 195 + "0" * 5, pairing="1:100")
        head = self._rate("1" * 10, pairing="1:5")
        row = compare.compare_metric(base, head)
        self.assertEqual(row["status"], "unchanged")
        self.assertAlmostEqual(row["diff"], 0.025)
        self.assertLess(row["diff_ci"][0], 0)

    def test_unpaired_rates_detect_large_differences(self):
        base = self._rate("10" * 100, pairing="1:100")
        head = self._rate("1" * 160 + "0" * 40, pairing="2:100")
        self.assertEqual(compare.compare_metric(base, head)["status"], "improved")
        self.assertEqual(compare.compare_metric(head, base)["status"], "regressed")

    def test_sections_missing_from_one_run_are_not_compared(self):
        full = _run("a", "1", metrics=[metric("rules.x", "x", 1), metric("speed.y", "y", 1.0)])
        partial = _run("b", "2", metrics=[metric("speed.y", "y", 1.0)])
        result = compare.compare_runs(full, partial)
        self.assertEqual([r["key"] for r in result["rows"]], ["speed.y"])
        self.assertTrue(any("rules" in w for w in result["warnings"]))

    def test_default_base_covers_head_sections(self):
        full = _run("a", "1", metrics=[metric("rules.x", "x", 1), metric("speed.y", "y", 1.0)])
        partial = _run("b", "2", metrics=[metric("speed.y", "y", 1.0)])
        head = _run("c", "3", metrics=[metric("rules.x", "x", 1), metric("speed.y", "y", 1.0)])
        for r in (full, partial, head):
            r["suite"] = "quick"
        self.assertEqual(store.default_base([full, partial, head], head)["id"], "a")
        self.assertEqual(store.default_base([full, partial, head], partial)["id"], "a")

    def test_new_and_removed_metrics(self):
        a = _run("a", "1", metrics=[metric("rules.old", "old", 1)])
        b = _run("b", "2", metrics=[metric("rules.new", "new", 1)])
        statuses = {r["key"]: r["status"] for r in compare.compare_runs(a, b)["rows"]}
        self.assertEqual(statuses, {"rules.new": "new", "rules.old": "removed"})


class EngineTests(unittest.TestCase):
    """Exercise the native board through the benchmark's game and rules code."""

    def test_seeded_games_are_reproducible(self):
        from bench.strength import play_game

        cache = {}
        first = play_game("random", "first", seed=123, move_limit=1000, cache=cache)
        self.assertEqual(first, play_game("random", "first", seed=123, move_limit=1000, cache=cache))
        self.assertIn(first[0], (1, -1))
        self.assertFalse(first[2])

    def test_rules_checks_on_opening_position(self):
        from bench import positions, rules

        opening = positions.snapshot(positions.to_board({
            "positions": [2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2],
            "turn": 1, "white_bar": 0, "black_bar": 0, "white_off": 0, "black_off": 0, "dice": [3, 1],
        }, roll=False), [3, 1], "opening", "t")
        result = rules.check_position(opening)
        self.assertGreater(result["distinct"], 0)
        self.assertEqual(result["leaks"], 0)
        self.assertEqual(result["refused"], 0)
        self.assertFalse(result["order"])

    def test_extensions_load_for_this_interpreter(self):
        from bench import env

        self.assertEqual(env.extension_problems(), [])

    def test_native_dice_can_be_seeded(self):
        from classes import board_cpp

        def rolls():
            board = board_cpp.Board()
            out = []
            for _ in range(20):
                out.append(tuple(board.roll_dice()[0]))
                board.move_from_sequence(board.valid_moves[0] if board.valid_moves else [])
            return out

        board_cpp.seed_dice(42)
        first = rolls()
        board_cpp.seed_dice(42)
        self.assertEqual(first, rolls())


class SuiteTests(unittest.TestCase):
    def test_suites_are_consistent(self):
        from bench import suites

        for name in suites.SUITES:
            cfg = suites.get(name)
            self.assertEqual(cfg["sections"], list(suites.SECTIONS))
            for spec in cfg["mcts_agents"]:
                self.assertEqual(agents.parse(spec)[0], "mcts", spec)
            for m in cfg["matches"]:
                agents.validate(m["a"])
                agents.validate(m["b"])

    def test_learning_check_is_off_in_quick(self):
        from bench import suites, training

        cfg = suites.get("quick")
        self.assertEqual(training.learning(cfg, log=lambda *a: None), [])


class SearchTests(unittest.TestCase):
    def test_counts_mcts_iterations(self):
        from bench import positions, search

        suite = positions.load()
        rate, per_decision = search.iterations("mcts:time=0.02", suite, repeats=1, limit=2, log=lambda *a: None)
        self.assertEqual(rate["section"], "search")
        self.assertGreater(rate["value"], 0)
        self.assertGreater(per_decision["value"], 0)
        with self.assertRaises(agents.SpecError):
            search.iterations("heuristic", suite, repeats=1, limit=1)


class TrainingTests(unittest.TestCase):
    def test_build_is_reproducible(self):
        import torch
        from bench import training

        model_cfg = {"hidden": [8], "lr": 0.1, "lambda": 0.7}
        a = training.build(model_cfg, seed=5)[0]
        b = training.build(model_cfg, seed=5)[0]
        for pa, pb in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(pa, pb))


if __name__ == "__main__":
    unittest.main()
