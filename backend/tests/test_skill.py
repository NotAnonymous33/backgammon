"""Tests for the skill evaluation: bearoff truth, reference labels, decision scoring and ratings."""
import random
import unittest

import numpy as np

from bench import agents, bearoff, compare, ratings, reference, skill
from bench.store import metric

DEFAULT_NN = agents.resolve_path(agents.DEFAULT_NN)
TDGAMMON = agents.resolve_path("models/tdgammon/tdgammon_checkpoint_latest.pt")


def _board(white, black, turn=1):
    from classes.board_cpp import Board  # type: ignore

    return Board({"positions": bearoff.to_positions(white, black), "dice": "", "rolled": False, "turn": turn,
                  "white_bar": 0, "black_bar": 0, "white_off": 15 - sum(white), "black_off": 15 - sum(black),
                  "game_over": False})


class BearoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = bearoff.BearoffDB.build(4)

    def test_hand_calculated_probabilities(self):
        p = self.db.probability
        ace = (1, 0, 0, 0, 0, 0)
        self.assertAlmostEqual(p(ace, ace), 1.0)
        # Two checkers on the ace come off with any roll.
        self.assertAlmostEqual(p((2, 0, 0, 0, 0, 0), ace), 1.0)
        # Three on the ace need doubles, else the opponent finishes first.
        self.assertAlmostEqual(p((3, 0, 0, 0, 0, 0), ace), 1 / 6)
        # One checker on the six-point needs 6 pips: fails on 21, 31, 41, 32 and 11.
        self.assertAlmostEqual(p((0, 0, 0, 0, 0, 1), ace), 27 / 36)

    def test_probabilities_are_valid_and_extra_checkers_never_help(self):
        win = self.db.win
        self.assertTrue(((win >= -1e-12) & (win <= 1 + 1e-12)).all())
        rng = random.Random(3)
        configs = [c for c in self.db.configs if 0 < sum(c) < self.db.max_checkers]
        for _ in range(300):
            a, b = rng.choice(configs), rng.choice(configs)
            k = rng.randrange(6)
            more = tuple(n + (i == k) for i, n in enumerate(a))
            self.assertLessEqual(self.db.probability(more, b), self.db.probability(a, b) + 1e-12)

    def test_move_generation_matches_native_board(self):
        rng = random.Random(1)
        configs = self.db.configs[1:]
        for _ in range(150):
            white, black = rng.choice(configs), rng.choice(configs)
            for roll, _ in bearoff.ROLLS:
                board = _board(white, black)
                board.set_dice(list(roll) * (2 if roll[0] == roll[1] else 1))
                native = set()
                for seq in board.valid_moves:
                    child = board.clone()
                    child.move_from_sequence(seq)
                    native.add(bearoff.sides(child.positions)[0])
                self.assertEqual(native, bearoff.play_roll(white, roll), (white, roll))

    def test_board_mapping(self):
        white, black = (1, 0, 2, 0, 0, 1), (0, 3, 0, 0, 0, 0)
        board = _board(white, black, turn=-1)
        self.assertEqual(self.db.board_state(board), (white, black))
        self.assertAlmostEqual(self.db.white_win(board, -1), 1 - self.db.probability(black, white))
        from classes.board_cpp import Board  # type: ignore

        self.assertIsNone(self.db.board_state(Board()))


class ValueNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from bench import positions

        cls.boards = [positions.to_board(p) for p in positions.load()["positions"]]

    def test_features_match_the_training_code(self):
        from bench import valuenet
        from classes.agents.NNAgent import extract_features

        for board in self.boards:
            self.assertEqual(valuenet.features(board), extract_features(board))

    def test_decisions_match_final_nn_agent(self):
        from bench import valuenet
        from classes.agents.NNAgent import FinalNNAgent

        ref = FinalNNAgent(checkpoint_path=DEFAULT_NN)
        fast = valuenet.FastNNAgent(valuenet.ValueNet.from_checkpoint(DEFAULT_NN))
        for board in self.boards:
            if not board.valid_moves:
                continue
            chosen = []
            for agent in (ref, fast):
                child = board.clone()
                child.move_from_sequence(agent.select_move(board.clone()))
                chosen.append(valuenet.features(child))
            # Different sequences reaching identically featured positions tie;
            # torch breaks such ties by float noise.
            self.assertEqual(chosen[0], chosen[1])

    def test_architecture_is_read_from_the_checkpoint(self):
        from bench import valuenet
        from classes.agents.NNAgent import FinalNNAgent

        self.assertEqual(valuenet.ValueNet.from_checkpoint(TDGAMMON).hidden, [40])
        self.assertEqual(valuenet.ValueNet.from_checkpoint(DEFAULT_NN).hidden, [128, 128])
        self.assertEqual(FinalNNAgent(checkpoint_path=TDGAMMON).model.model[0].out_features, 40)

    def test_rejects_other_state_dicts(self):
        import torch
        from bench import valuenet

        with self.assertRaises(ValueError):
            valuenet.layer_shapes({"model.0.weight": torch.zeros(4, 10), "model.0.bias": torch.zeros(4)})
        with self.assertRaises(ValueError):
            valuenet.layer_shapes({"fc.weight": torch.zeros(1, 34)})


def _entry(values, turn=1, category="contact", halves=None):
    labels = {}
    for i, v in enumerate(values):
        labels[f"a{i}"] = {"p": v, "seq": []}
        if halves:
            labels[f"a{i}"]["halves"] = halves[i]
    return {"id": f"p{len(values)}{turn}{category}", "category": category, "turn": turn, "afterstates": labels}


class ScoringTests(unittest.TestCase):
    def test_losses_are_from_the_movers_point_of_view(self):
        white = _entry([0.6, 0.5, 0.2], turn=1)
        self.assertEqual(skill.losses(white, "a0"), (0.0, True))
        self.assertAlmostEqual(skill.losses(white, "a2")[0], 0.4)
        black = _entry([0.6, 0.5, 0.2], turn=-1)  # black wants white's chances low
        self.assertEqual(skill.losses(black, "a2"), (0.0, True))
        self.assertAlmostEqual(skill.losses(black, "a0")[0], 0.4)
        self.assertIsNone(skill.losses(white, "missing"))

    def test_noise_floor(self):
        exact = _entry([0.7, 0.3])
        self.assertEqual(skill.noise_floor([exact]), 0.0)
        # Each half prefers a different play, so each choice loses 0.2 on the other half.
        noisy = _entry([0.5, 0.5], halves=[[0.6, 0.4], [0.4, 0.6]])
        self.assertAlmostEqual(skill.noise_floor([noisy]), 0.2)

    def test_agent_metrics(self):
        entries = [_entry([0.6, 0.5], category="contact"), _entry([0.9, 0.2, 0.1], category="exact")]
        choices = {entries[0]["id"]: ["a1"], entries[1]["id"]: ["a0", "a2"]}
        ms = {m["key"]: m for m in skill.agent_metrics("x", entries, choices, [0.001], [], "pair")}
        self.assertAlmostEqual(ms["skill[x].loss.contact"]["value"], 0.1)
        self.assertAlmostEqual(ms["skill[x].loss.exact"]["value"], 0.4)  # mean of 0 and 0.8
        self.assertAlmostEqual(ms["skill[x].loss"]["value"], 0.25)
        self.assertEqual(ms["skill[x].loss"]["samples"], [0.1, 0.4])
        self.assertAlmostEqual(ms["skill[x].best_rate"]["value"], 0.25)
        self.assertAlmostEqual(ms["skill[x].blunder_rate"]["value"], 0.75)

    def test_repeats_only_for_stochastic_agents(self):
        self.assertEqual(skill.repeats_for("nn", 3), 1)
        self.assertEqual(skill.repeats_for("heuristic", 3), 1)
        self.assertEqual(skill.repeats_for("mcts:iters=50", 3), 3)
        self.assertEqual(skill.repeats_for("random", 3), 3)


class ReferenceTests(unittest.TestCase):
    def test_afterstates_are_distinct(self):
        from bench import positions

        board = positions.to_board(dict(positions.load()["positions"][1]))
        found = reference.afterstates(board)
        self.assertLessEqual(len(found), len(board.valid_moves))
        self.assertEqual(len({reference.afterstate_id(c) for _, c in found.values()}), len(found))

    def test_rollout_stops_at_the_bearoff_database(self):
        db = bearoff.BearoffDB.build(2)
        board = _board((1, 1, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), turn=-1)
        value = reference.rollout(board, None, policy=None, db=db)
        self.assertAlmostEqual(value, db.white_win(board, -1))

    def test_build_labels_every_play(self):
        from bench import positions

        db = bearoff.BearoffDB.load()
        exact = positions.snapshot(_board((1, 0, 1, 0, 0, 1), (2, 0, 0, 1, 0, 0)), [6, 2], "exact", "t-exact")
        opening = positions.snapshot(positions.to_board(positions.load()["positions"][1], roll=False),
                                     [6, 5], "opening", "t-open")
        doc = reference.build([exact, opening], db, stages=(4, 8), workers=1, log=lambda *a: None)
        by_id = {p["id"]: p for p in doc["positions"]}
        self.assertTrue(all(label["exact"] for label in by_id["t-exact"]["afterstates"].values()))
        rolled = by_id["t-open"]["afterstates"].values()
        self.assertTrue(all(label["n"] in (4, 8) and 0 <= label["p"] <= 1 for label in rolled))
        self.assertTrue(any(label["n"] == 8 for label in rolled))  # the best play was topped up
        # Common random numbers: identical inputs give identical labels.
        again = reference.build([exact, opening], db, stages=(4, 8), workers=1, log=lambda *a: None)
        self.assertEqual(reference.fingerprint(doc), reference.fingerprint(again))


class RatingTests(unittest.TestCase):
    def test_fit_recovers_ratings(self):
        truth = np.array([1500.0, 1700.0, 1200.0])
        n = 4000
        games = np.full((3, 3), n, dtype=float)
        np.fill_diagonal(games, 0)
        wins = games * ratings.expected(truth[:, None], truth[None, :])
        r, se = ratings.fit(3, 0, 1500, wins, games)
        np.testing.assert_allclose(r, truth, atol=2)
        self.assertEqual(se[0], 0)
        self.assertTrue((se[1:] > 0).all())

    def test_perfect_scores_stay_finite_and_are_flagged(self):
        ladder = ratings.Ladder(["a", "b"], "a", 1500)
        ladder.matchups[(0, 1)] = [(1.0, 1.0)] * 20
        r, _ = ladder.refit()
        self.assertTrue(np.isfinite(r).all())
        self.assertLess(r[1], 1500)
        self.assertEqual(ladder.bounds(), {0: "lower", 1: "upper"})

    def test_scheduler_prefers_close_matchups(self):
        ladder = ratings.Ladder(["anchor", "close", "far"], "anchor", 1500)
        ladder.matchups[(0, 1)] = [(1.0, 0.0)] * 10        # even
        ladder.matchups[(0, 2)] = [(1.0, 1.0)] * 20        # lopsided: no information
        ladder.matchups[(1, 2)] = [(1.0, 1.0)] * 9 + [(1.0, 0.0)]
        ladder.refit()
        schedule = ladder.next_round(target_se=1.0, max_pairs=100, batch_pairs=10)
        self.assertIn((0, 1), schedule)
        self.assertNotIn((0, 2), schedule)

    def test_bootstrap_interval_covers_estimate(self):
        rng = random.Random(0)
        ladder = ratings.Ladder(["a", "b"], "a", 1500)
        p = ratings.expected(1600, 1500)
        ladder.matchups[(0, 1)] = [(float(rng.random() > p), float(rng.random() > p)) for _ in range(300)]
        r, _ = ladder.refit()
        draws = ladder.bootstrap(samples=100)
        lo, hi = np.percentile(draws[:, 1], [2.5, 97.5])
        self.assertLess(lo, r[1])
        self.assertGreater(hi, r[1])
        self.assertLess(lo, 1600)
        self.assertGreater(hi, 1600)


class MetricCompareTests(unittest.TestCase):
    def _mean(self, samples, pairing="ref:2"):
        return metric("skill[x].loss", "x", sum(samples) / len(samples), unit="fraction", kind="mean",
                      better="lower", samples=samples, pairing=pairing,
                      ci=[sum(samples) / len(samples) - 0.01, sum(samples) / len(samples) + 0.01])

    def test_paired_means(self):
        base = self._mean([0.02, 0.05, 0.01, 0.04] * 25)
        head = self._mean([0.01, 0.04, 0.00, 0.03] * 25)
        row = compare.compare_metric(base, head)
        self.assertEqual(row["status"], "improved")
        self.assertIn("paired", row["test"])
        self.assertEqual(compare.compare_metric(base, base)["status"], "unchanged")

    def test_unpaired_means_use_intervals(self):
        base = self._mean([0.03] * 10, pairing="a")
        head = self._mean([0.025] * 10, pairing="b")
        row = compare.compare_metric(base, head)
        self.assertIn("unpaired", row["test"])
        self.assertEqual(row["status"], "unchanged")

    def test_ratings(self):
        a = metric("ratings[x].elo", "x", 1600, unit="Elo", kind="rating", better="higher", ci=[1570, 1630])
        b = metric("ratings[x].elo", "x", 1700, unit="Elo", kind="rating", better="higher", ci=[1670, 1730])
        c = metric("ratings[x].elo", "x", 1620, unit="Elo", kind="rating", better="higher", ci=[1590, 1650])
        self.assertEqual(compare.compare_metric(a, b)["status"], "improved")
        self.assertEqual(compare.compare_metric(a, c)["status"], "unchanged")


class MctsBudgetTests(unittest.TestCase):
    def test_iteration_budget_is_exact_and_reproducible(self):
        from bench.strength import play_game

        agent = agents.make_agent("mcts:iters=30", {})
        from classes.board_cpp import Board  # type: ignore

        board = Board()
        board.set_dice([6, 4])
        agent.select_move(board)
        self.assertEqual(agent.mcts.sim_count, 30)
        cache = {}
        first = play_game("mcts:iters=20", "random", seed=7, move_limit=1000, cache=cache)
        self.assertEqual(first, play_game("mcts:iters=20", "random", seed=7, move_limit=1000, cache=cache))

    def test_unknown_mcts_keys_are_rejected(self):
        with self.assertRaises(agents.SpecError):
            agents.parse("mcts:iterations=20")


if __name__ == "__main__":
    unittest.main()
