"""Tests for the expectiminimax agent: features, greedy agreement, and search against a naive full-width tree."""
import unittest

import numpy as np

from bench import agents, reference
from bench.positions import to_board
from bench.valuenet import FastNNAgent, ValueNet
from classes.agents import features
from classes.agents.Expectiminimax import ROLLS, ExpectiminimaxAgent, _encode_batch, _key

DEFAULT_NN = agents.resolve_path(agents.DEFAULT_NN)
SHIPPED_NN = agents.resolve_path("models/main/main_checkpoint_latest.pt")


def _naive(net, board, depth):
    """P(side to roll wins), searching every reply to every roll ``depth`` moves deep."""
    if depth == 0:
        return net.schema.win_probability(board, float(net.evaluate([net.schema.features(board)])[0]), board.turn)
    total = 0.0
    for dice, p in ROLLS:
        rolled = board.clone()
        rolled.set_dice(dice)
        children = [child for _, child in reference.afterstates(rolled).values()]
        if not children:  # no legal move: the turn passes
            rolled.move_from_sequence([])
            children = [rolled]
        total += p * max(_naive_after(net, child, depth - 1) for child in children)
    return total


def _naive_after(net, child, depth):
    """An afterstate's value to the side that just moved."""
    return 1.0 if child.game_over else 1.0 - _naive(net, child, depth)


def _played(board, move):
    child = board.clone()
    assert child.move_from_sequence(move) is not False
    return reference.afterstate_id(child)


class ExpectiminimaxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = ValueNet.from_checkpoint(DEFAULT_NN)
        cls.entries = reference.load()["positions"]

    def boards(self, category, n):
        return [to_board(e) for e in self.entries if e["category"] == category][:n]

    def test_rolls_cover_every_outcome(self):
        self.assertEqual(len(ROLLS), 21)
        self.assertAlmostEqual(sum(p for _, p in ROLLS), 1.0)

    def test_batch_features_match_schemas(self):
        kids = [c for e in self.entries[::7] for _, c in reference.afterstates(to_board(e)).values() if not c.game_over]
        for schema in (features.CANONICAL, features.LEGACY):
            expected = np.array([schema.features(k) for k in kids], dtype=np.float32)
            np.testing.assert_array_equal(_encode_batch([_key(k) for k in kids], schema.canonical), expected)

    def test_one_ply_is_the_greedy_network(self):
        for path in (DEFAULT_NN, SHIPPED_NN):
            net = ValueNet.from_checkpoint(path)
            greedy, emm = FastNNAgent(net), ExpectiminimaxAgent(net, ply=1)
            for board in [to_board(e) for e in self.entries[::5]]:
                self.assertEqual(_played(board, emm.select_move(board.clone())),
                                 _played(board, greedy.select_move(board.clone())), path)

    def assert_matches_naive(self, boards, ply):
        # Without pruning the search must find a play as good as the full tree's best.
        agent = ExpectiminimaxAgent(self.net, ply=ply, width=1000, margin=1.0, reply_width=1000)
        for board in boards:
            values = {aid: _naive_after(self.net, child, ply - 1) for aid, (_, child) in reference.afterstates(board).items()}
            chosen = _played(board, agent.select_move(board.clone()))
            self.assertAlmostEqual(values[chosen], max(values.values()), places=6)

    def test_two_ply_matches_full_width_search(self):
        self.assert_matches_naive(self.boards("contact", 3) + self.boards("bar", 2), ply=2)

    def test_three_ply_matches_full_width_search(self):
        self.assert_matches_naive(self.boards("exact", 3), ply=3)

    def test_spec_builds_a_legal_deterministic_agent(self):
        self.assertEqual(agents.parse("emm:ply=3,reply=2"), ("emm", None, {"ply": 3, "reply": 2}))
        with self.assertRaises(agents.SpecError):
            agents.parse("emm:depth=2")
        cache = {}
        board = self.boards("contact", 1)[0]
        first = agents.make_agent("emm:ply=2", cache).select_move(board.clone())
        self.assertEqual(first, agents.make_agent("emm:ply=2", cache).select_move(board.clone()))
        self.assertIn(first, board.valid_moves)
        self.assertIn(agents.DEFAULT_NN, agents.describe("emm:ply=2")["checkpoints"])


if __name__ == "__main__":
    unittest.main()
