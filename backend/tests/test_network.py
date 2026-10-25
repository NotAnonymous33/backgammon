"""Tests for the canonical value network: features, TD(λ) targets, checkpoints and agents."""
import copy
import os
import random
import tempfile
import unittest

import numpy as np
import torch

from classes.agents import features as F
from classes.agents.NNAgent import (DEFAULT_CHECKPOINT, BackgammonNN, BackgammonTrainer, FinalNNAgent, NNAgent,
                                    TDLambda, extract_features, hidden_sizes_of, load_network)
from classes.board_cpp import Board  # type: ignore

LEGACY_NN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models/main/main_checkpoint_latest.pt")


def suite_boards(roll=True):
    from bench import positions

    return [(p, positions.to_board(p, roll=roll)) for p in positions.load()["positions"]]


def mirror_dict(p):
    """The same position with the colours swapped."""
    return {"positions": [-x for x in reversed(p["positions"])], "turn": -p["turn"],
            "white_bar": p["black_bar"], "black_bar": p["white_bar"],
            "white_off": p["black_off"], "black_off": p["white_off"], "dice": list(p["dice"])}


def canonical_checkpoint(path, hidden=(16,), seed=0):
    torch.manual_seed(seed)
    model = BackgammonNN(F.CANONICAL.size, hidden_sizes=list(hidden))
    trainer = BackgammonTrainer(model, F.CANONICAL, TDLambda(model), games_per_epoch=1, eval_games=0, num_workers=1)
    trainer.save_checkpoint(path, epoch=1)
    return model


class FeatureTests(unittest.TestCase):
    def test_legacy_schema_matches_original_features(self):
        for _, board in suite_boards(roll=False):
            self.assertEqual(F.LEGACY.features(board), extract_features(board))

    def test_canonical_equals_legacy_with_white_on_roll(self):
        for _, board in suite_boards(roll=False):
            self.assertEqual(F.CANONICAL.encode(board, 1), extract_features(board))

    def test_canonical_features_ignore_colour(self):
        from bench import positions

        for p, board in suite_boards(roll=False):
            mirrored = positions.to_board(mirror_dict(p), roll=False)
            self.assertEqual(F.CANONICAL.features(board), F.CANONICAL.features(mirrored))
            # ...and the side on roll matters: the two views differ unless the position is symmetric.
            if board.positions != [-x for x in reversed(board.positions)]:
                self.assertNotEqual(F.CANONICAL.encode(board, 1), F.CANONICAL.encode(board, -1))

    def test_schema_descriptions_round_trip(self):
        for schema in (F.LEGACY, F.CANONICAL):
            self.assertIs(F.from_description(schema.describe()), schema)
        self.assertIs(F.from_description(None), F.LEGACY)
        with self.assertRaises(F.SchemaError):
            F.from_description({"name": "canonical-34", "version": 99})
        with self.assertRaises(F.SchemaError):
            F.from_description(dict(F.CANONICAL.describe(), inputs=198))


def fixed_frame_update(model, states, players, winner, lr, lam):
    """Reference TD(λ) on W = P(white wins), written independently of TDLambda."""
    x = torch.as_tensor(np.asarray(states, dtype=np.float32))
    params = list(model.parameters())
    traces = [torch.zeros_like(p) for p in params]

    def w(t):
        v = model(x[t:t + 1])[0, 0]
        return v if players[t] == 1 else 1 - v

    n = len(players)
    for t in range(n):
        if t == n - 1:
            if winner is None:
                break
            target = 1.0 if winner == 1 else 0.0
        else:
            with torch.no_grad():
                target = w(t + 1).item()
        model.zero_grad()
        wt = w(t)
        wt.backward()
        delta = target - wt.item()
        with torch.no_grad():
            for p, e in zip(params, traces):
                e.mul_(lam).add_(p.grad)
                p.add_(e, alpha=lr * delta)


class TDLambdaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.model = BackgammonNN(F.CANONICAL.size, hidden_sizes=[6])
        rng = np.random.default_rng(2)
        self.states = rng.random((7, F.CANONICAL.size), dtype=np.float32)

    def _check_against_reference(self, players, winner):
        a, b = copy.deepcopy(self.model), copy.deepcopy(self.model)
        TDLambda(a, learning_rate=0.3, lambda_param=0.7).update(self.states[:len(players)], players, winner)
        fixed_frame_update(b, self.states[:len(players)], players, winner, lr=0.3, lam=0.7)
        moved = False
        for pa, pb, p0 in zip(a.parameters(), b.parameters(), self.model.parameters()):
            np.testing.assert_allclose(pa.detach().numpy(), pb.detach().numpy(), atol=1e-6)
            moved |= not torch.equal(pa, p0)
        self.assertTrue(moved)

    def test_matches_fixed_frame_td_lambda(self):
        self._check_against_reference([1, -1, 1, -1, 1, -1, 1], winner=1)
        self._check_against_reference([-1, 1, -1, 1, -1, 1], winner=1)
        self._check_against_reference([1, -1, 1, -1, 1], winner=None)

    def test_matches_when_a_player_repeats(self):
        # Never happens in real games, but the update must stay correct.
        self._check_against_reference([1, 1, -1, -1, 1, -1, -1], winner=-1)

    def test_terminal_state_is_trained_on_the_outcome(self):
        x = self.states[:1]
        for winner, direction in ((1, 1), (-1, -1)):
            model = copy.deepcopy(self.model)
            before = model(torch.as_tensor(x)).item()
            TDLambda(model, learning_rate=0.5).update(x, [1], winner)
            after = model(torch.as_tensor(x)).item()
            self.assertEqual(np.sign(after - before), direction)
        model = copy.deepcopy(self.model)
        TDLambda(model, learning_rate=0.5).update(x, [1], None)  # cut off: nothing to learn
        for p, p0 in zip(model.parameters(), self.model.parameters()):
            self.assertTrue(torch.equal(p, p0))

    def test_learns_a_hand_calculable_game(self):
        # White to roll in s0, black in s1, and black bears off: V(s1) -> 1, V(s0) = 1 - V(s1) -> 0.
        x = self.states[:2]
        td = TDLambda(self.model, learning_rate=0.5, lambda_param=0.7)
        for _ in range(400):
            td.update(x, [1, -1], -1)
        v0, v1 = self.model(torch.as_tensor(x)).detach().numpy()[:, 0]
        self.assertGreater(v1, 0.9)
        self.assertLess(v0, 0.1)


class TrainingTests(unittest.TestCase):
    def test_episodes_record_the_roller(self):
        from classes import board_cpp

        torch.manual_seed(0)
        random.seed(0)
        board_cpp.seed_dice(0)
        model = BackgammonNN(F.CANONICAL.size, hidden_sizes=[8])
        trainer = BackgammonTrainer(model, F.CANONICAL, TDLambda(model), eval_games=0, num_workers=1)
        for _ in range(5):
            ep = trainer.play_game()
            self.assertEqual(ep.states.shape, (len(ep.players), F.CANONICAL.size))
            self.assertTrue(all(a == -b for a, b in zip(ep.players, ep.players[1:])))
            if ep.winner is not None:
                self.assertEqual(ep.winner, ep.players[-1])  # only the side to move can bear off

    def test_trainer_refuses_white_perspective_features(self):
        model = BackgammonNN(F.LEGACY.size, hidden_sizes=[4])
        with self.assertRaises(ValueError):
            BackgammonTrainer(model, extract_features, TDLambda(model))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "net.pt")
        self.model = canonical_checkpoint(self.path)

    def tearDown(self):
        self.dir.cleanup()

    def test_checkpoints_describe_themselves(self):
        ckpt = torch.load(self.path)
        self.assertEqual(ckpt["feature_schema"], F.CANONICAL.describe())
        self.assertEqual(ckpt["architecture"], {"input_size": 34, "hidden_sizes": [16]})
        model, schema = load_network(self.path)
        self.assertIs(schema, F.CANONICAL)
        self.assertIs(load_network(LEGACY_NN)[1], F.LEGACY)

    def test_unknown_schema_is_rejected(self):
        ckpt = torch.load(self.path)
        ckpt["feature_schema"]["version"] = 3
        torch.save(ckpt, self.path)
        with self.assertRaises(F.SchemaError):
            FinalNNAgent(checkpoint_path=self.path)

    def test_resume_checks_the_schema(self):
        ckpt = torch.load(self.path)
        del ckpt["feature_schema"]  # now looks like an old white-34 checkpoint
        torch.save(ckpt, self.path)
        model = BackgammonNN(F.CANONICAL.size, hidden_sizes=[16])
        trainer = BackgammonTrainer(model, F.CANONICAL, TDLambda(model), eval_games=0, num_workers=1)
        self.assertEqual(trainer.load_checkpoint(self.path), (0, []))  # refused, starts fresh


class DefaultNetworkTests(unittest.TestCase):
    def test_default_is_the_canonical_128x128_network(self):
        from bench import agents

        self.assertEqual(os.path.realpath(agents.resolve_path(agents.DEFAULT_NN)),
                         os.path.realpath(DEFAULT_CHECKPOINT))
        agent = FinalNNAgent()  # what the server and neural MCTS build
        self.assertIs(agent.schema, F.CANONICAL)
        self.assertEqual(hidden_sizes_of(agent.model.state_dict()), [128, 128])

    def test_server_shares_one_network(self):
        import app

        self.assertIs(app.neural_agent(), app.neural_agent())


class CanonicalAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.dir.name, "net.pt")
        canonical_checkpoint(cls.path, hidden=(24, 12), seed=3)
        cls.agent = FinalNNAgent(checkpoint_path=cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_mirrored_positions_get_mirrored_moves(self):
        from bench import positions

        for p, board in suite_boards():
            if len(board.valid_moves) < 2 or p["category"] == "known":
                continue
            mirrored = positions.to_board(mirror_dict(p))
            chosen = []
            for b in (board, mirrored):
                child = b.clone()
                child.move_from_sequence(self.agent.select_move(b.clone()))
                chosen.append(F.CANONICAL.encode(child, -b.turn))
            self.assertEqual(chosen[0], chosen[1], p["id"])

    def test_win_probabilities_are_consistent(self):
        from bench import positions

        for p, board in suite_boards(roll=False)[:80]:
            mine = self.agent.win_probability(board, board.turn)
            self.assertAlmostEqual(mine + self.agent.win_probability(board, -board.turn), 1.0, places=6)
            mirrored = positions.to_board(mirror_dict(p), roll=False)
            self.assertAlmostEqual(mine, self.agent.win_probability(mirrored, mirrored.turn), places=6)

    def test_takes_a_winning_move(self):
        from bench import bearoff

        # White bears off its last checker with any roll; black is far from done.
        board = Board({"positions": bearoff.to_positions((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 5)), "dice": "",
                       "rolled": False, "turn": 1, "white_bar": 0, "black_bar": 0, "white_off": 14,
                       "black_off": 10, "game_over": False})
        board.set_dice([2, 1])
        child = board.clone()
        child.move_from_sequence(self.agent.select_move(board.clone()))
        self.assertTrue(child.game_over)

    def test_fast_evaluator_agrees(self):
        from bench import valuenet

        net = valuenet.ValueNet.from_checkpoint(self.path)
        self.assertIs(net.schema, F.CANONICAL)
        fast = valuenet.FastNNAgent(net)
        for _, board in suite_boards():
            if board.valid_moves:
                self.assertEqual(fast.select_move(board.clone()), self.agent.select_move(board.clone()))
        for _, board in suite_boards(roll=False)[:50]:
            self.assertAlmostEqual(net.value(board), self.agent.win_probability(board, 1), places=5)

    def test_neural_mcts_uses_the_canonical_network(self):
        from classes.agents.CMCTS2 import MCTSAgent2  # type: ignore

        agent = MCTSAgent2(mode="neural", eval_depth=4, nn_agent=self.agent, time_budget=60, max_iterations=25)
        board = Board()
        board.set_dice([6, 1])
        self.assertIn(agent.select_move(board), board.valid_moves)
        self.assertEqual(agent.mcts.sim_count, 25)


if __name__ == "__main__":
    unittest.main()
