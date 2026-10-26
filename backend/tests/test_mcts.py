"""Tests for MCTSAgent2's search: value perspective, chance handling and afterstate merging."""
import random
import unittest
from collections import Counter

from bench import agents


def _board(dice=None, plies=0, seed=0):
    """The opening position, optionally advanced by random plays, with ``dice`` rolled."""
    from classes import board_cpp  # type: ignore

    board_cpp.seed_dice(seed)
    rng = random.Random(seed)
    board = board_cpp.Board()
    for _ in range(plies):
        board.roll_dice()
        board.move_from_sequence(rng.choice(board.valid_moves) if board.valid_moves else [])
    if dice:
        board.set_dice(dice)
    else:
        board.roll_dice()
    return board


def _nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        for roll in node.rolls.values():
            stack.extend(roll.children.values())


class MctsSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = {}
        cls.nn = agents._load_nn(agents.DEFAULT_NN, cls.cache)

    def _search(self, board, iters, **kwargs):
        from classes.agents.CMCTS2 import MCTSAgent2  # type: ignore

        random.seed(1)
        kwargs.setdefault("nn_agent", self.nn)
        agent = MCTSAgent2(time_budget=60, max_iterations=iters, **kwargs)
        move = agent.select_move(board.clone())
        self.assertIn(move, board.valid_moves)
        return agent.mcts

    def test_backups_score_each_node_for_its_mover(self):
        # Every node holds its own evaluation plus its children's results, each
        # seen by the node's mover. Flipping the sign by distance from the leaf
        # (the original bug) breaks this as soon as the tree is two plies deep.
        board = _board(plies=6, seed=3)
        mcts = self._search(board, 600)
        player = mcts.player_color
        deep = 0
        for node in _nodes(mcts.root):
            if node is mcts.root:
                continue
            children = [c for roll in node.rolls.values() for c in roll.children.values()]
            deep += bool(children)
            from_children = sum(c.W if c.mover == node.mover else c.N - c.W for c in children)
            self.assertEqual(node.N, 1 + sum(c.N for c in children))
            own = mcts.evaluate(node.state)
            own = own if node.mover == player else 1.0 - own
            self.assertAlmostEqual(node.W - from_children, own, places=5)
        self.assertGreater(deep, 5)

    def test_dice_are_drawn_on_every_visit(self):
        mcts = self._search(_board(plies=4, seed=5), 800)
        busiest = max(mcts.root_roll.children.values(), key=lambda c: c.N)
        self.assertGreater(len(busiest.rolls), 10)
        from classes.agents.CMCTS2 import ROLLS  # type: ignore

        counts = Counter(ROLLS)
        self.assertEqual(sum(counts.values()), 36)
        self.assertEqual({d: n for d, n in counts.items() if d[0] == d[1]}, {(i, i): 1 for i in range(1, 7)})
        self.assertTrue(all(n == 2 for d, n in counts.items() if d[0] != d[1]))

    def test_move_orders_reaching_one_position_share_a_child(self):
        from classes.agents.CMCTS2 import afterstate_key  # type: ignore

        board = _board(dice=[2, 1])
        distinct = set()
        for move in board.valid_moves:
            b = board.clone()
            b.move_from_sequence(move)
            distinct.add(afterstate_key(b))
        self.assertLess(len(distinct), len(board.valid_moves))
        mcts = self._search(board, 200)
        self.assertEqual(len(mcts.root_roll.children), len(distinct))
        self.assertEqual(mcts.root_roll.untried, [])

    def test_eval_depth_zero_evaluates_the_leaf(self):
        from classes.agents.CMCTS2 import MCTSBackgammonAgent  # type: ignore

        self.assertEqual(MCTSBackgammonAgent(mode="rollout", eval_depth=0).eval_depth, 0)
        self.assertEqual(MCTSBackgammonAgent(mode="rollout").eval_depth, 50)
        self.assertEqual(MCTSBackgammonAgent(nn_agent=self.nn).eval_depth, 0)
        self.assertEqual(MCTSBackgammonAgent(mode="default").mode, "rollout")
        with self.assertRaises(ValueError):
            MCTSBackgammonAgent(mode="nueral")

    def test_evaluations_are_for_the_searching_player(self):
        from classes.agents.CMCTS2 import MCTSBackgammonAgent  # type: ignore

        board = _board(plies=8, seed=7)
        board.move_from_sequence(board.valid_moves[0] if board.valid_moves else [])  # leaves it unrolled
        for mode in ("heuristic", "rollout", "neural"):
            mcts = MCTSBackgammonAgent(mode=mode, eval_depth=0, nn_agent=self.nn)
            mcts.player_color = 1
            white = mcts.evaluate(board)
            mcts.player_color = -1
            black = mcts.evaluate(board)
            self.assertAlmostEqual(white + black, 1.0, places=5, msg=mode)

    def test_black_and_white_searches_both_run(self):
        for seed in (1, 2):
            board = _board(plies=seed, seed=seed)
            mcts = self._search(board, 100, mode="rollout")
            self.assertEqual(mcts.player_color, board.turn)
            self.assertTrue(all(c.mover == board.turn for c in mcts.root_roll.children.values()))


if __name__ == "__main__":
    unittest.main()
