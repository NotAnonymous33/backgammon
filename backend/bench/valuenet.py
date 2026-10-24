"""Fast NumPy inference for the project's value networks.

``FinalNNAgent`` evaluates afterstates with PyTorch, whose per-call overhead
dominates for networks this small (performance.md #17). Reference rollouts
make millions of such decisions, so they use this evaluator instead. It
reproduces ``classes.agents.NNAgent.extract_features`` and the
``BackgammonNN`` layer layout exactly; tests/test_skill.py checks that it
chooses the same moves as ``FinalNNAgent``.

The architecture is read from the checkpoint's weight shapes, so 34-40-1
(``tdgammon``) and 34-128-128-1 (``main``) checkpoints both load. Loading is
strict: unexpected or missing tensors raise instead of silently leaving
layers at their random initialisation.
"""
import re

import numpy as np

N_FEATURES = 34


def features(board):
    """The 34 inputs of ``NNAgent.extract_features``, in one pass over the points."""
    pos = board.positions
    f = [p / 15.0 for p in pos]
    white_singles = black_singles = 0
    white_run = black_run = white_prime = black_prime = 0
    for p in pos:
        if p >= 2:
            white_run += 1
            black_run = 0
            if white_run > white_prime:
                white_prime = white_run
        elif p <= -2:
            black_run += 1
            white_run = 0
            if black_run > black_prime:
                black_prime = black_run
        else:
            white_run = black_run = 0
            if p == 1:
                white_singles += 1
            elif p == -1:
                black_singles += 1
    # Same orientation as the original: white "anchors" on 18-23, black on 0-5.
    white_anchors = (pos[18] >= 2) + (pos[19] >= 2) + (pos[20] >= 2) + (pos[21] >= 2) + (pos[22] >= 2) + (pos[23] >= 2)
    black_anchors = (pos[0] <= -2) + (pos[1] <= -2) + (pos[2] <= -2) + (pos[3] <= -2) + (pos[4] <= -2) + (pos[5] <= -2)
    f += (
        board.white_bar / 15.0, board.black_bar / 15.0, board.white_off / 15.0, board.black_off / 15.0,
        white_singles / 15.0, black_singles / 15.0, white_prime / 6.0, black_prime / 6.0,
        white_anchors / 6.0, black_anchors / 6.0,
    )
    return f


def load_state_dict(path):
    import torch

    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def layer_shapes(state_dict):
    """``[(in, out), ...]`` for each Linear layer of a ``BackgammonNN`` state dict."""
    linear = {}
    for key, tensor in state_dict.items():
        m = re.fullmatch(r"model\.(\d+)\.(weight|bias)", key)
        if not m:
            raise ValueError(f"unexpected tensor {key!r}; not a BackgammonNN checkpoint")
        linear.setdefault(int(m.group(1)), {})[m.group(2)] = tuple(tensor.shape)
    shapes = []
    for i, idx in enumerate(sorted(linear)):
        # Linear layers sit at 0, 2, 4, ... with ReLU (and the final Sigmoid) between them.
        if idx != 2 * i or set(linear[idx]) != {"weight", "bias"}:
            raise ValueError(f"unexpected layer layout {sorted(linear)}; not a BackgammonNN checkpoint")
        out, inp = linear[idx]["weight"]
        if shapes and shapes[-1][1] != inp:
            raise ValueError(f"layer {idx} expects {inp} inputs but the previous layer has {shapes[-1][1]} outputs")
        shapes.append((inp, out))
    if not shapes or shapes[0][0] != N_FEATURES or shapes[-1][1] != 1:
        raise ValueError(f"expected {N_FEATURES} inputs and one output, got layers {shapes}")
    return shapes


def hidden_sizes(state_dict):
    return [out for _, out in layer_shapes(state_dict)[:-1]]


class ValueNet:
    """P(white wins) for positions, as estimated by a ``BackgammonNN`` checkpoint."""

    def __init__(self, state_dict):
        shapes = layer_shapes(state_dict)
        self.hidden = [out for _, out in shapes[:-1]]
        self.layers = [
            (state_dict[f"model.{2 * i}.weight"].detach().cpu().numpy().astype(np.float32).T.copy(),
             state_dict[f"model.{2 * i}.bias"].detach().cpu().numpy().astype(np.float32))
            for i in range(len(shapes))
        ]

    @classmethod
    def from_checkpoint(cls, path):
        return cls(load_state_dict(path))

    def evaluate(self, rows):
        """Win probabilities for white, one per feature row."""
        x = np.asarray(rows, dtype=np.float32)
        last = len(self.layers) - 1
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < last:
                np.maximum(x, 0.0, out=x)
        x = x[:, 0]
        return 1.0 / (1.0 + np.exp(-x))

    def value(self, board):
        return float(self.evaluate([features(board)])[0])

    def choose(self, board, moves=None):
        """Index of the move FinalNNAgent would play, and every afterstate's white-win value."""
        moves = board.valid_moves if moves is None else moves
        children = []
        for move in moves:
            child = board.clone()
            child.move_from_sequence(move)
            children.append(child)
        values = self.evaluate([features(c) for c in children])
        best = int(values.argmax() if board.turn == 1 else values.argmin())
        return best, values


class FastNNAgent:
    """Drop-in for FinalNNAgent's decisions, evaluated with NumPy."""

    def __init__(self, net):
        self.net = net

    def select_move(self, board):
        moves = board.valid_moves
        if not moves:
            return []
        if len(moves) == 1:
            return moves[0]
        best, _ = self.net.choose(board, moves)
        return moves[best]
