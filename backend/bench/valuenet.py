"""Fast NumPy inference for the project's value networks.

``FinalNNAgent`` evaluates afterstates with PyTorch, whose per-call overhead
dominates for networks this small (performance.md #17). Reference rollouts
make millions of such decisions, so they use this evaluator instead. It
uses the same feature schemas (classes/agents/features.py) and the
``BackgammonNN`` layer layout; tests check that it chooses the same moves as
``FinalNNAgent`` for both ``white-34`` and ``canonical-34`` networks.

The architecture is read from the checkpoint's weight shapes, so 34-40-1
(``tdgammon``) and 34-128-128-1 (``main``) checkpoints both load, and the
feature schema from the checkpoint itself (``white-34`` when absent).
Loading is strict: unexpected or missing tensors raise instead of silently
leaving layers at their random initialisation.
"""
import re

import numpy as np

from classes.agents import features as feature_schemas

N_FEATURES = feature_schemas.LEGACY.size
features = feature_schemas.legacy_features  # white-34 features, as NNAgent.extract_features


def load_checkpoint(path):
    """(state dict, feature schema) of a checkpoint or bare state dict."""
    import torch

    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], feature_schemas.from_description(ckpt.get("feature_schema"))
    return ckpt, feature_schemas.LEGACY


def load_state_dict(path):
    return load_checkpoint(path)[0]


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
    """Position values from a ``BackgammonNN`` checkpoint, interpreted through its feature schema."""

    def __init__(self, state_dict, schema=feature_schemas.LEGACY):
        shapes = layer_shapes(state_dict)
        self.schema = schema
        self.hidden = [out for _, out in shapes[:-1]]
        self.layers = [
            (state_dict[f"model.{2 * i}.weight"].detach().cpu().numpy().astype(np.float32).T.copy(),
             state_dict[f"model.{2 * i}.bias"].detach().cpu().numpy().astype(np.float32))
            for i in range(len(shapes))
        ]

    @classmethod
    def from_checkpoint(cls, path):
        return cls(*load_checkpoint(path))

    def evaluate(self, rows):
        """Raw network outputs, one per feature row (see the schema for their meaning)."""
        x = np.asarray(rows, dtype=np.float32)
        last = len(self.layers) - 1
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < last:
                np.maximum(x, 0.0, out=x)
        x = x[:, 0]
        return 1.0 / (1.0 + np.exp(-x))

    def value(self, board):
        """The network's P(white wins) for ``board`` with ``board.turn`` about to roll."""
        if board.game_over:
            return self.schema.win_probability(board, None, 1)
        output = float(self.evaluate([self.schema.features(board)])[0])
        return self.schema.win_probability(board, output, 1)

    def choose(self, board, moves=None):
        """Index of the move FinalNNAgent would play, and the raw network output for each afterstate."""
        moves = board.valid_moves if moves is None else moves
        children = []
        for move in moves:
            child = board.clone()
            child.move_from_sequence(move)
            children.append(child)
        values = self.evaluate(self.schema.afterstate_rows(board.turn, children))
        if self.schema.canonical:
            return self.schema.best_index(board.turn, children, values), values
        # white-34: exactly FinalNNAgent's argmax/argmin, including how ties break.
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
