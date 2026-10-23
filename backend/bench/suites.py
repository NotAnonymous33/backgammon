"""Named benchmark configurations.

``quick`` is for a fast check while iterating; ``standard`` is for before
merging; ``full`` adds an MCTS match and longer training, for milestones.
"""
import copy

DEFAULT_SEED = 20261023
SECTIONS = ("rules", "speed", "search", "training", "strength")

_MATCHES = [("nn", "random"), ("nn", "heuristic"), ("heuristic", "random")]

# The three MCTS evaluation modes cost very different amounts per iteration.
_MCTS = ["mcts:time=0.1", "mcts:time=0.1,mode=heuristic,depth=40", "mcts:time=0.1,mode=neural,depth=40"]

# Mirrors the network and learner constructed in NNAgent.main().
_TRAIN_MODEL = {"hidden": [40], "lr": 0.1, "lambda": 0.7}


def _training(**overrides):
    cfg = {
        "model": dict(_TRAIN_MODEL),
        "speed_repeats": 3,
        "speed_min_time": 1.0,
        "epoch_games": 32,
        "epoch_workers": 4,
        "epoch_repeats": 2,
        # Independent training runs, each from its own seed. 0 games skips the
        # learning check: historical training only improved after ~8k games,
        # so short budgets cannot show learning (see README.md).
        "learn_seeds": 4,
        "learn_games": 0,
        "learn_pairs": 50,
        "learn_opponents": ["random", "heuristic"],
    }
    cfg.update(overrides)
    return cfg


SUITES = {
    "quick": {
        "repeats": 5,
        "min_time": 0.2,
        "latency_agents": ["heuristic", "nn"],
        "latency_positions": 60,
        "mcts_agents": _MCTS,
        "search_positions": 20,
        "search_repeats": 1,
        "training": _training(),
        "matches": [{"a": a, "b": b, "pairs": 100} for a, b in _MATCHES],
    },
    "standard": {
        "repeats": 5,
        "min_time": 0.3,
        "latency_agents": ["heuristic", "nn"],
        "latency_positions": 200,
        "mcts_agents": _MCTS,
        "search_positions": 60,
        "search_repeats": 2,
        "training": _training(speed_repeats=5, epoch_games=64, epoch_repeats=3,
                              learn_seeds=8, learn_games=3000, learn_pairs=100),
        "matches": [{"a": a, "b": b, "pairs": 1000} for a, b in _MATCHES],
    },
    "full": {
        "repeats": 7,
        "min_time": 0.5,
        "latency_agents": ["heuristic", "nn"],
        "latency_positions": 400,
        "mcts_agents": _MCTS + ["mcts:time=1.0"],
        "search_positions": 120,
        "search_repeats": 2,
        "training": _training(speed_repeats=7, epoch_games=128, epoch_workers=8, epoch_repeats=3,
                              learn_seeds=16, learn_games=10000, learn_pairs=200),
        "matches": [{"a": a, "b": b, "pairs": 3000} for a, b in _MATCHES]
        + [{"a": "mcts:time=0.05", "b": "heuristic", "pairs": 150}],
    },
}


def get(name):
    if name not in SUITES:
        raise KeyError(f"unknown suite {name!r}; choose from {', '.join(SUITES)}")
    cfg = copy.deepcopy(SUITES[name])
    cfg.update(seed=DEFAULT_SEED, move_limit=1000, sections=list(SECTIONS))
    return cfg
