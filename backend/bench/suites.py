"""Named benchmark configurations.

``quick`` is for a fast check while iterating; ``standard`` is for before
merging; ``full`` adds an MCTS match and longer training, for milestones.

Skill is measured two ways (see README.md): ``skill`` scores single
decisions against labelled reference positions, which is fast and free of
dice luck, and ``ratings`` places a pool of agents on one Elo scale from
games. MCTS is measured with fixed iteration budgets (``iters=``), which are
reproducible and isolate search quality from speed, alongside the time
budget the server uses.
"""
import copy

DEFAULT_SEED = 20261023
SECTIONS = ("rules", "speed", "search", "training", "strength", "skill", "ratings")

# `nn` is the default network (canonical-34, 128x128). The white-34 network that was the default
# before it stays in the pools as the reference point for what the canonical network improved on.
_SHIPPED_NN = "nn:models/main/main_checkpoint_latest.pt"

_MATCHES = [("nn", "random"), ("nn", "heuristic"), ("heuristic", "random"), ("nn", _SHIPPED_NN)]

# Iteration budgets double, so skill against log2(iterations) shows what a
# faster search would be worth.
_MCTS_LADDER = ["mcts:iters=100", "mcts:iters=200", "mcts:iters=400", "mcts:iters=800", "mcts:iters=1600"]


def _ratings(pool, **overrides):
    cfg = {
        "pool": pool,
        "anchor": "heuristic",
        "anchor_elo": 1500,
        "initial_pairs": 20,
        "batch_pairs": 20,
        "max_pairs": 200,  # per matchup
        "target_se": 20.0,  # Elo; about ±40 at 95%
        "max_games": 4000,
    }
    cfg.update(overrides)
    return cfg

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
        "skill_agents": ["heuristic", "nn", "mcts:iters=100", "mcts:iters=200", "mcts:iters=400"],
        "skill_repeats": 1,
        "skill_positions": None,
        "ratings": _ratings(["random", "first", "heuristic", "nn"], initial_pairs=25, max_pairs=100,
                            target_se=30.0, max_games=1200),
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
        "matches": [{"a": a, "b": b, "pairs": 1000} for a, b in _MATCHES] + [{"a": "nn", "b": "mcts:iters=400", "pairs": 100}],
        "skill_agents": ["heuristic", "nn", _SHIPPED_NN, "nn:models/tdgammon/tdgammon_checkpoint_latest.pt"] + _MCTS_LADDER[:4]
        + ["mcts:iters=400,mode=heuristic,depth=40", "mcts:time=1.0"],
        "skill_repeats": 2,
        "skill_positions": None,
        # Slow MCTS budgets are rated in the full suite; skill covers them here.
        "ratings": _ratings(["random", "first", "heuristic", "nn", _SHIPPED_NN,
                             "nn:models/tdgammon/tdgammon_checkpoint_latest.pt", "mcts:iters=100"], max_games=3500),
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
        + [{"a": "mcts:time=0.05", "b": "heuristic", "pairs": 150}, {"a": "nn", "b": "mcts:iters=400", "pairs": 300}],
        "skill_agents": ["heuristic", "nn", _SHIPPED_NN, "nn:models/tdgammon/tdgammon_checkpoint_latest.pt",
                         "nn:models/momentum/momentum_checkpoint_latest.pt"] + _MCTS_LADDER
        + ["mcts:iters=400,mode=heuristic,depth=40", "mcts:iters=400,mode=neural,depth=40", "mcts:time=1.0"],
        "skill_repeats": 3,
        "skill_positions": None,
        "ratings": _ratings(["random", "first", "heuristic", "nn", _SHIPPED_NN,
                             "nn:models/tdgammon/tdgammon_checkpoint_latest.pt",
                             "nn:models/momentum/momentum_checkpoint_latest.pt", "nn:models/main/checkpoint_epoch_main50.pt",
                             "nn:models/main/checkpoint_epoch_main150.pt", "mcts:iters=100", "mcts:iters=400",
                             "mcts:iters=1600"], target_se=15.0, max_pairs=400, max_games=16000),
    },
}


def get(name):
    if name not in SUITES:
        raise KeyError(f"unknown suite {name!r}; choose from {', '.join(SUITES)}")
    cfg = copy.deepcopy(SUITES[name])
    cfg.update(seed=DEFAULT_SEED, move_limit=1000, sections=list(SECTIONS))
    return cfg
