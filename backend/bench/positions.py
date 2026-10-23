"""The fixed position suite shared by the rules and speed benchmarks.

Positions live in ``data/positions.json`` and are committed, so every run
measures exactly the same decisions even after the move generator changes.
Regenerate only deliberately (``python -m bench positions --force``), since
doing so breaks comparability with earlier runs.
"""
import json
import os
import random

from .seeding import DiceStream, derive_seed

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "positions.json")
CATEGORIES = ("opening", "contact", "bar", "race", "bearoff")

# Known move-generation reproduction (improvements.md #1): with white to play
# 5-3, only 16 -> 19 is legal, whichever order the dice are listed in.
KNOWN_POSITIONS = [
    {
        "id": "known-dice-order-16-19",
        "category": "known",
        "note": "Only the 3 can be played; [5,3] and [3,5] must agree",
        "positions": [-13] + [0] * 15 + [2] + [0] * 4 + [-2, 0, 13],
        "turn": 1,
        "white_bar": 0,
        "black_bar": 0,
        "white_off": 0,
        "black_off": 0,
        "dice": [5, 3],
    },
]


def load(path=DATA_PATH):
    with open(path) as f:
        return json.load(f)


def suite_fingerprint(suite):
    return format(derive_seed(json.dumps(suite["positions"], sort_keys=True)), "x")[:12]


def to_board(pos, roll=True):
    """Construct a board from a suite entry, optionally applying its dice."""
    from classes.board_cpp import Board  # type: ignore

    board = Board(
        {
            "positions": list(pos["positions"]),
            "dice": "",
            "rolled": False,
            "turn": pos["turn"],
            "white_bar": pos["white_bar"],
            "black_bar": pos["black_bar"],
            "white_off": pos["white_off"],
            "black_off": pos["black_off"],
            "game_over": False,
        }
    )
    if roll:
        board.set_dice(list(pos["dice"]))
    return board


def category(board):
    turn = board.turn
    if (turn == 1 and board.white_bar) or (turn == -1 and board.black_bar):
        return "bar"
    if board.passed:
        own = [i for i, n in enumerate(board.positions) if n * turn > 0]
        home = range(18, 24) if turn == 1 else range(0, 6)
        if all(i in home for i in own):
            return "bearoff"
        return "race"
    return "contact"


def snapshot(board, dice, cat, ident):
    return {
        "id": ident,
        "category": cat,
        "positions": list(board.positions),
        "turn": board.turn,
        "white_bar": board.white_bar,
        "black_bar": board.black_bar,
        "white_off": board.white_off,
        "black_off": board.black_off,
        "dice": dice,
    }


def generate(per_category=80, seed=1, epsilon=0.25):
    """Sample positions from heuristic self-play with some random moves.

    Self-play with exploration gives more realistic positions than purely
    random play while still covering awkward structures.
    """
    from classes.board_cpp import Board  # type: ignore
    from classes.agents.HeuristicAgent import HeuristicAgent

    agent = HeuristicAgent()
    rng = random.Random(seed)
    buckets = {c: [] for c in CATEGORIES if c != "opening"}
    opening = []
    for d1 in range(1, 7):
        for d2 in range(d1 + 1, 7):
            opening.append(snapshot(Board(), [d2, d1], "opening", f"opening-{d2}{d1}"))

    game = 0
    while any(len(v) < per_category for v in buckets.values()) and game < 20000:
        dice_stream = DiceStream(derive_seed("positions", seed, game))
        board = Board()
        turns = 0
        while not board.game_over and turns < 400:
            dice = dice_stream.next()
            cat = category(board)
            bucket = buckets[cat]
            # Sample sparsely so one game does not dominate a category.
            if len(bucket) < per_category and rng.random() < 0.08:
                bucket.append(snapshot(board, dice, cat, f"{cat}-{len(bucket):03d}"))
            board.set_dice(dice)
            if not board.valid_moves:
                move = []
            elif rng.random() < epsilon:
                move = rng.choice(board.valid_moves)
            else:
                move = agent.select_move(board)
            board.move_from_sequence(move)
            turns += 1
        game += 1

    positions = list(KNOWN_POSITIONS) + opening
    for cat in buckets:
        positions.extend(buckets[cat])
    return {
        "version": 1,
        "seed": seed,
        "generator": f"heuristic self-play, epsilon={epsilon}",
        "positions": positions,
    }


def save(suite, path=DATA_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(suite, f, separators=(",", ":"))
        f.write("\n")
