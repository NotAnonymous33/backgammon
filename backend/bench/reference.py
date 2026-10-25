"""Reference positions with the winning chances of every legal play.

``data/reference.json`` holds the decision suite the ``skill`` section
scores agents on. Each entry is a position with dice, and for every distinct
position the mover can reach (an "afterstate") it records white's chance of
winning from there:

* **exact** afterstates are in the bearoff database (bearoff.py), so the
  value is the true probability under perfect play.
* **rollout** afterstates are estimated by playing the game out many times
  with a pinned policy (the main network, choosing greedily), on dice that
  are shared between all afterstates of a position (common random numbers),
  and stopping exactly once a rollout reaches the bearoff database.

Rollouts run in two stages: every afterstate gets ``stage1`` games, then
afterstates that could still be the best play get topped up to ``stage2``.
Each label keeps the means of its even and odd games separately, so the
benchmark can estimate how much loss label noise alone produces.

Labels are expensive (about 20 minutes on 16 cores) and depend only on the
positions, the pinned policy and the rules, so they are committed and only
rebuilt deliberately with ``python -m bench reference --build``.
"""
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import time

from . import agents as agent_specs
from .positions import category as board_category, snapshot, to_board
from .seeding import DiceStream, derive_seed

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "reference.json")
VERSION = 1
POLICY = agent_specs.DEFAULT_NN
CATEGORIES = ("opening", "contact", "bar", "race", "exact")
PER_CATEGORY = {"contact": 120, "bar": 60, "race": 60, "exact": 100}
STAGES = (128, 640)
# After stage 1, afterstates within this much winning chance of the best
# (or three standard errors, if larger) are topped up to stage 2.
CANDIDATE_MARGIN = 0.10
ROLLOUT_PLY_LIMIT = 1000
ROLLOUT_CHUNK = 32  # games per worker task


def afterstate_id(board):
    """Short stable id of the position a play leads to."""
    key = (tuple(board.positions), board.white_bar, board.black_bar, board.white_off, board.black_off)
    return hashlib.sha1(repr(key).encode()).hexdigest()[:12]


def afterstates(board):
    """``{id: (first sequence reaching it, child board)}`` for every legal play."""
    out = {}
    for seq in board.valid_moves:
        child = board.clone()
        child.move_from_sequence(seq)
        out.setdefault(afterstate_id(child), (seq, child))
    return out


def category(board, db):
    if db.board_state(board):
        return "exact"
    cat = board_category(board)
    return "race" if cat == "bearoff" else cat


# --- generating the positions -------------------------------------------------

def generate_positions(net, db, per_category=None, seed=1, epsilon=0.15, max_games=20000, log=print):
    """Sample non-forced decisions from the pinned policy's self-play.

    Some moves are random so the suite also covers positions a strong
    player would avoid, as the rules suite does.
    """
    from classes.board_cpp import Board  # type: ignore
    from .valuenet import FastNNAgent

    per_category = dict(PER_CATEGORY, **(per_category or {}))
    policy = FastNNAgent(net)
    rng = random.Random(seed)
    buckets = {c: [] for c in per_category}
    opening = []
    for d1 in range(1, 7):
        for d2 in range(d1 + 1, 7):
            opening.append(snapshot(Board(), [d2, d1], "opening", f"opening-{d2}{d1}"))

    game = 0
    while any(len(buckets[c]) < n for c, n in per_category.items()) and game < max_games:
        dice_stream = DiceStream(derive_seed("reference-positions", seed, game))
        board = Board()
        turns = 0
        while not board.game_over and turns < 400:
            dice = dice_stream.next()
            cat = category(board, db)
            board.set_dice(dice)
            moves = board.valid_moves
            bucket = buckets.get(cat)
            # Exact positions only occur at the very end of a game, so take more of them.
            rate = 0.3 if cat == "exact" else 0.06
            if bucket is not None and len(bucket) < per_category[cat] and rng.random() < rate:
                if len(afterstates(board)) > 1:
                    snap = snapshot(board, dice, cat, f"{cat}-{len(bucket):03d}")
                    bucket.append(snap)
            if moves and rng.random() < epsilon:
                move = rng.choice(moves)
            else:
                move = policy.select_move(board)
            board.move_from_sequence(move)
            turns += 1
        game += 1
    log(f"  sampled positions from {game} self-play games: "
        + ", ".join(f"{c} {len(b)}" for c, b in buckets.items()))
    return opening + [p for c in per_category for p in buckets[c]]


# --- rollouts -----------------------------------------------------------------

_worker = {}


def _init_worker(policy_path):
    os.environ["OMP_NUM_THREADS"] = "1"
    _load_worker(policy_path)


def _load_worker(policy_path):
    from .bearoff import BearoffDB
    from .valuenet import FastNNAgent, ValueNet

    _worker["policy"] = FastNNAgent(ValueNet.from_checkpoint(agent_specs.resolve_path(policy_path)))
    _worker["db"] = BearoffDB.load()


def rollout(board, dice, policy, db, ply_limit=ROLLOUT_PLY_LIMIT):
    """White's result from ``board`` (side to move about to roll): 1, 0, or the exact bearoff value."""
    plies = 0
    while not board.game_over:
        if board.passed and db.board_state(board):
            return db.white_win(board, board.turn)
        if plies >= ply_limit:
            return 0.5
        board.set_dice(dice.next())
        board.move_from_sequence(policy.select_move(board))
        plies += 1
    return 1.0 if board.white_off == 15 else 0.0


def _rollout_task(args):
    pos, aid, seq, start, stop = args
    board = to_board(pos)
    board.move_from_sequence(seq)
    results = []
    for i in range(start, stop):
        dice = DiceStream(derive_seed("reference-rollout", pos["id"], i))
        results.append(rollout(board.clone(), dice, _worker["policy"], _worker["db"]))
    return pos["id"], aid, start, results


def _summarise(results):
    n = len(results)
    mean = sum(results) / n
    var = sum((r - mean) ** 2 for r in results) / (n - 1) if n > 1 else 0.0
    even, odd = results[0::2], results[1::2]
    return {
        "p": round(mean, 5),
        "se": round(math.sqrt(var / n), 5),
        "n": n,
        "halves": [round(sum(even) / len(even), 5), round(sum(odd) / max(1, len(odd)), 5)],
    }


def mover_value(label_p, turn):
    """A white-win probability seen from the side that just moved."""
    return label_p if turn == 1 else 1.0 - label_p


def build(positions_list, db, stages=STAGES, workers=None, policy_path=POLICY, log=print):
    """Label every afterstate of every position. Returns the reference document."""
    workers = workers or os.cpu_count() or 1
    entries = {}
    pending = []  # (position id, afterstate id, sequence) estimated by rollouts
    for pos in positions_list:
        board = to_board(pos)
        labels = {}
        for aid, (seq, child) in afterstates(board).items():
            if child.game_over:
                labels[aid] = {"p": 1.0 if child.white_off == 15 else 0.0, "exact": True}
            elif db.board_state(child):
                labels[aid] = {"p": round(db.white_win(child, child.turn), 6), "exact": True}
            else:
                labels[aid] = {}
                pending.append((pos["id"], aid, seq))
            labels[aid]["seq"] = seq
        entries[pos["id"]] = dict(pos, afterstates=labels)

    def run_stage(targets, start, stop, label):
        jobs = [(strip(entries[pid]), aid, seq, s, min(s + ROLLOUT_CHUNK, stop))
                for pid, aid, seq in targets for s in range(start, stop, ROLLOUT_CHUNK)]
        if not jobs:
            return
        log(f"  {label}: {len(targets)} afterstates, {len(jobs)} batches on {workers} worker(s)")
        collected = {}
        t0 = last = time.perf_counter()

        def gather(results_iter):
            nonlocal last
            for done, (pid, aid, first, results) in enumerate(results_iter, 1):
                collected.setdefault((pid, aid), {})[first] = results
                now = time.perf_counter()
                if now - last > 30 or done == len(jobs):
                    last = now
                    log(f"    {done}/{len(jobs)} ({now - t0:.0f}s)")

        if workers == 1:
            _load_worker(policy_path)
            gather(map(_rollout_task, jobs))
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers, initializer=_init_worker, initargs=(policy_path,)) as pool:
                gather(pool.imap_unordered(_rollout_task, jobs))
        for (pid, aid), parts in collected.items():
            label = entries[pid]["afterstates"][aid]
            results = label.pop("_results", []) + [r for first in sorted(parts) for r in parts[first]]
            label.update(_summarise(results))
            label["_results"] = results

    run_stage(pending, 0, stages[0], f"stage 1 ({stages[0]} games each)")
    if len(stages) > 1 and stages[1] > stages[0]:
        candidates = []
        for pid, aid, seq in pending:
            entry = entries[pid]
            values = [mover_value(l["p"], entry["turn"]) for l in entry["afterstates"].values()]
            label = entry["afterstates"][aid]
            if mover_value(label["p"], entry["turn"]) >= max(values) - max(CANDIDATE_MARGIN, 3 * label["se"]):
                candidates.append((pid, aid, seq))
        run_stage(candidates, stages[0], stages[1], f"stage 2 (topping candidates up to {stages[1]} games)")
    for entry in entries.values():
        for label in entry["afterstates"].values():
            label.pop("_results", None)

    return {
        "version": VERSION,
        "policy": {"spec": f"nn:{policy_path}", "digest": agent_specs.file_digest(policy_path),
                   "description": "greedy 1-ply play by the pinned network"},
        "stages": list(stages),
        "candidate_margin": CANDIDATE_MARGIN,
        "positions": list(entries.values()),
    }


def strip(entry):
    """A position without its labels, small enough to send to workers."""
    return {k: v for k, v in entry.items() if k != "afterstates"}


def load(path=DATA_PATH):
    with open(path) as f:
        return json.load(f)


def save(doc, path=DATA_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
        f.write("\n")
    os.replace(tmp, path)


def fingerprint(doc):
    """Changes whenever positions or labels change, so runs on different labels are not paired."""
    body = json.dumps([[p["id"], p["afterstates"]] for p in doc["positions"]], sort_keys=True)
    return hashlib.sha1(body.encode()).hexdigest()[:12]
