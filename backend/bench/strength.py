"""Playing strength from seeded, colour-swapped game pairs.

Pair ``i`` of every match uses the same dice stream twice: once with agent A
as white and once as black. Swapping seats on identical dice cancels most of
the luck, and because pair seeds depend only on the base seed, two runs with
the same seed play the same dice, so their per-game results can be compared
pairwise (see compare.py).

Outcomes are stored per game from A's point of view:
    1 A won   0 A lost   d move limit reached   x error (excluded from scores)
"""
import multiprocessing as mp
import os
import sys
import time

from . import agents as agent_specs
from .seeding import DiceStream, derive_seed, seed_agents
from .stats import mean_ci, wilson
from .store import metric

_cache = {}


def _init_worker():
    # Each worker runs one game at a time; intra-op threads only contend.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def play_game(white_spec, black_spec, seed, move_limit, cache=None):
    """Play one seeded game. Returns (winner, turns, illegal) with winner 1, -1 or 0."""
    cache = _cache if cache is None else cache
    white = agent_specs.make_agent(white_spec, cache)
    black = agent_specs.make_agent(black_spec, cache)
    torch = sys.modules.get("torch")
    if torch is not None and torch.get_num_threads() != 1 and cache is _cache:
        torch.set_num_threads(1)

    seed_agents(derive_seed(seed, "agents", white_spec, black_spec))
    return play_agents(white, black, seed, move_limit)


def play_agents(white, black, seed, move_limit):
    """Play one game between agent objects on the seeded dice for ``seed``."""
    from classes.board_cpp import Board  # type: ignore

    dice = DiceStream(derive_seed(seed, "dice"))
    board = Board()
    turns = 0
    while not board.game_over:
        if turns >= move_limit:
            return 0, turns, False
        mover = board.turn
        board.set_dice(dice.next())
        move = (white if mover == 1 else black).select_move(board)
        if board.move_from_sequence(move) is False:
            return -mover, turns, True
        turns += 1
    return (1 if board.white_off == 15 else -1), turns, False


def _task(args):
    match_idx, a, b, pair, a_white, seed, move_limit = args
    white, black = (a, b) if a_white else (b, a)
    try:
        winner, turns, illegal = play_game(white, black, seed, move_limit)
    except Exception as e:  # keep the run alive; report the failure
        return match_idx, pair, a_white, "x", 0, False, f"{type(e).__name__}: {e}"
    return match_idx, pair, a_white, outcome_for(winner, a_white), turns, illegal, None


def outcome_for(winner, a_white):
    """Outcome character from A's point of view (see module docstring)."""
    if winner == 0:
        return "d"
    return "1" if (winner == 1) == a_white else "0"


def score(outcomes):
    """Per-game scores for A, skipping errors."""
    return [{"1": 1.0, "0": 0.0, "d": 0.5}[o] for o in outcomes if o != "x"]


def run(cfg, log=print):
    matches = cfg["matches"]
    seed = cfg["seed"]
    move_limit = cfg["move_limit"]
    tasks = []
    for mi, m in enumerate(matches):
        for pair in range(m["pairs"]):
            pair_seed = derive_seed(seed, "pair", pair)
            for a_white in (True, False):
                tasks.append((mi, m["a"], m["b"], pair, a_white, pair_seed, move_limit))
    # Interleave matches so slow agents do not all land at the end.
    tasks.sort(key=lambda t: (t[3], t[0], not t[4]))

    results = [dict(outcomes={}, turns=[], illegal=0, errors=[]) for _ in matches]
    workers = max(1, min(cfg["workers"], len(tasks)))
    log(f"  {len(tasks)} games across {len(matches)} match(es) on {workers} worker(s)")

    start = time.perf_counter()
    done = 0
    last = start

    def record(res):
        nonlocal done, last
        mi, pair, a_white, outcome, turns, illegal, err = res
        r = results[mi]
        r["outcomes"][(pair, a_white)] = outcome
        if outcome != "x":
            r["turns"].append(turns)
        r["illegal"] += illegal
        if err and len(r["errors"]) < 10:
            r["errors"].append(err)
        done += 1
        now = time.perf_counter()
        if now - last > 5 or done == len(tasks):
            last = now
            log(f"    {done}/{len(tasks)} games ({now - start:.0f}s)")

    if workers == 1:
        for t in tasks:
            record(_task(t))
    else:
        ctx = mp.get_context("spawn")  # fork after torch/OpenMP init can deadlock
        with ctx.Pool(workers, initializer=_init_worker) as pool:
            for res in pool.imap_unordered(_task, tasks, chunksize=4):
                record(res)

    metrics = []
    for m, r in zip(matches, results):
        a, b = m["a"], m["b"]
        ordered = [r["outcomes"][(p, w)] for p in range(m["pairs"]) for w in (True, False)]
        mean_turns, turns_ci = mean_ci(r["turns"])
        base = f"strength[{a} vs {b}]"
        win = rate_metric(f"{base}.win_rate", f"{a} vs {b}: score", ordered, pairing=f"{seed}:{m['pairs']}",
                          detail={"a": agent_specs.describe(a), "b": agent_specs.describe(b)})
        metrics += [
            win,
            metric(f"{base}.turns", f"{a} vs {b}: mean game length", mean_turns, unit="turns",
                   ci=[mean_turns - turns_ci, mean_turns + turns_ci]),
            metric(f"{base}.truncated", f"{a} vs {b}: games hitting the move limit",
                   ordered.count("d"), unit="games", kind="count", better="lower"),
            metric(f"{base}.illegal", f"{a} vs {b}: illegal moves submitted",
                   r["illegal"], unit="games", kind="count", better="lower"),
            metric(f"{base}.errors", f"{a} vs {b}: games that crashed",
                   ordered.count("x"), unit="games", kind="count", better="lower",
                   detail={"examples": r["errors"]} if r["errors"] else None),
        ]
        lo, hi = win["ci"]
        log(f"  {a} vs {b}: {win['value']:.1%} (95% CI {lo:.1%}–{hi:.1%}), n={win['n']}")
    return metrics


def rate_metric(key, label, ordered, pairing, detail=None):
    """Score metric over per-game outcomes (in seed order, so runs can be paired)."""
    scores = score(ordered)
    n = len(scores)
    wins = sum(scores)
    lo, hi = wilson(wins, n)
    detail = dict(detail or {}, wins=ordered.count("1"), losses=ordered.count("0"),
                  as_white=_rate(ordered[0::2]), as_black=_rate(ordered[1::2]))
    return metric(key, label, wins / n if n else float("nan"), unit="fraction", kind="rate",
                  better="higher", ci=[lo, hi], n=n, outcomes="".join(ordered), pairing=pairing,
                  detail=detail)


def _rate(outcomes):
    s = score(outcomes)
    return sum(s) / len(s) if s else None
