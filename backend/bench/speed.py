"""Throughput and latency microbenchmarks on the position suite.

Each benchmark is measured ``repeats`` times; the spread of those samples is
what compare.py uses to tell real changes from noise. Numbers are only
comparable between runs on the same machine.
"""
import random
from time import perf_counter

from . import agents as agent_specs
from .positions import to_board
from .seeding import DiceStream, derive_seed, seed_agents
from .stats import percentile, summarize
from .store import metric


def throughput(pass_fn, repeats, min_time):
    """Samples of operations/second; each sample repeats pass_fn for >= min_time."""
    pass_fn()  # warm-up
    samples = []
    for _ in range(repeats):
        ops = 0
        start = perf_counter()
        while True:
            ops += pass_fn()
            elapsed = perf_counter() - start
            if elapsed >= min_time:
                break
        samples.append(ops / elapsed)
    return samples


# Smallest relative change treated as real, from repeated runs of identical
# code on an otherwise idle desktop (see README.md). Tail latency over a few
# hundred decisions is much noisier than throughput.
ENGINE_FLOOR = 0.15
P50_FLOOR = 0.15
P95_FLOOR = 0.20


def timing_metric(key, label, unit, samples, better, noise_floor, value=None):
    s = summarize(samples)
    if value is None:
        # Background load only ever slows a sample down, so the best sample is
        # the most repeatable estimate of the code's own speed.
        value = s["max"] if better == "higher" else s["min"]
    return metric(
        key, label, value, unit=unit, kind="timing", better=better,
        samples=[round(x, 6) for x in samples], noise_floor=noise_floor,
        spread={k: s[k] for k in ("median", "q1", "q3", "min", "max")},
    )


def play_random_game(seed):
    from classes.board_cpp import Board  # type: ignore

    dice = DiceStream(derive_seed(seed, "dice"))
    rng = random.Random(derive_seed(seed, "moves"))
    board = Board()
    turns = 0
    while not board.game_over and turns < 1000:
        board.set_dice(dice.next())
        moves = board.valid_moves
        board.move_from_sequence(rng.choice(moves) if moves else [])
        turns += 1
    return turns


def engine_benchmarks(suite, repeats, min_time, log=print):
    positions = [p for p in suite["positions"] if p["category"] != "known"]
    unrolled = [(to_board(p, roll=False), list(p["dice"])) for p in positions]
    rolled = [to_board(p) for p in positions]
    with_moves = [(b, list(b.valid_moves)) for b in rolled]
    n_sequences = sum(len(m) for _, m in with_moves)

    def movegen():
        for board, dice in unrolled:
            board.clone().set_dice(dice)
        return len(unrolled)

    def clone():
        for board in rolled:
            board.clone()
        return len(rolled)

    def apply_moves():
        for board, moves in with_moves:
            for seq in moves:
                board.clone().move_from_sequence(seq)
        return n_sequences

    def games():
        for i in range(10):
            play_random_game(derive_seed("speed-games", i))
        return 10

    results = []
    for key, label, unit, fn in [
        ("speed.movegen", "Move generation (clone + set_dice)", "pos/s", movegen),
        ("speed.clone_rolled", "Clone a rolled board", "clones/s", clone),
        ("speed.apply_sequence", "Clone + apply a legal sequence", "seq/s", apply_moves),
        ("speed.random_games", "Random-vs-random full games", "games/s", games),
    ]:
        samples = throughput(fn, repeats, min_time)
        m = timing_metric(key, label, unit, samples, better="higher", noise_floor=ENGINE_FLOOR)
        results.append(m)
        log(f"  {label}: {m['value']:,.0f} {unit}")
    return results


def agent_latency(spec, suite, repeats, max_positions, log=print):
    """Per-decision latency of an agent over non-forced suite positions."""
    cache = {}
    candidates = [p for p in suite["positions"] if p["category"] != "known"]
    boards = [b for b in (to_board(p) for p in candidates) if len(b.valid_moves) > 1]
    boards = boards[:max_positions]
    agent_specs.make_agent(spec, cache).select_move(boards[0].clone())  # load models, warm caches

    best = [float("inf")] * len(boards)
    p50s, p95s = [], []
    for r in range(repeats):
        seed_agents(derive_seed("latency", spec, r))
        times = []
        for i, board in enumerate(boards):
            agent = agent_specs.make_agent(spec, cache)
            b = board.clone()
            start = perf_counter()
            agent.select_move(b)
            ms = (perf_counter() - start) * 1000
            times.append(ms)
            best[i] = min(best[i], ms)
        p50s.append(percentile(times, 50))
        p95s.append(percentile(times, 95))

    # Percentiles over each position's fastest repeat: the tail then reflects
    # positions that are genuinely expensive rather than scheduler hiccups.
    # Per-repeat percentiles are kept as samples for inspection only; their
    # spread is mostly those hiccups, so the noise floor alone sets sensitivity.
    base = f"speed.latency[{spec}]"
    p50 = timing_metric(f"{base}.p50", f"{spec}: median decision time", "ms", p50s, better="lower",
                        noise_floor=P50_FLOOR, value=percentile(best, 50))
    p95 = timing_metric(f"{base}.p95", f"{spec}: p95 decision time", "ms", p95s, better="lower",
                        noise_floor=P95_FLOOR, value=percentile(best, 95))
    for m in (p50, p95):
        m["samples_role"] = "diagnostic"
        del m["spread"]
    p50["detail"] = {"decisions": repeats * len(boards), "positions": len(boards)}
    log(f"  {spec}: p50 {p50['value']:.2f} ms, p95 {p95['value']:.2f} ms over {len(boards)} positions")
    return [p50, p95]


def run(suite, cfg, log=print):
    results = engine_benchmarks(suite, cfg["repeats"], cfg["min_time"], log)
    for spec in cfg["latency_agents"]:
        results.extend(agent_latency(spec, suite, cfg["repeats"], cfg["latency_positions"], log))
    return results
