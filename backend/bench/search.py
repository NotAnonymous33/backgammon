"""MCTS search throughput: iterations completed within the time budget.

Each configured MCTS agent searches every non-forced, contact position in
the suite from a fresh tree. MCTSAgent2 resets ``mcts.sim_count`` at the
start of each search, so after ``select_move`` it holds the number of
select/expand/simulate/backpropagate iterations completed. Rates use the
measured search time rather than the nominal budget.

Rollout dice are seeded per position when the native board supports it, so
runs simulate the same games and differ only in how fast they run them.
"""
from time import perf_counter

from . import agents as agent_specs
from . import progress
from .positions import to_board
from .seeding import derive_seed, seed_agents
from .stats import percentile
from .store import metric

# Iteration counts vary more than engine microbenchmarks: each depends on the
# lengths of random rollouts and on the time-based stopping rule.
NOISE_FLOOR = 0.15


def _positions(suite, limit):
    boards = []
    for pos in suite["positions"]:
        if pos["category"] == "known":
            continue
        board = to_board(pos)
        # MCTSAgent2 skips search for forced moves and for passed (race) boards.
        if len(board.valid_moves) > 1 and not board.passed:
            boards.append((pos["id"], board))
    return boards[:limit]


def iterations(spec, suite, repeats, limit, log=print):
    from classes import board_cpp  # type: ignore

    kind, _, kwargs = agent_specs.parse(spec)
    if kind != "mcts":
        raise agent_specs.SpecError(f"search benchmarks need an mcts agent, got {spec!r}")
    seedable = hasattr(board_cpp, "seed_dice")
    cache = {}
    boards = _positions(suite, limit)
    agent_specs.make_agent(spec, cache)  # load any checkpoint before timing

    best_rate = [0.0] * len(boards)
    counts = [0] * len(boards)
    for r in range(repeats):
        for i, (pid, board) in enumerate(boards):
            seed = derive_seed("search", spec, pid)
            seed_agents(seed)
            if seedable:
                board_cpp.seed_dice(seed & 0xFFFFFFFF)
            agent = agent_specs.make_agent(spec, cache)
            b = board.clone()
            start = perf_counter()
            agent.select_move(b)
            elapsed = perf_counter() - start
            sims = agent.mcts.sim_count
            # Keep each position's best repeat: interference only slows a search down.
            if sims / elapsed > best_rate[i]:
                best_rate[i] = sims / elapsed
                counts[i] = sims

    budget = float(kwargs.get("time", 1.0))
    base = f"search[{spec}]"
    rate = metric(
        f"{base}.iterations_per_s", f"{spec}: MCTS iterations per second", percentile(best_rate, 50),
        unit="iter/s", kind="timing", better="higher", noise_floor=NOISE_FLOOR,
        samples=[round(x, 1) for x in best_rate], samples_role="diagnostic",
        detail={"positions": len(boards), "repeats": repeats, "seeded_rollouts": seedable,
                "p10": percentile(best_rate, 10), "p90": percentile(best_rate, 90)},
    )
    per_decision = metric(
        f"{base}.iterations_per_decision", f"{spec}: iterations per decision ({budget:g}s budget)",
        round(percentile(counts, 50)), unit="iterations",
        detail={"min": min(counts) if counts else None, "max": max(counts) if counts else None},
    )
    log(f"  {spec}: {rate['value']:,.0f} iter/s, median {per_decision['value']:,.0f} per decision "
        f"over {len(boards)} positions")
    return [rate, per_decision]


def run(suite, cfg, log=print):
    results = []
    progress.update(0, len(cfg["mcts_agents"]), "configurations")
    for i, spec in enumerate(cfg["mcts_agents"], 1):
        results.extend(iterations(spec, suite, cfg["search_repeats"], cfg["search_positions"], log))
        progress.update(i, len(cfg["mcts_agents"]), "configurations")
    return results
