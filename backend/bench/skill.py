"""Decision quality: how much winning chance each agent gives up per move.

Every agent plays every position of the labelled reference suite
(reference.py). A decision's **loss** is the winning chance of the best play
minus that of the play chosen, both from the mover's point of view, so a
perfect player scores 0. Unlike match results this needs no opponent, has no
dice luck, and breaks down by game phase:

    opening  the 15 non-double opening rolls
    contact  ordinary middle-game positions
    bar      the mover has checkers on the bar
    race     no contact left, but not yet in the bearoff database
    exact    both sides bearing off with at most 6 checkers: labels are exact

Labels from rollouts carry some noise, which makes even a perfect player
appear to lose a little. ``skill.noise_floor`` estimates that amount from the
two independent halves of each rollout. Comparisons between runs are paired
by position, so the floor cancels out of them.

Stochastic agents (random, MCTS) play each position several times with
different seeds. Their per-position loss is the mean over those repeats.

For network agents the section also checks the value function itself: the
network's estimate for every labelled afterstate against the label.
"""
import math
import multiprocessing as mp
import os
import sys
import time

from . import agents as agent_specs
from . import reference
from .positions import to_board
from .seeding import derive_seed, seed_agents
from .stats import mean_ci
from .store import metric

BLUNDER = 0.05  # a play giving up at least 5 points of winning chance
DETERMINISTIC = ("first", "heuristic", "nn")

_cache = {}


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def repeats_for(spec, repeats):
    return 1 if agent_specs.parse(spec)[0] in DETERMINISTIC else repeats


def decide(spec, pos, repeat, cache=None):
    """One seeded decision. Returns (afterstate id or None if refused, seconds, MCTS iterations or None)."""
    from classes import board_cpp  # type: ignore

    cache = _cache if cache is None else cache
    torch = sys.modules.get("torch")
    if torch is not None and torch.get_num_threads() != 1 and cache is _cache:
        torch.set_num_threads(1)
    seed = derive_seed("skill", spec, pos["id"], repeat)
    seed_agents(seed)
    board_cpp.seed_dice(seed & 0xFFFFFFFF)  # MCTS rolls the native dice inside its tree
    agent = agent_specs.make_agent(spec, cache)
    board = to_board(pos)
    start = time.perf_counter()
    move = agent.select_move(board.clone())
    elapsed = time.perf_counter() - start
    # MCTSAgent2 skips search for forced moves and races, leaving the count at 0.
    sims = agent.mcts.sim_count if hasattr(agent, "mcts") else None
    if board.move_from_sequence(move) is False:
        return None, elapsed, sims
    return reference.afterstate_id(board), elapsed, sims


def _task(args):
    spec, pos, repeat = args
    try:
        aid, elapsed, sims = decide(spec, pos, repeat)
    except Exception as e:  # keep the run alive; report the failure
        return spec, pos["id"], repeat, None, 0.0, None, f"{type(e).__name__}: {e}"
    return spec, pos["id"], repeat, aid, elapsed, sims, None


def mover_values(entry):
    return {aid: reference.mover_value(label["p"], entry["turn"]) for aid, label in entry["afterstates"].items()}


def losses(entry, aid):
    """(loss, chose the best play) for a choice, or None if the play is not labelled."""
    values = mover_values(entry)
    if aid not in values:
        return None
    best = max(values.values())
    return best - values[aid], values[aid] == best


def noise_floor(entries):
    """Mean loss a player choosing by one half of the rollouts shows when scored on the other half."""
    floors = []
    for e in entries:
        labels = e["afterstates"]
        halves = [[], []]
        for label in labels.values():
            h = label.get("halves") or [label["p"], label["p"]]
            for i in (0, 1):
                halves[i].append(reference.mover_value(h[i], e["turn"]))
        for pick, score in ((0, 1), (1, 0)):
            chosen = max(range(len(labels)), key=lambda i: halves[pick][i])
            floors.append(max(halves[score]) - halves[score][chosen])
    return sum(floors) / len(floors) if floors else float("nan")


def _interval(mean, half):
    """``[low, high]``, or None when there is no usable interval: no samples (NaN) or one sample (infinite)."""
    return [mean - half, mean + half] if math.isfinite(half) else None


def _mean_metric(key, label, samples, better, pairing, unit="fraction", detail=None):
    clean = [s for s in samples if s is not None]
    mean, half = mean_ci(clean)
    return metric(key, label, mean, unit=unit, kind="mean", better=better, ci=_interval(mean, half), n=len(clean),
                  samples=[None if s is None else round(s, 5) for s in samples], pairing=pairing, detail=detail)


def agent_metrics(spec, entries, choices, times, errors, pairing, sims=None):
    """Metrics for one agent; ``choices[pid]`` lists its chosen afterstate per repeat."""
    per_pos = {}
    unlabeled = refused = 0
    for e in entries:
        results = []
        for aid in choices.get(e["id"], []):
            if aid is None:
                refused += 1
                continue
            r = losses(e, aid)
            if r is None:
                unlabeled += 1
                continue
            results.append(r)
        if results:
            per_pos[e["id"]] = (
                sum(r[0] for r in results) / len(results),
                sum(r[1] for r in results) / len(results),
                sum(r[0] >= BLUNDER for r in results) / len(results),
            )

    def column(i, cat=None):
        return [per_pos[e["id"]][i] if e["id"] in per_pos else None
                for e in entries if cat is None or e["category"] == cat]

    base = f"skill[{spec}]"
    cats = [c for c in reference.CATEGORIES if any(e["category"] == c for e in entries)]
    by_cat = {}
    for c in cats:
        vals = [v for v in column(0, c) if v is not None]
        by_cat[c] = sum(vals) / len(vals) if vals else None
    out = [
        _mean_metric(f"{base}.loss", f"{spec}: winning chance lost per decision", column(0), "lower", pairing,
                     detail={"by_category": by_cat}),
    ]
    for c in cats:
        out.append(_mean_metric(f"{base}.loss.{c}", f"{spec}: winning chance lost per {c} decision",
                                column(0, c), "lower", f"{pairing}:{c}"))
    out += [
        _mean_metric(f"{base}.best_rate", f"{spec}: decisions choosing the best play", column(1), "higher", pairing),
        _mean_metric(f"{base}.blunder_rate", f"{spec}: decisions losing {BLUNDER:.0%}+ winning chance",
                     column(2), "lower", pairing),
        metric(f"{base}.unlabeled", f"{spec}: plays missing from the reference labels", unlabeled,
               unit="decisions", kind="count", better="lower",
               detail={"refused": refused, "errors": errors[:10]} if (refused or errors) else None),
    ]
    if times:
        ordered = sorted(times)
        out += [
            metric(f"{base}.decision_ms", f"{spec}: median decision time during the skill test",
                   round(ordered[len(ordered) // 2] * 1000, 3), unit="ms"),
            # The mean is the cost per decision: medians hide MCTS search, which
            # it skips for forced moves and races.
            metric(f"{base}.decision_ms_mean", f"{spec}: mean decision time during the skill test",
                   round(sum(times) / len(times) * 1000, 3), unit="ms"),
        ]
    searched = [n for n in (sims or []) if n]
    if searched:
        out.append(metric(f"{base}.iterations", f"{spec}: MCTS iterations per searched decision",
                          round(sum(searched) / len(searched), 1), unit="iterations",
                          detail={"searched": len(searched), "decisions": len(sims)}))
    return out


def value_metrics(spec, entries, pairing):
    """Network value estimates of labelled afterstates against their labels."""
    from .valuenet import ValueNet

    _, target, _ = agent_specs.parse(spec)
    net = ValueNet.from_checkpoint(agent_specs.resolve_path(target))
    errors = {"exact": [], "rollout": []}
    signed = []
    # Error in P(white wins), split by who is on roll after the move: a network
    # that cannot see the turn is biased in opposite directions for the two.
    on_roll = {(side, group): [] for side in (1, -1) for group in ("exact", "rollout")}
    for e in entries:
        board = to_board(e)
        children = {aid: child for aid, (_, child) in reference.afterstates(board).items()}
        for aid, label in e["afterstates"].items():
            child = children.get(aid)
            if child is None:
                continue
            if child.game_over:
                continue  # the network is never asked about finished games
            v = net.value(child)
            err = reference.mover_value(v, e["turn"]) - reference.mover_value(label["p"], e["turn"])
            group = "exact" if label.get("exact") else "rollout"
            errors[group].append(abs(err))
            signed.append(err)
            on_roll[(child.turn, group)].append(v - label["p"])
    base = f"skill[{spec}]"
    out = []
    for group, vals in errors.items():
        if vals:
            out.append(_mean_metric(
                f"{base}.value_error.{group}",
                f"{spec}: value error on {group}-labelled afterstates", vals, "lower", f"{pairing}:value:{group}"))
    if signed:
        mean, half = mean_ci(signed)
        out.append(metric(f"{base}.value_bias", f"{spec}: value bias (+ = overrates its own plays)", mean,
                          unit="fraction", ci=_interval(mean, half), n=len(signed)))
    for side, name in ((1, "white"), (-1, "black")):
        vals = on_roll[(side, "exact")] + on_roll[(side, "rollout")]
        if not vals:
            continue
        mean, half = mean_ci(vals)
        exact = on_roll[(side, "exact")]
        out.append(metric(
            f"{base}.value_bias.{name}_on_roll",
            f"{spec}: P(white wins) error with {name} on roll (+ = overrates white)", mean,
            unit="fraction", ci=_interval(mean, half), n=len(vals),
            detail={"exact_only": sum(exact) / len(exact) if exact else None, "exact_n": len(exact)}))
    return out


def run(cfg, log=print):
    doc = reference.load()
    entries = doc["positions"]
    limit = cfg.get("skill_positions")
    if limit:
        # Keep every category represented when sampling a subset.
        entries = [e for c in reference.CATEGORIES for e in [x for x in entries if x["category"] == c][:limit]]
    pairing = f"{reference.fingerprint(doc)}:{len(entries)}"
    specs = cfg["skill_agents"]
    repeats = cfg.get("skill_repeats", 2)
    tasks = [(spec, reference.strip(e), r) for spec in specs for r in range(repeats_for(spec, repeats)) for e in entries]
    # Interleave agents so slow ones do not all land at the end.
    tasks.sort(key=lambda t: (t[2], t[1]["id"]))
    workers = max(1, min(cfg.get("workers", 1), len(tasks)))
    log(f"  {len(tasks)} decisions over {len(entries)} labelled positions on {workers} worker(s)")

    choices = {s: {} for s in specs}
    times = {s: [] for s in specs}
    sims = {s: [] for s in specs}
    errors = {s: [] for s in specs}
    start = last = time.perf_counter()

    def record(res, done):
        nonlocal last
        spec, pid, _, aid, elapsed, n, err = res
        if err:
            errors[spec].append(err)
        else:
            choices[spec].setdefault(pid, []).append(aid)
            times[spec].append(elapsed)
            if n is not None:
                sims[spec].append(n)
        now = time.perf_counter()
        if now - last > 5 or done == len(tasks):
            last = now
            log(f"    {done}/{len(tasks)} decisions ({now - start:.0f}s)")

    if workers == 1:
        for i, t in enumerate(tasks, 1):
            record(_task(t), i)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker) as pool:
            for i, res in enumerate(pool.imap_unordered(_task, tasks, chunksize=4), 1):
                record(res, i)

    out = [
        metric("skill.noise_floor", "Loss label noise alone produces (rough upper bound)", noise_floor(entries),
               unit="fraction", detail={"reference": reference.fingerprint(doc), "policy": doc["policy"],
                                        "stages": doc.get("stages")}),
        metric("skill.positions", "Labelled positions", len(entries), unit="positions",
               detail={c: sum(e["category"] == c for e in entries) for c in reference.CATEGORIES}),
    ]
    for spec in specs:
        ms = agent_metrics(spec, entries, choices[spec], times[spec], errors[spec], pairing, sims[spec])
        if agent_specs.parse(spec)[0] == "nn":
            ms += value_metrics(spec, entries, pairing)
        out += ms
        loss = ms[0]
        if loss.get("ci"):
            by_cat = ", ".join(f"{c} {v:.1%}" for c, v in loss["detail"]["by_category"].items() if v is not None)
            log(f"  {spec}: loses {loss['value']:.2%} per decision "
                f"(95% CI {loss['ci'][0]:.2%}–{loss['ci'][1]:.2%}); {by_cat}")
        else:
            # An agent that fails every decision must not stop the other agents or sections being saved.
            log(f"  {spec}: no scored decisions ({loss['n']} of {len(entries)} positions); "
                f"first errors: {errors[spec][:3] or 'none recorded'}")
    return out
