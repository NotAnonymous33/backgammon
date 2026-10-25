"""Elo ratings for a pool of agents, from adaptively scheduled matches.

Head-to-head win rates (strength.py) answer "does A beat B?", but a 100%
score against ``random`` says nothing about how much better an agent is, and
scores against different opponents cannot be compared directly. This section
places every agent in a pool on one scale:

1. Every pair of agents plays ``initial_pairs`` seeded, colour-swapped game
   pairs (the same games strength.py plays).
2. A Bradley-Terry model is fitted on the Elo scale, with the anchor agent
   fixed (``heuristic`` = 1500 by default).
3. While some agent's rating is less precise than ``target_se``, more pairs
   go to the informative matchups of the least precise agents, preferring
   evenly matched opponents. Lopsided matchups barely move a rating, so they
   are not extended. Scheduling stops at the target or the game budget.
4. Confidence intervals come from a bootstrap that resamples whole game
   pairs, which keeps the luck-cancelling pairing intact.

Scheduling depends only on game outcomes, so a rerun of identical code plays
identical games. An agent that won (or lost) every game it played has only a
lower (or upper) bound; its rating is flagged and the fix is to add an
opponent of similar strength to the pool.
"""
import math
import multiprocessing as mp
import random
import time

import numpy as np

from . import agents as agent_specs
from . import strength
from .seeding import derive_seed
from .store import metric

K = math.log(10) / 400  # Elo points to logits
PRIOR_SD = 1000.0  # weak prior keeping perfect scores finite
BOOTSTRAP = 300


def fit(n, anchor, anchor_elo, wins, games, start=None):
    """Bradley-Terry ratings on the Elo scale. ``wins[i, j]`` is i's total score against j.

    Returns (ratings, standard errors from the curvature).
    """
    r = np.full(n, float(anchor_elo)) if start is None else np.array(start, dtype=float)
    r[anchor] = anchor_elo
    free = [i for i in range(n) if i != anchor]
    hess = None
    for _ in range(100):
        diff = r[:, None] - r[None, :]
        p = 1.0 / (1.0 + np.exp(-K * diff))
        grad = K * (wins - games * p).sum(axis=1) - (r - anchor_elo) / PRIOR_SD ** 2
        w = K * K * games * p * (1 - p)
        hess = -w
        np.fill_diagonal(hess, w.sum(axis=1) + 1 / PRIOR_SD ** 2)  # negative Hessian
        h = hess[np.ix_(free, free)]
        step = np.linalg.solve(h, grad[free])
        step = np.clip(step, -400, 400)
        r[free] += step
        if np.abs(step).max() < 1e-6:
            break
    se = np.zeros(n)
    if free:
        cov = np.linalg.inv(hess[np.ix_(free, free)])
        se[free] = np.sqrt(np.diag(cov))
    return r, se


def expected(ra, rb):
    return 1.0 / (1.0 + 10 ** (-(ra - rb) / 400))


class Ladder:
    """Matchup results for a pool; each matchup holds A's scores per pair (as white, as black)."""

    def __init__(self, pool, anchor, anchor_elo):
        self.pool = list(pool)
        self.n = len(self.pool)
        self.anchor = self.pool.index(anchor)
        self.anchor_elo = anchor_elo
        self.matchups = {(i, j): [] for i in range(self.n) for j in range(i + 1, self.n)}
        self.outcomes = {m: [] for m in self.matchups}  # outcome characters, two per pair
        self.ratings = np.full(self.n, float(anchor_elo))
        self.se = np.full(self.n, float("inf"))

    def totals(self, pairs_by_matchup=None):
        wins = np.zeros((self.n, self.n))
        games = np.zeros((self.n, self.n))
        for (i, j), pairs in (pairs_by_matchup or self.matchups).items():
            for scores in pairs:
                for s in scores:
                    if s is None:
                        continue
                    wins[i, j] += s
                    wins[j, i] += 1 - s
                    games[i, j] += 1
                    games[j, i] += 1
        return wins, games

    def refit(self):
        wins, games = self.totals()
        self.ratings, self.se = fit(self.n, self.anchor, self.anchor_elo, wins, games, start=self.ratings)
        return self.ratings, self.se

    def bootstrap(self, samples=BOOTSTRAP, seed=0):
        rng = random.Random(seed)
        draws = []
        for _ in range(samples):
            resampled = {m: [pairs[rng.randrange(len(pairs))] for _ in pairs] if pairs else []
                         for m, pairs in self.matchups.items()}
            wins, games = self.totals(resampled)
            r, _ = fit(self.n, self.anchor, self.anchor_elo, wins, games, start=self.ratings)
            draws.append(r)
        return np.array(draws)

    def informativeness(self, m):
        """Fisher information per game, p(1-p), using smoothed observed scores where available."""
        i, j = m
        wins, games = self.totals({m: self.matchups[m]})
        if games[i, j]:
            p = (wins[i, j] + 1) / (games[i, j] + 2)
        else:
            p = expected(self.ratings[i], self.ratings[j])
        return p * (1 - p)

    def next_round(self, target_se, max_pairs, batch_pairs):
        """Matchups to extend: the most informative one for each agent above the target precision."""
        chosen = {}
        needy = sorted((i for i in range(self.n) if i != self.anchor and self.se[i] > target_se),
                       key=lambda i: -self.se[i])
        for i in needy:
            options = [m for m in self.matchups if i in m and len(self.matchups[m]) < max_pairs]
            # A near-certain result carries little information about the rating gap.
            options = [m for m in options if self.informativeness(m) > 0.04]
            if not options:
                continue
            best = max(options, key=self.informativeness)
            chosen[best] = min(batch_pairs, max_pairs - len(self.matchups[best]))
        return chosen

    def bounds(self):
        """'lower' for agents that won every game, 'upper' for agents that lost every game."""
        wins, games = self.totals()
        out = {}
        for i in range(self.n):
            total = games[i].sum()
            if total and wins[i].sum() == total:
                out[i] = "lower"
            elif total and wins[i].sum() == 0:
                out[i] = "upper"
        return out


def _games_for(ladder, schedule, seed, move_limit):
    tasks = []
    for (i, j), count in schedule.items():
        start = len(ladder.matchups[(i, j)])
        for pair in range(start, start + count):
            pair_seed = derive_seed(seed, "pair", pair)
            for a_white in (True, False):
                tasks.append(((i, j), ladder.pool[i], ladder.pool[j], pair, a_white, pair_seed, move_limit))
    return tasks


def _task(args):
    m, a, b, pair, a_white, seed, move_limit = args
    _, _, _, outcome, _, illegal, err = strength._task((0, a, b, pair, a_white, seed, move_limit))
    return m, pair, a_white, outcome, illegal, err


def run(cfg, log=print):
    rc = cfg["ratings"]
    pool = list(dict.fromkeys(rc["pool"]))
    anchor = rc.get("anchor", "heuristic")
    if anchor not in pool:
        pool.insert(0, anchor)
    if len(pool) < 2:
        raise agent_specs.SpecError("ratings need at least two agents in the pool")
    ladder = Ladder(pool, anchor, rc.get("anchor_elo", 1500))
    seed, move_limit = cfg["seed"], cfg["move_limit"]
    workers = max(1, cfg.get("workers", 1))
    stats = {"illegal": 0, "errors": [], "error_count": 0, "truncated": 0}
    start = time.perf_counter()

    def play(schedule, pool_):
        tasks = _games_for(ladder, schedule, seed, move_limit)
        results = {}
        iterator = map(_task, tasks) if pool_ is None else pool_.imap_unordered(_task, tasks, chunksize=2)
        for m, pair, a_white, outcome, illegal, err in iterator:
            results.setdefault((m, pair), {})[a_white] = outcome
            stats["illegal"] += illegal
            if outcome == "x":
                stats["error_count"] += 1
                if len(stats["errors"]) < 10:
                    stats["errors"].append(err)
            stats["truncated"] += outcome == "d"
        for (m, pair) in sorted(results):
            r = results[(m, pair)]
            assert len(ladder.matchups[m]) == pair, "pairs must be recorded in order"
            ladder.matchups[m].append(tuple(_score(r[w]) for w in (True, False)))
            ladder.outcomes[m] += [r[True], r[False]]
        return len(tasks)

    def rounds(pool_):
        total = 0
        schedule = {m: rc["initial_pairs"] for m in ladder.matchups}
        rnd = 0
        while schedule:
            rnd += 1
            played = play(schedule, pool_)
            total += played
            ladder.refit()
            # random is always the least precise and hides the agents that matter, so it is not reported.
            candidates = [i for i in range(ladder.n) if i != ladder.anchor and ladder.pool[i] != "random"]
            line = f"    round {rnd}: {played} games ({total} total, {time.perf_counter() - start:.0f}s)"
            if candidates:
                worst = max(candidates, key=lambda i: ladder.se[i])
                line += f"; least precise: {ladder.pool[worst]} ±{1.96 * ladder.se[worst]:.0f} Elo"
            log(line)
            if total >= rc["max_games"]:
                log(f"    stopping at the game budget ({rc['max_games']})")
                break
            schedule = ladder.next_round(rc["target_se"], rc["max_pairs"], rc["batch_pairs"])
            # Keep every worker busy: slow agents make small rounds expensive.
            if schedule and sum(schedule.values()) < workers:
                scale = math.ceil(workers / sum(schedule.values()))
                schedule = {m: min(c * scale, rc["max_pairs"] - len(ladder.matchups[m])) for m, c in schedule.items()}
            # Respect the overall budget.
            room = (rc["max_games"] - total) // 2
            for m in list(schedule):
                schedule[m] = min(schedule[m], room)
                room -= schedule[m]
                if schedule[m] <= 0:
                    del schedule[m]
        return total

    log(f"  {len(pool)} agents, {len(ladder.matchups)} matchups, anchor {anchor} = {ladder.anchor_elo:g} Elo")
    if workers == 1:
        total = rounds(None)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=strength._init_worker) as p:
            total = rounds(p)

    draws = ladder.bootstrap(seed=derive_seed(seed, "bootstrap") & 0xFFFFFFFF)
    lo, hi = np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)
    sd = draws.std(axis=0)
    bounds = ladder.bounds()
    wins, games = ladder.totals()

    metrics = []
    order = sorted(range(ladder.n), key=lambda i: -ladder.ratings[i])
    for i in order:
        spec = pool[i]
        bound = bounds.get(i)
        note = {"lower": " (lower bound: won every game)", "upper": " (upper bound: lost every game)"}.get(bound, "")
        metrics.append(metric(
            f"ratings[{spec}].elo", f"{spec}: Elo rating{note}", round(float(ladder.ratings[i]), 1),
            unit="Elo", kind="rating", better="higher",
            ci=[round(float(lo[i]), 1), round(float(hi[i]), 1)], se=round(float(sd[i]), 1),
            detail={"games": int(games[i].sum()), "score": round(float(wins[i].sum() / games[i].sum()), 4)
                    if games[i].sum() else None, "bound": bound, "anchor": i == ladder.anchor},
        ))
        lo_i, hi_i = metrics[-1]["ci"]
        log(f"  {spec:40} {ladder.ratings[i]:7.0f} Elo  [{lo_i:.0f}, {hi_i:.0f}]{note}")

    table = []
    for (i, j), pairs in ladder.matchups.items():
        if not pairs:
            continue
        g = games[i, j]
        table.append({"a": pool[i], "b": pool[j], "pairs": len(pairs), "games": int(g),
                      "score": round(float(wins[i, j] / g), 4) if g else None,
                      "outcomes": "".join(ladder.outcomes[(i, j)])})
    metrics += [
        metric("ratings.games", "Games played for the ratings", total, unit="games",
               detail={"matchups": table, "anchor": anchor, "anchor_elo": ladder.anchor_elo,
                       "target_se": rc["target_se"], "pool": pool}),
        metric("ratings.truncated", "Rating games hitting the move limit", stats["truncated"], unit="games",
               kind="count", better="lower"),
        metric("ratings.illegal", "Rating games with an illegal move", stats["illegal"], unit="games",
               kind="count", better="lower"),
        metric("ratings.errors", "Rating games that crashed", stats["error_count"], unit="games", kind="count",
               better="lower", detail={"examples": stats["errors"]} if stats["errors"] else None),
    ]
    return metrics


def _score(outcome):
    return {"1": 1.0, "0": 0.0, "d": 0.5}.get(outcome)
