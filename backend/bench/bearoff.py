"""Exact two-sided bearoff database: ground truth for the end of the game.

Once both sides have every remaining checker in their home board (and no
contact is possible), the game is a pure race to bear off, and the exact
probability of winning follows from a small dynamic program. This module
solves it for up to ``MAX_CHECKERS`` checkers per side, as GNU Backgammon's
standard two-sided database does (6 checkers on 6 points: 924 configurations
per side, 853,776 positions).

It depends on neither the native board nor any agent: it has its own move
generator for home-board play, so it is also an independent check on the
native generator (tests/test_bench.py compares them).

A side's configuration is a tuple ``c`` of six counts, where ``c[k]`` is the
number of checkers ``k + 1`` pips from being borne off. ``win[a, b]`` is the
probability that the side to roll, holding ``a``, beats an opponent holding
``b``, when both play to maximise their winning chances.
"""
import itertools
import os

import numpy as np

MAX_CHECKERS = 6
POINTS = 6
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
VERSION = 1

# The 21 distinct rolls with their probabilities: doubles 1/36, others 2/36.
ROLLS = [((d1, d2), (1 if d1 == d2 else 2) / 36) for d1 in range(1, 7) for d2 in range(d1, 7)]


def configurations(max_checkers=MAX_CHECKERS):
    """Every home-board configuration with at most ``max_checkers`` checkers, by pip count."""
    configs = [c for c in itertools.product(range(max_checkers + 1), repeat=POINTS) if sum(c) <= max_checkers]
    configs.sort(key=lambda c: (pips(c), c))
    return configs


def pips(config):
    return sum((k + 1) * n for k, n in enumerate(config))


def play_die(config, die):
    """Configurations reachable by playing one die (all checkers already home)."""
    out = set()
    highest = max((k for k in range(POINTS) if config[k]), default=-1)
    if highest < 0:
        return out
    d = die - 1  # index of the point exactly ``die`` pips from off
    for k in range(POINTS):
        if not config[k]:
            continue
        if k > d:  # move within the home board
            c = list(config)
            c[k] -= 1
            c[k - die] += 1
            out.add(tuple(c))
        elif k == d or k == highest:  # bear off exactly, or from the highest point with a larger die
            c = list(config)
            c[k] -= 1
            out.add(tuple(c))
    return out


def play_roll(config, roll):
    """Configurations reachable with ``roll``, using as many dice as possible.

    With every checker home a die can always be played, so a turn uses all its
    dice unless the last checker comes off first.
    """
    d1, d2 = roll
    orders = [(d1,) * 4] if d1 == d2 else [(d1, d2), (d2, d1)]
    results = set()
    for dice in orders:
        frontier = {config}
        for die in dice:
            nxt = set()
            for c in frontier:
                if sum(c) == 0:
                    nxt.add(c)
                else:
                    nxt |= play_die(c, die)
            frontier = nxt
        results |= frontier
    return results


class BearoffDB:
    def __init__(self, configs, win):
        self.configs = configs
        self.index = {c: i for i, c in enumerate(configs)}
        self.win = win
        self.max_checkers = max(sum(c) for c in configs)

    @classmethod
    def build(cls, max_checkers=MAX_CHECKERS):
        configs = configurations(max_checkers)
        index = {c: i for i, c in enumerate(configs)}
        n = len(configs)
        empty = index[(0,) * POINTS]
        config_pips = np.array([pips(c) for c in configs])
        by_pips = {p: np.flatnonzero(config_pips == p) for p in np.unique(config_pips)}

        # For each configuration and roll: whether it can bear off completely,
        # and otherwise the (non-empty) configurations it can reach.
        moves = []
        for c in configs:
            per_roll = []
            for roll, w in ROLLS:
                succ = play_roll(c, roll)
                if (0,) * POINTS in succ:
                    per_roll.append((w, None))
                else:
                    per_roll.append((w, np.array(sorted(index[s] for s in succ))))
            moves.append(per_roll)

        # Every move lowers the mover's pip count, so win[b, a'] is always
        # known when win[a, b] is computed in order of the pair's total pips.
        win = np.zeros((n, n))
        max_pips = int(config_pips.max())
        for total in range(2, 2 * max_pips + 1):
            for a in range(n):
                if a == empty:
                    continue
                pb = total - config_pips[a]
                if pb not in by_pips or pb == 0:
                    continue
                bs = by_pips[pb]
                acc = np.zeros(len(bs))
                for w, succ in moves[a]:
                    if succ is None:
                        acc += w
                    else:
                        acc += w * (1.0 - win[np.ix_(bs, succ)].min(axis=1))
                win[a, bs] = acc
        return cls(configs, win)

    @classmethod
    def load(cls, max_checkers=MAX_CHECKERS, cache_dir=CACHE_DIR):
        """Load the cached database, building it (about a minute) on first use."""
        path = os.path.join(cache_dir, f"bearoff-{max_checkers}x{POINTS}-v{VERSION}.npy")
        configs = configurations(max_checkers)
        if os.path.exists(path):
            win = np.load(path)
            if win.shape == (len(configs), len(configs)):
                return cls(configs, win)
        db = cls.build(max_checkers)
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".tmp.npy"
        np.save(tmp, db.win)
        os.replace(tmp, path)
        return db

    def probability(self, mover, opponent):
        """P(the side to roll wins); configurations as tuples."""
        if sum(opponent) == 0:
            return 0.0
        if sum(mover) == 0:
            return 1.0
        return float(self.win[self.index[mover], self.index[opponent]])

    def covers(self, config):
        return config is not None and sum(config) <= self.max_checkers

    def board_state(self, board):
        """``(white, black)`` configurations if ``board`` is in the database, else None."""
        if board.white_bar or board.black_bar:
            return None
        white, black = sides(board.positions)
        if white is None or black is None or not (self.covers(white) and self.covers(black)):
            return None
        return white, black

    def white_win(self, board, to_move):
        """P(white wins) for a covered board with ``to_move`` (1 white, -1 black) about to roll."""
        white, black = self.board_state(board)
        if to_move == 1:
            return self.probability(white, black)
        return 1.0 - self.probability(black, white)


def sides(positions):
    """Home-board configurations of white and black; None for a side with checkers outside home."""
    white = [0] * POINTS
    black = [0] * POINTS
    w_ok = b_ok = True
    for i, n in enumerate(positions):
        if n > 0:
            if i >= 18:
                white[23 - i] += n  # index 23 is one pip from off for white
            else:
                w_ok = False
        elif n < 0:
            if i < 6:
                black[i] -= n  # index 0 is one pip from off for black
            else:
                b_ok = False
    return (tuple(white) if w_ok else None), (tuple(black) if b_ok else None)


def to_positions(white, black):
    """Inverse of ``sides``: a 24-point list for two home-board configurations."""
    pos = [0] * 24
    for k in range(POINTS):
        pos[23 - k] += white[k]
        pos[k] -= black[k]
    return pos
