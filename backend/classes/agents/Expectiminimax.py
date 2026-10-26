"""Expectiminimax search with a value network at the leaves.

The agent looks ``ply`` moves ahead, averaging over all 21 dice rolls
between moves, and scores the positions at the end of the search with the
network:

    ply=1  greedy: play the afterstate the network likes best (as ``nn``)
    ply=2  for each candidate play, the opponent's best reply to each of
           their 21 rolls, weighted by the roll's probability
    ply=3  the same, then the mover's best reply to each of their next
           21 rolls

Full-width search is far too slow at ply 3 (about 20 x 21 x 20 x 21 x 20
leaves), so moves are pruned forward the way gnubg's move filters do: at
every move choice that is searched deeper, the network ranks the candidates
and only the best few are expanded:

* at the root, up to ``width`` plays within ``margin`` winning chance of the
  network's favourite;
* at inner move choices (ply 3's opponent replies), up to ``reply_width``.

Leaf evaluations are batched into single network calls, and identical
positions reached by different sequences or branches are evaluated once.

The search is deterministic: no dice are drawn, every roll is enumerated.

Values are P(side to roll wins). A position whose game is over is worth 1
to the side that just moved.
"""
import numpy as np

# The 21 distinct rolls with their probabilities; doubles play four times.
ROLLS = [([a, b], 2 / 36) for a in range(6, 0, -1) for b in range(a - 1, 0, -1)] + \
        [([a] * 4, 1 / 36) for a in range(6, 0, -1)]


def _key(board):
    """Hashable position with the side to roll; ``_encode_batch`` reads features straight from it."""
    return (tuple(board.positions), board.white_bar, board.black_bar, board.white_off, board.black_off, board.turn)


def _finished(key):
    return key[3] == 15 or key[4] == 15


def _children(board):
    """Distinct afterstates of ``board`` (dice set) as ``{key: (move, child)}``."""
    moves = board.valid_moves
    if not moves:
        child = board.clone()
        child.move_from_sequence([])
        return {_key(child): ([], child)}
    out = {}
    for move in moves:
        child = board.clone()
        child.move_from_sequence(move)
        out.setdefault(_key(child), (move, child))
    return out


def _encode_batch(keys, canonical):
    """``features._encode`` for many position keys at once, from each side to roll's view if ``canonical``."""
    n = len(keys)
    pos = np.array([k[0] for k in keys], dtype=np.int8).reshape(n, 24)
    extra = np.array([k[1:5] for k in keys], dtype=np.float32).reshape(n, 4)
    if canonical:
        black = np.array([k[5] == -1 for k in keys])
        pos[black] = -pos[black, ::-1]
        extra[black] = extra[black][:, [1, 0, 3, 2]]
    own, opp = pos >= 2, pos <= -2
    own_prime = np.zeros(n, dtype=np.int8)
    opp_prime = np.zeros(n, dtype=np.int8)
    own_run = np.zeros(n, dtype=np.int8)
    opp_run = np.zeros(n, dtype=np.int8)
    for j in range(24):
        own_run = (own_run + 1) * own[:, j]
        opp_run = (opp_run + 1) * opp[:, j]
        np.maximum(own_prime, own_run, out=own_prime)
        np.maximum(opp_prime, opp_run, out=opp_prime)
    rows = np.empty((n, 34), dtype=np.float32)
    rows[:, :24] = pos / 15.0
    rows[:, 24:28] = extra / 15.0
    rows[:, 28] = (pos == 1).sum(axis=1) / 15.0
    rows[:, 29] = (pos == -1).sum(axis=1) / 15.0
    rows[:, 30] = own_prime / 6.0
    rows[:, 31] = opp_prime / 6.0
    rows[:, 32] = own[:, 18:].sum(axis=1) / 6.0
    rows[:, 33] = opp[:, :6].sum(axis=1) / 6.0
    return rows


class ExpectiminimaxAgent:
    def __init__(self, net, ply=2, width=8, margin=0.12, reply_width=3):
        """
        Args:
            net: value network with ``schema`` and ``evaluate(rows)``, e.g. ``bench.valuenet.ValueNet``.
            ply: moves searched, counting the one being chosen (1 is greedy).
            width: most root plays searched beyond the network's own ranking.
            margin: root plays further than this below the network's favourite are not searched.
            reply_width: most replies searched at inner move choices (ply 3 and deeper).
        """
        if ply < 1:
            raise ValueError(f"ply must be at least 1, got {ply}")
        self.net = net
        self.schema = net.schema
        self.ply = ply
        self.width = width
        self.margin = margin
        self.reply_width = reply_width
        self.static_cache = {}
        self.expect_cache = {}
        self.evaluations = 0  # network evaluations in the last decision

    def select_move(self, board):
        moves = board.valid_moves
        if not moves:
            return []
        if len(moves) == 1:
            return moves[0]
        # Positions recur between turns only by chance, so caches are per decision.
        self.static_cache.clear()
        self.expect_cache.clear()
        self.evaluations = 0
        kids = _children(board)
        if len(kids) == 1:
            return next(iter(kids.values()))[0]
        keys = list(kids)
        values = self._mover_values(keys)
        best = int(np.argmax(values))
        if self.ply == 1 or values[best] >= 1.0:
            return kids[keys[best]][0]
        order = np.argsort(-values, kind="stable")[:self.width]
        searched = [keys[i] for i in order if values[i] >= values[best] - self.margin]
        if len(searched) == 1:
            return kids[keys[best]][0]
        deep = 1.0 - self._expect(searched, [kids[k][1] for k in searched], self.ply - 1)
        return kids[searched[int(np.argmax(deep))]][0]

    def _static(self, keys):
        """The network's P(side to roll wins) for unfinished positions."""
        cache = self.static_cache
        todo = list({k: None for k in keys if k not in cache})
        if todo:
            raw = self.net.evaluate(_encode_batch(todo, self.schema.canonical)).astype(np.float64)
            self.evaluations += len(todo)
            if not self.schema.canonical:  # white-34 outputs P(white wins)
                black = np.array([k[5] == -1 for k in todo])
                raw[black] = 1.0 - raw[black]
            cache.update(zip(todo, raw.tolist()))
        return np.array([cache[k] for k in keys])

    def _mover_values(self, keys):
        """Each afterstate's value to the side that just moved, by the network."""
        values = np.ones(len(keys))
        live = [i for i, k in enumerate(keys) if not _finished(k)]
        if live:
            values[live] = 1.0 - self._static([keys[i] for i in live])
        return values

    def _expect(self, keys, states, depth):
        """P(side to roll wins) for unfinished positions, searching ``depth`` more moves."""
        if depth == 0:
            return self._static(keys)
        cache = self.expect_cache
        todo = list({k: s for k, s in zip(keys, states) if (k, depth) not in cache}.items())
        if not todo:
            return np.array([cache[(k, depth)] for k in keys])

        # Every (state, roll) is one move choice, a group of afterstates.
        groups, flat_keys, flat = [], [], []
        for i, (_, state) in enumerate(todo):
            for dice, p in ROLLS:
                rolled = state.clone()
                rolled.set_dice(dice)
                kids = _children(rolled)
                groups.append((i, p, len(flat_keys), len(kids)))
                flat_keys.extend(kids)
                if depth >= 2:
                    flat.extend(child for _, child in kids.values())
        values = self._mover_values(flat_keys)

        if depth >= 2:
            # Search the network's favourite replies one move deeper; they alone compete.
            searched = np.where(values >= 1.0, 1.0, -np.inf)
            deeper = []
            for _, _, start, n in groups:
                group = values[start:start + n]
                if group.max() < 1.0:  # a win needs no search
                    deeper.extend(start + j for j in np.argsort(-group, kind="stable")[:self.reply_width])
            if deeper:
                searched[deeper] = 1.0 - self._expect([flat_keys[j] for j in deeper], [flat[j] for j in deeper],
                                                      depth - 1)
            values = searched

        totals = np.zeros(len(todo))
        for i, p, start, n in groups:
            totals[i] += p * values[start:start + n].max()
        for (k, _), v in zip(todo, totals.tolist()):
            cache[(k, depth)] = v
        return np.array([cache[(k, depth)] for k in keys])
