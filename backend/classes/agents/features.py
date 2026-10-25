"""Feature schemas for the value network.

A schema fixes three things that must agree between training, serving and
evaluation: the input features, whose point of view they take, and what the
network's output means. Checkpoints record their schema (``describe()``), so
a loader always knows how to use the network it loaded.

``white-34`` (version 1, legacy)
    The original features, always from white's side. The output is P(white
    wins), but nothing in the input says whose turn it is, so one value has
    to cover both "white to roll" and "black to roll". Checkpoints saved
    before schemas existed use this schema.

``canonical-34`` (version 2)
    The same 34 features, computed from the point of view of the side about
    to roll: the board is mirrored for black, so "own" checkers are always
    positive and always move from point 0 towards point 23. The output is
    P(the side to roll wins). Being on roll is part of every input by
    construction, and white and black positions share one representation.

For both schemas the 34 inputs are, in order: the 24 points / 15; own bar,
opponent bar, own off, opponent off / 15; own and opponent blots / 15; own
and opponent longest run of made points / 6; own made points on 18-23 and
opponent made points on 0-5 / 6. "Own" means white for ``white-34``.
"""

FEATURE_NAMES = (
    [f"point_{i}" for i in range(24)]
    + ["own_bar", "opp_bar", "own_off", "opp_off", "own_blots", "opp_blots",
       "own_prime", "opp_prime", "own_home_points", "opp_home_points"]
)


def _encode(pos, own_bar, opp_bar, own_off, opp_off):
    """The 34 features of a board seen from the side whose checkers are positive."""
    f = [p / 15.0 for p in pos]
    own_blots = opp_blots = 0
    own_run = opp_run = own_prime = opp_prime = 0
    for p in pos:
        if p >= 2:
            own_run += 1
            opp_run = 0
            if own_run > own_prime:
                own_prime = own_run
        elif p <= -2:
            opp_run += 1
            own_run = 0
            if opp_run > opp_prime:
                opp_prime = opp_run
        else:
            own_run = opp_run = 0
            if p == 1:
                own_blots += 1
            elif p == -1:
                opp_blots += 1
    own_home = (pos[18] >= 2) + (pos[19] >= 2) + (pos[20] >= 2) + (pos[21] >= 2) + (pos[22] >= 2) + (pos[23] >= 2)
    opp_home = (pos[0] <= -2) + (pos[1] <= -2) + (pos[2] <= -2) + (pos[3] <= -2) + (pos[4] <= -2) + (pos[5] <= -2)
    f += (
        own_bar / 15.0, opp_bar / 15.0, own_off / 15.0, opp_off / 15.0,
        own_blots / 15.0, opp_blots / 15.0, own_prime / 6.0, opp_prime / 6.0,
        own_home / 6.0, opp_home / 6.0,
    )
    return f


def canonical_view(board, player):
    """``(points, own_bar, opp_bar, own_off, opp_off)`` with ``player``'s checkers positive.

    For black the board is mirrored (point i becomes 23 - i), so black's home
    board lands on 18-23 and black moves towards higher points, like white.
    """
    pos = board.positions
    if player == 1:
        return pos, board.white_bar, board.black_bar, board.white_off, board.black_off
    return [-p for p in reversed(pos)], board.black_bar, board.white_bar, board.black_off, board.white_off


class Schema:
    def __init__(self, name, version, perspective, value, description):
        self.name = name
        self.version = version
        self.perspective = perspective
        self.value = value
        self.description = description
        self.size = len(FEATURE_NAMES)

    def __repr__(self):
        return f"Schema({self.name!r}, v{self.version})"

    def describe(self):
        """What a checkpoint stores, so loaders can check they use the network correctly."""
        return {
            "name": self.name,
            "version": self.version,
            "inputs": self.size,
            "perspective": self.perspective,
            "value": self.value,
            "features": list(FEATURE_NAMES),
        }

    @property
    def canonical(self):
        return self.perspective == "side-to-roll"

    def encode(self, board, player):
        """Features of ``board`` with ``player`` about to roll."""
        if self.canonical:
            return _encode(*canonical_view(board, player))
        return _encode(board.positions, board.white_bar, board.black_bar, board.white_off, board.black_off)

    def features(self, board):
        """Features of ``board`` with ``board.turn`` about to roll."""
        return self.encode(board, board.turn)

    __call__ = features

    def afterstate_rows(self, mover, children):
        """Feature rows for the positions a mover can reach (one per child board)."""
        return [self.encode(child, -mover) for child in children]

    def best_index(self, mover, children, values):
        """Index of the move the mover should play, given the network's value of each child.

        ``values`` holds the network outputs for ``afterstate_rows``: P(white
        wins) for ``white-34`` and P(the opponent, now on roll, wins) for
        ``canonical-34``. A move that ends the game wins for the mover.
        """
        if not self.canonical:
            best, best_v = 0, None
            for i, v in enumerate(values):
                if best_v is None or (v > best_v if mover == 1 else v < best_v):
                    best, best_v = i, v
            return best
        best, best_v = 0, None
        for i, (child, v) in enumerate(zip(children, values)):
            if child.game_over:
                return i  # winning immediately beats anything the network says
            if best_v is None or v < best_v:
                best, best_v = i, v
        return best

    def win_probability(self, board, output, player):
        """P(``player`` wins) from the network ``output`` for ``board`` with ``board.turn`` to roll."""
        if board.game_over:
            return 1.0 if (board.white_off == 15) == (player == 1) else 0.0
        if self.canonical:
            return output if player == board.turn else 1.0 - output
        return output if player == 1 else 1.0 - output


LEGACY = Schema(
    "white-34", 1, "white", "P(white wins)",
    "original features from white's side; whose turn it is is not an input",
)
CANONICAL = Schema(
    "canonical-34", 2, "side-to-roll", "P(side to roll wins)",
    "features from the side about to roll (board mirrored for black)",
)
SCHEMAS = {(s.name, s.version): s for s in (LEGACY, CANONICAL)}
DEFAULT = CANONICAL


class SchemaError(ValueError):
    pass


def from_description(desc):
    """The schema a checkpoint describes; checkpoints without one predate schemas (``white-34``)."""
    if desc is None:
        return LEGACY
    key = (desc.get("name"), desc.get("version"))
    if key not in SCHEMAS:
        known = ", ".join(f"{n} v{v}" for n, v in SCHEMAS)
        raise SchemaError(f"checkpoint uses feature schema {key[0]} v{key[1]}; this code knows {known}")
    schema = SCHEMAS[key]
    if desc.get("inputs", schema.size) != schema.size:
        raise SchemaError(f"checkpoint schema {schema.name} declares {desc.get('inputs')} inputs, expected {schema.size}")
    return schema


def legacy_features(board):
    """``white-34`` features of a board (the original ``extract_features``)."""
    return LEGACY.features(board)
