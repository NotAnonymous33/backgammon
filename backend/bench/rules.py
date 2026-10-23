"""Rules invariants checked over the position suite.

These are exact counts: on a correct move generator every violation count is
zero, and any change between runs is real rather than noise.
"""
from .positions import to_board
from .store import metric

MAX_EXAMPLES = 25


def afterstate_key(board):
    return (tuple(board.positions), board.white_bar, board.black_bar, board.white_off, board.black_off)


def checker_totals(board):
    white = sum(n for n in board.positions if n > 0) + board.white_bar + board.white_off
    black = -sum(n for n in board.positions if n < 0) + board.black_bar + board.black_off
    return white, black


def afterstates(board):
    """Distinct resulting positions, plus sequences the board refused or that lost checkers."""
    keys = set()
    refused = []
    leaks = []
    for seq in board.valid_moves:
        child = board.clone()
        if child.move_from_sequence(seq) is False:
            refused.append(seq)
            continue
        if checker_totals(child) != (15, 15):
            leaks.append(seq)
        keys.add(afterstate_key(child))
    return keys, refused, leaks


def _with_dice(pos, dice):
    return to_board(dict(pos, dice=dice))


def check_position(pos):
    """Return a dict of violation lists for one suite entry."""
    out = {"order": False, "refused": 0, "leaks": 0, "mixed_lengths": False, "sequences": 0, "distinct": 0}
    board = to_board(pos)
    moves = board.valid_moves
    keys, refused, leaks = afterstates(board)
    out["sequences"] = len(moves)
    out["distinct"] = len(keys)
    out["refused"] = len(refused)
    out["leaks"] = len(leaks)
    out["mixed_lengths"] = len({len(m) for m in moves}) > 1

    d = list(pos["dice"])
    if len(d) == 2 and d[0] != d[1]:
        other_keys, _, _ = afterstates(_with_dice(pos, [d[1], d[0]]))
        out["order"] = other_keys != keys
    return out


def run(suite, log=print):
    positions = suite["positions"]
    order, refused, leaks, mixed = [], [], [], []
    sequences = distinct = 0
    for pos in positions:
        r = check_position(pos)
        sequences += r["sequences"]
        distinct += r["distinct"]
        if r["order"]:
            order.append(pos["id"])
        if r["refused"]:
            refused.append(pos["id"])
        if r["leaks"]:
            leaks.append(pos["id"])
        if r["mixed_lengths"]:
            mixed.append(pos["id"])
    log(f"  rules: {len(positions)} positions, {len(order)} dice-order mismatches")

    def violations(key, label, ids):
        return metric(
            key, label, len(ids), unit="positions", kind="count", better="lower",
            detail={"examples": ids[:MAX_EXAMPLES]},
        )

    return [
        violations("rules.dice_order_mismatch", "Legal turns differ by dice order", order),
        violations("rules.mixed_length_turns", "Turns mixing sequence lengths (max-dice rule)", mixed),
        violations("rules.refused_sequences", "Listed moves the board refuses to play", refused),
        violations("rules.checker_leaks", "Moves that create or lose checkers", leaks),
        metric("rules.positions", "Positions checked", len(positions), unit="positions"),
        metric(
            "rules.duplicate_afterstates", "Sequences reaching an already-seen position",
            round(1 - distinct / sequences, 4) if sequences else 0.0, unit="fraction",
        ),
    ]
