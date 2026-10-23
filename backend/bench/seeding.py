"""Deterministic seeds and dice.

The native board seeds its own generator from std::random_device, so the
benchmark never calls ``Board.roll_dice``. Dice come from a Python stream and
are applied with ``Board.set_dice``; the same seed always yields the same game
dice regardless of what the agents do with their own randomness.
"""
import hashlib
import random


def derive_seed(*parts):
    """Stable 63-bit seed from arbitrary parts (unlike hash(), stable across processes)."""
    text = "\x1f".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") >> 1


class DiceStream:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def next(self):
        d1 = self.rng.randint(1, 6)
        d2 = self.rng.randint(1, 6)
        if d1 == d2:
            return [d1, d1, d1, d1]
        return [d1, d2]


def seed_agents(seed):
    """Seed the global RNGs agents draw from (python ``random`` and torch if loaded)."""
    import sys

    random.seed(seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed & 0xFFFFFFFF)
