from time import perf_counter
from Board import Board as Pyboard
from board_cpp import Board as Cboard# type: ignore
from copy import deepcopy

board = Cboard()
start = perf_counter()
for i in range(10_000):
    for d1 in range(1, 7):
        for d2 in range(1, 7):
            if d1 == d2:
                board.set_dice([d1, d1, d1, d1])
            else:
                board.set_dice([d1, d2])

print(f"C time to set 100,000 dice: {perf_counter() - start}")


# with min point optimisation 1.8276 seconds
# without min point optimisation 