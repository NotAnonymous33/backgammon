from time import perf_counter
from Board import Board
from copy import deepcopy

board = Board()
for _ in range(5):
    start = perf_counter()
    for i in range(1000000): # 1 million
        b = deepcopy(board)
    print(perf_counter() - start)


    