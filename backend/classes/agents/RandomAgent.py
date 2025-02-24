# from ..Board import Board
import random

class RandomAgent:
    def select_move(self, board):
        if len(board.valid_moves) == 0:
            return []
        return random.choice(board.valid_moves)
    