# from ..Board import Board
import random

class RandomAgent:
    def select_move(self, board):
        valid_moves = board.get_valid_moves()
        if len(valid_moves) == 0:
            return []
        return random.choice(valid_moves)
    