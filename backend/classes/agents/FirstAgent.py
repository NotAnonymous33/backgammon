# from ..Board import Board


class FirstAgent:
    def select_move(self, board):
        valid_moves = board.get_valid_moves()
        if len(valid_moves) == 0:
            return []
        return board.get_valid_moves()[0]
    