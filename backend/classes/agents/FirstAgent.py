# from ..Board import Board


class FirstAgent:
    def select_move(self, board):
        if len(board.valid_moves) == 0:
            return []
        return board.valid_moves[0]
    