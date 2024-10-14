from classes.Color import Color

class Board:
    def __init__(self):
        self.positions = [[]] * 24
        initial_white = [[0, 2], [11, 5], [16, 3], [18, 5]]
        initial_black = [[5, 5], [7, 3], [12, 5], [23, 2]]
        
        for pos, count in initial_white:
            self.positions[pos] = [Color.WHITE] * count
        for pos, count in initial_black:
            self.positions[pos] = [Color.BLACK] * count
        
        self.turn = Color.WHITE
        
    def convert(self):
        positions = []
        for pos in self.positions:
            positions.append(pos.count(Color.WHITE) - pos.count(Color.BLACK))
        return {"positions": positions, "turn": 1 if self.turn == Color.WHITE else 0}
            