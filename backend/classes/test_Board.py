import unittest
from Board import Board
from Color import Color

class TestBoard(unittest.TestCase):

    def setUp(self):
        self.board = Board()

    def test_get_invalid_dice_initial_setup(self):
        self.board.dice = [1, 2]
        self.board.rolled = True
        invalid_dice = self.board.get_invalid_dice()
        self.assertEqual(invalid_dice, [])

    def test_get_invalid_dice_no_moves(self):
        self.board.positions = [[Color.BLACK] * 2 for _ in range(24)]
        self.board.turn = Color.WHITE
        self.board.dice = [1, 2]
        self.board.rolled = True
        invalid_dice = self.board.get_invalid_dice()
        self.assertEqual(invalid_dice, [1, 2])

    def test_get_invalid_dice_with_moves(self):
        self.board.positions = [[Color.WHITE] * 2 if i in [0, 11, 16, 18] else [] for i in range(24)]
        self.board.turn = Color.WHITE
        self.board.dice = [1, 2]
        self.board.rolled = True
        invalid_dice = self.board.get_invalid_dice()
        self.assertEqual(invalid_dice, [])
    
    def test_from_the_paper(self):
        self.board.turn = Color.BLACK
        self.board.dice = [2, 2, 2, 2]
        self.board.rolled = True
        self.board.black_bar = 2
        
        self.board.positions = [[] for _ in range(24)]
        self.board.positions[2] = [Color.WHITE] * 2
        self.board.positions[4] = [Color.BLACK]
        self.board.positions[5] = [Color.WHITE] * 2
        self.board.positions[9] = [Color.BLACK]
        self.board.positions[20] = [Color.WHITE] * 2    
        
        invalid_dice = self.board.get_invalid_dice()
        self.assertEqual(invalid_dice, [2])
             
    
    def test_get_1invalid_dice_reentering_checkers(self):
        self.board.positions = [[Color.BLACK] * 2 if i in range(6, 24) else [] for i in range(24)]
        self.board.turn = Color.WHITE
        self.board.white_bar = 1
        self.board.dice = [1, 6]
        self.board.rolled = True
        invalid_dice = self.board.get_invalid_dice()
        self.assertEqual(invalid_dice, [1])

if __name__ == '__main__':
    unittest.main()