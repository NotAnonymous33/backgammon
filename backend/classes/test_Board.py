import unittest
# from Board import Board
# from CBoard import Board
from board_cpp import Board

class TestBoard(unittest.TestCase):

    def setUp(self):
        self.board = Board()

    def test_get_invalid_dice_initial_setup(self):
        self.board.set_dice([1, 2])
        invalid_dice = self.board.get_invalid_dice()
        self.assertEqual(invalid_dice, [])

    def test_get_invalid_dice_no_moves(self):
        self.board.positions = [-2 for _ in range(24)]
        self.board.turn = 1
        self.board.set_dice([1, 2])
        self.assertEqual(self.board.invalid_dice, [1, 2])

    def test_get_invalid_dice_with_moves(self):
        self.board.positions = [2 if i in [0, 11, 16, 18] else 0 for i in range(24)]
        self.board.turn = 1
        self.board.set_dice([1, 2])
        self.assertEqual(self.board.invalid_dice, [])

    def test_from_the_paper(self):
        self.board.turn = -1
        self.board.black_bar = 2
        
        positions = [0 for _ in range(24)]
        positions[0] = -2
        positions[2] = 2
        positions[4] = -1
        positions[5] = 2
        positions[9] = -1
        positions[20] = 2    
        self.board.positions = positions
        
        self.board.set_dice([2, 2, 2, 2])
        self.assertEqual(self.board.invalid_dice, [2])
        
    def test_get_invalid_dice_reentering_off(self):
        self.board.positions = [-2 if i in range(6, 24) else 0 for i in range(24)]
        self.board.turn = 1
        self.board.white_bar = 1
        self.board.set_dice([1, 6])
        self.assertEqual(self.board.invalid_dice, [1])
             
    
    def test_get_1invalid_dice_reentering_checkers(self):
        self.board = Board()
        self.board.positions = [-2 if i in range(6, 24) else 0 for i in range(24)]
        self.board.turn = 1
        self.board.white_bar = 1
        self.board.set_dice([1, 6])
        self.assertEqual(self.board.invalid_dice, [1])

    def test_get_single_moves_initial_setup(self):
        self.board = Board()
        self.board.turn = 1
        self.board.set_dice([1, 2])
        single_moves = self.board.get_single_moves()
        expected_moves = {(0, 1), (0, 2), (11, 13), (16, 17), (16, 18), (18, 19), (18, 20)}
        self.assertEqual(single_moves, expected_moves)
    
    def test_get_single_moves_reentering_checkers(self):
        self.board.positions = [-2 if i in range(6, 24) else 0 for i in range(24)]
        self.board.turn = 1
        self.board.white_bar = 1
        self.board.set_dice([1, 6])
        single_moves = self.board.get_single_moves()
        expected_moves = {(-1, 5)}
        self.assertEqual(single_moves, expected_moves)

    
    def test_get_single_moves_black_turn(self):
        self.board = Board()
        self.board.turn = -1
        self.board.set_dice([1, 2])
        single_moves = self.board.get_single_moves()
        expected_moves = {(5, 4), (5, 3), (7, 6), (7, 5), (12, 10), (23, 22), (23, 21)}
        self.assertEqual(single_moves, expected_moves)
    
    def test_get_single_moves_black_reentering_checkers(self):
        self.board.positions = [2 if i in range(0, 18) else 0 for i in range(24)]
        self.board.turn = -1
        self.board.black_bar = 1
        self.board.set_dice([1, 6])
        
        single_moves = self.board.get_single_moves()
        expected_moves = {(-1, 18)}
        self.assertEqual(single_moves, expected_moves)
    
    def test_get_single_moves_bearing_off(self):
        self.board.positions = [0 if i < 18 else 2 for i in range(24)]
        self.board.turn = 1
        self.board.set_dice([1, 2])
        single_moves = self.board.get_single_moves()
        expected_moves = {(18, 19), (18, 20), (19, 20), (19, 21), (20, 21), (20, 22), (21, 22), (21, 23), (22, 23), (22, 100), (23, 100)}
        self.assertEqual(single_moves, expected_moves)
        
    def test_get_single_moves_bearing_off_black(self):
        self.board.positions = [0 for _ in range(24)]
        self.board.black_bar = 1
        self.board.turn = -1
        self.board.set_dice([1, 2])
        single_moves = self.board.get_single_moves()
        expected_moves = {(-1, 23), (-1, 22)}
        self.assertEqual(single_moves, expected_moves)


    def test_get_valid_moves_no_moves(self):
        self.board.positions = [-2 for _ in range(24)]
        self.board.turn = 1
        self.board.set_dice([1, 2])
        self.assertEqual(self.board.valid_moves, [])

    def test_get_valid_moves_reentering_1checker(self):
        self.board.positions = [-2 if i in range(6, 24) else 0 for i in range(24)]
        self.board.turn = 1
        self.board.white_bar = 1
        self.board.set_dice([3, 6])
        expected_moves = [[(-1, 5)]]
        self.assertCountEqual(self.board.valid_moves, expected_moves)
        
        
    def test_get_valid_moves_reentering_2checkers(self):
        self.board.positions = [-2 if i in range(6, 24) else 0 for i in range(24)]
        self.board.turn = 1
        self.board.white_bar = 2
        self.board.set_dice([3, 6])
        self.board.set_valid_moves()
        expected_moves = [[(-1, 2), (-1, 5)], [(-1, 5), (-1, 2)]]
        self.assertCountEqual(self.board.valid_moves, expected_moves)

    def test_get_valid_moves_reentering_plus_one(self):
        self.board = Board()
        self.board.white_bar = 1
        self.board.set_dice([1, 2])
        self.board.set_valid_moves()
        expected_moves = [[(-1, 0), (0, 2)], [(-1, 0), (11, 13)], [(-1, 0), (16, 18)], [(-1, 0), (18, 20)],
                          [(-1, 1), (0, 1)], [(-1, 1), (1, 2)], [(-1, 1), (16, 17)], [(-1, 1), (18, 19)]]
        self.assertCountEqual(self.board.valid_moves, expected_moves)

    def test_has_passed_initial_setup(self):
        self.assertFalse(self.board.has_passed())

    def test_has_passed_white_bar(self):
        self.board.white_bar = 1
        self.assertFalse(self.board.has_passed())

    def test_has_passed_black_bar(self):
        self.board.black_bar = 1
        self.assertFalse(self.board.has_passed())

    def test_has_passed_all_black_past_white(self):
        positions = [0 for _ in range(24)]
        positions[0] = -2
        positions[1] = -2
        positions[22] = 2
        positions[23] = 2
        self.board.positions = positions
        self.assertTrue(self.board.has_passed())

    def test_has_passed_not_all_black_past_white(self):
        positions = [0 for _ in range(24)]
        positions[0] = -2
        positions[2] = -2
        positions[1] = 2
        self.board.positions = positions
        self.assertFalse(self.board.has_passed())

    def test_has_passed_white_off(self):
        self.board.positions = [0 for _ in range(24)]
        self.board.white_off = 15
        self.assertTrue(self.board.has_passed())

    def test_has_passed_black_off(self):
        self.board.positions = [0 for _ in range(24)]
        self.board.black_off = 15
        self.assertTrue(self.board.has_passed())



if __name__ == '__main__':
    unittest.main()