try:
    from Color import Color
except:
    from classes.Color import Color
from pprint import pprint
from random import randint
from itertools import permutations
from copy import deepcopy


def list_diff(a, b):
    a_copy = a[:]
    for i in b:
        if i in a_copy:
            a_copy.remove(i)
    return a_copy

class Board:
    rolled: bool
    turn: Color
    white_off: int
    black_off: int
    white_bar: int
    black_bar: int
    
    def __init__(self, board_dict=None, board_db=None, verbose=False):
        self.verbose = False
        if board_dict is None and board_db is None: # no params
            self.positions = [[] for i in range(24)]
            initial_white = [(0, 2), (11, 5), (16, 3), (18, 5)]
            initial_black = [(5, 5), (7, 3), (12, 5), (23, 2)]
            
            # initial_white = [(11, 4), (16, 2), (17, 1), (18, 4), (20, 2)]
            # initial_black = [(1, 2), (2, 1), (4, 1), (5, 3), (6, 1), (7, 2), (12, 3), (21, 1)]
            
            # game over testing initial
            # initial_white = [(22, 2)]
            # initial_black = [(0, 2)]
                   
            
            for pos, count in initial_white:
                self.positions[pos] = [Color.WHITE for i in range(count)]
            for pos, count in initial_black:
                self.positions[pos] = [Color.BLACK for i in range(count)]
                
            self.dice = []
            self.invalid_dice = []
            self.valid_moves = []
            self.rolled = False
            self.turn = Color.WHITE
            
            self.white_off = 0
            self.black_off = 0
            
            self.white_bar = 0
            self.black_bar = 0
            
            self.game_over = False
        elif board_dict is not None: # board dict
            try:
                self.positions = []
                for pos in board_dict["positions"]:
                    if pos > 0:
                        self.positions.append([Color.WHITE for _ in range(pos)])
                    else:
                        self.positions.append([Color.BLACK for _ in range(-pos)])
                self.dice = list(map(int, list(board_dict["dice"])))
                self.rolled = board_dict["rolled"]
                self.turn = Color.WHITE if board_dict["turn"] == 1 else Color.BLACK
                self.white_bar = board_dict["white_bar"]
                self.black_bar = board_dict["black_bar"]
                self.white_off = board_dict["white_off"]
                self.black_off = board_dict["black_off"]
                self.valid_moves = self.get_valid_moves()
                self.invalid_dice = self.get_invalid_dice()
                self.game_over = board_dict["game_over"]
            except KeyError:
                self.__init__()
        else: # board db
            self.positions = []
            for pos in board_db.positions:
                if pos > 0:
                    self.positions.append([Color.WHITE for _ in range(pos)])
                else:
                    self.positions.append([Color.BLACK for _ in range(-pos)])
            self.dice = list(map(int, list(board_db.dice)))
            self.rolled = board_db.rolled
            self.turn = Color.WHITE if board_db.turn == 1 else Color.BLACK
            self.white_bar = board_db.white_bar
            self.black_bar = board_db.black_bar
            self.white_off = board_db.white_off
            self.black_off = board_db.black_off
            self.game_over = board_db.game_over
            self.invalid_dice = self.get_invalid_dice()
            self.valid_moves = self.get_valid_moves()
            # TODO if there is an attribute(?) error, call init
            
    def __str__(self):
        """
        Returns the backgammon board as a formatted string.
        """
        def format_position(position):
            """Helper to format a position with its checkers."""
            if len(position) == 0:
                return " . "  # Empty position
            color = "W" if position[0] == Color.WHITE else "B"
            return f" {color}{len(position)}"  # Format as 'W2' or 'B5'

        # Top row (positions 12-23)
        top_row = [format_position(self.positions[i]) for i in range(12, 24)]
        top = (
            "\n\n 13  14  15  16  17  18 | 19  20  21  22  23  24 \n"
            "-------------------------------------------------\n"
            + " ".join(top_row[:6]) + " | " + " ".join(top_row[6:])
        )

        # Middle section (bar)
        bar = f"\n\nBar:\nWhite: {self.white_bar}, Black: {self.black_bar}\n"

        # Bottom row (positions 11-0, reversed)
        bottom_row = [format_position(self.positions[i]) for i in range(11, -1, -1)]
        bottom = (
            " 12  11  10   9   8   7 |  6   5   4   3   2   1 \n"
            "-------------------------------------------------\n"
            + " ".join(bottom_row[:6]) + " | " + " ".join(bottom_row[6:])
        )

        # Off-board area
        off_board = f"\n\nOff-board:\nWhite: {self.white_off}, Black: {self.black_off}"

        # Combine all parts
        return bar + bottom + top + off_board

    def get_invalid_dice(self):
        self.verbose and print("Board:get_invalid_dice")
        invalid_dice = self.dice[:]
        max_length = 0
        max_die = 0
        
        # returns whether or not there is a valid move using all dice
        def verify_permutation(board: Board, remaining_dice, move_sequence):
            move_sequence = move_sequence[:]
            nonlocal max_length
            nonlocal max_die
            nonlocal invalid_dice
            # base case
            # all dice are useable
            if not remaining_dice:
                invalid_dice = []
                return True            
            # there are dice remaining
            
            # if you can move more dice than current best, replace
            if len(move_sequence) > max_length:
                max_length = len(move_sequence)
                invalid_dice = remaining_dice

            # if you can move the same number of dice but the max die is greater, replace
            if len(move_sequence) == max_length and move_sequence and max(move_sequence) > max_die:
                max_die = max(move_sequence)
                invalid_dice = remaining_dice
            
            # reentering moves
            if board.turn == Color.WHITE and board.white_bar > 0:
                for i in range(6):
                    board_copy = deepcopy(board)
                    if board_copy.move(-1, i):
                        if verify_permutation(board_copy, board_copy.dice, move_sequence + list_diff(board.dice, board_copy.dice)):
                            return True
                return
            if board.turn == Color.BLACK and board.black_bar > 0:
                for i in range(23, 17, -1):
                    board_copy = deepcopy(board)
                    if board_copy.move(-1, i):
                        if verify_permutation(board_copy, board_copy.dice, move_sequence + list_diff(board.dice, board_copy.dice)):
                            return True
                return
            
            # bearing off
            if board.can_bearoff():
                for i in range(24):
                    board_copy = deepcopy(board)
                    if board_copy.move(i, 100 * board.turn.value):
                        if verify_permutation(board_copy, remaining_dice[1:], move_sequence + [remaining_dice[0]]):
                            return True
                
            # normal moves
            for start in range(24):
                if not (len(board.positions[start]) and board.positions[start][0] == self.turn):
                    continue
                end = start + remaining_dice[0] * self.turn.value
                if board.is_valid(start, end):
                    board_copy = deepcopy(board)
                    if board_copy.move(start, end):
                        if verify_permutation(board_copy, remaining_dice[1:], move_sequence + [remaining_dice[0]]):
                            return True         
        
        if verify_permutation(deepcopy(self), self.dice, []):
            return []
        
        
        for die in invalid_dice:
            self.dice.remove(die)
        return invalid_dice
    
    def get_single_moves(self):
        moves = set()
        # TODO: can maybe make more efficient by moving only by dice
        if self.turn == Color.WHITE:
            # reentering checkers
            if self.white_bar > 0:
                for i in range(6):
                    if self.is_valid(-1, i):
                        moves.add((-1, i))
                return moves
            # normal moves
            for start in range(24):
                for end in range(start + 1, 24):
                    if self.is_valid(start, end):
                        moves.add((start, end))
                if self.is_valid(start, 100):
                    moves.add((start, 100))
            return moves
        
        if self.black_bar > 0:
            for i in range(23, 17, -1):
                if self.is_valid(-1, i):
                    moves.add((-1, i))
            return moves
        for start in range(23, -1, -1):
            for end in range(start - 1, -1, -1):
                if self.is_valid(start, end):
                    moves.add((start, end))
            if self.is_valid(start, -100):
                moves.add((start, -100))
        return moves
        
    def get_valid_moves(self):
        self.verbose and print("Board:get_valid_moves")
        def dfs(board: Board, prev_moves):
            if not board.dice:
                return [prev_moves]
            moves = []
            for move in board.get_single_moves():
                board_copy = deepcopy(board)
                board_copy.move(*move)
                moves += dfs(board_copy, prev_moves + [move])
            return moves
        
        dfs_moves = dfs(deepcopy(self), [])
        return dfs_moves
    
    
    def can_bearoff(self):
        if self.turn == Color.WHITE:
            if self.white_bar > 0:
                return False
            for i in range(18):
                if len(self.positions[i]) > 0 and self.positions[i][0] == Color.WHITE:
                    return False
        else:
            if self.black_bar > 0:
                return False
            for i in range(6, 24):
                if len(self.positions[i]) > 0 and self.positions[i][0] == Color.BLACK:
                    return False
        return True
        
    def convert(self):
        positions = []
        for pos in self.positions:
            positions.append(pos.count(Color.WHITE) - pos.count(Color.BLACK))
        ret = {
            "positions": positions, 
            "turn": 1 if self.turn == Color.WHITE else -1, 
            "dice": "".join(str(d) for d in self.dice),
            "invalid_dice": "".join(str(d) for d in self.invalid_dice),
            "white_bar": self.white_bar,
            "black_bar": self.black_bar,
            "rolled": self.rolled,
            "white_off": self.white_off,
            "black_off": self.black_off,
            "valid_moves": self.valid_moves,
            "game_over": self.game_over
        }        
        return ret
    
    def move_from_sequence(self, sequence):
        # TODO: you can probably make partial moves which should not be a thing i imagine
        # i mean theres nothing wrong with it exactly i just dont like the idea of it
        # also might mess up some stuff frontend based off my assumptions when writing 
        # maybe not because i thankfully consider variable number of dice
        # doesnt matter, just need to remove it in the future pls thank you future ismail
        self.verbose and print("Board:move_from_sequence")
        for move in sequence:
            if not self.move(*move):
                return False
        if self.has_won():
            self.game_over = True
            return True
        if self.rolled and len(self.dice) == 0:
            self.swap_turn()
    
    def has_won(self):
        return self.white_off == 15 or self.black_off == 15
    
    def move(self, current, next):
        if not self.is_valid(current, next):
            return False
        
        # bearing off
        if next == 100 or next == -100:
            if self.turn == Color.WHITE:
                self.white_off += 1
                if (24 - current) in self.dice:
                    self.dice.remove(24 - current)
                else:
                    self.dice.remove(max(self.dice))
            else:
                self.black_off += 1
                if (current + 1) in self.dice:
                    self.dice.remove(current + 1)
                else:
                    self.dice.remove(max(self.dice))
            self.positions[current].pop()
            if len(self.dice) == 0:
                self.swap_turn()
            return True
        
        # reentering checkers
        if current == -1:
            if self.turn == Color.WHITE:
                self.white_bar -= 1
                self.dice.remove(next+1)
            else:
                self.black_bar -= 1
                self.dice.remove(24-next)
            if len(self.positions[next]) == 1 and self.positions[next][0] != self.turn:
                if self.turn == Color.WHITE:
                    self.black_bar += 1
                else:
                    self.white_bar += 1
                self.positions[next].pop()
            self.positions[next].append(self.turn)        
        else: # not reentering
            # eat the piece
            if len(self.positions[next]) == 1 and self.positions[next][0] != self.turn:
                if self.turn == Color.WHITE:
                    self.black_bar += 1
                else:
                    self.white_bar += 1
                self.positions[next].pop()
                self.positions[next].append(self.positions[current].pop())
            else:
                self.positions[next].append(self.positions[current].pop())
            self.dice.remove((next - current) * self.turn.value)
        return True
    
    def swap_turn(self):
        self.verbose and print("Board:swap_turn")
        self.turn = Color.WHITE if self.turn == Color.BLACK else Color.BLACK
        self.rolled = False
        self.dice = []
        self.invalid_dice = []
        self.valid_moves = []
        
    
    def is_valid(self, current, next): 
        # TODO: unit tests
        # current can't be empty
        if current not in range(24) and current != -1:
            return False
        if next not in range(24) and abs(next) != 100:
            return False
        # bearing off 
        if (next == 100 and self.turn == Color.WHITE or next == -100 and self.turn == Color.BLACK):
            if not self.can_bearoff():
                return False
            if len(self.positions[current]) == 0:
                return False
            if self.positions[current][0] != self.turn:
                return False
            if self.turn == Color.WHITE:
                for dice in self.dice:
                    if dice == 24 - current:
                        return True
                # no exact dice
                if 24 - current > max(self.dice):
                    return False
                for pos in range(18, 24):
                    if len(self.positions[pos]) > 0 and self.positions[pos][0] == Color.WHITE:
                        if current == pos:
                            return True
                        else:
                            return False
                # maybe i need to return something here, check later TODO
            else: # self.turn == Color.BLACK
                for dice in self.dice:
                    if dice == current + 1:
                        return True
                if current + 1 > max(self.dice):
                    return False
                for pos in range(5, -1, -1):
                    if len(self.positions[pos]) > 0 and self.positions[pos][0] == Color.BLACK:
                        if current == pos:
                            return True
                        else:
                            return False
                return False
            # TODO im pretty sure theres a problem where you can bear off with a value greater than the dice
            # i think im just too stupid to fix this 
        
        # reentering checkers
        if self.turn == Color.WHITE and self.white_bar > 0:
            if not (next+1) in self.dice:
                return False
            if current != -1:
                return False
            if next > 5:
                return False
            if len(self.positions[next]) > 1 and self.positions[next][0] == Color.BLACK:
                return False
            return True
        
        if self.turn == Color.BLACK and self.black_bar > 0:
            if current != -1:
                return False
            if next < 18:
                return False
            if not (24-next) in self.dice:
                return False
            if len(self.positions[next]) > 1 and self.positions[next][0] == Color.WHITE:
                return False
            return True
        
        if len(self.positions[current]) == 0:
            return False
        # current must be type of current player
        if self.positions[current][0] != self.turn:
            return False
        if len(self.positions[next]) > 1 and self.positions[next][0] != self.turn:
            return False
        
        for dice in self.dice:
            if (next - current) * self.turn.value == dice:
                return True
        return False
    
    def roll_dice(self) -> list[int]:
        self.verbose and print("Board:roll_dice")
        if self.game_over:
            return False
        if self.rolled:
            return self.dice, self.invalid_dice, self.valid_moves
        self.dice = [randint(1, 6), randint(1, 6)]
        if self.dice[0] == self.dice[1]:
            self.dice.append(self.dice[0])
            self.dice.append(self.dice[0])
        self.rolled = True
        self.invalid_dice = self.get_invalid_dice()
        self.valid_moves = self.get_valid_moves()
        return self.dice, self.invalid_dice, self.valid_moves
    
    def set_dice(self, dice) -> list[int]:
        self.verbose and print("Board:roll_dice")
        if self.game_over:
            return False
        if self.rolled:
            return self.dice, self.invalid_dice, self.valid_moves
        self.dice = [randint(1, 6), randint(1, 6)]
        self.dice = dice
        self.rolled = True
        self.invalid_dice = self.get_invalid_dice()
        self.valid_moves = self.get_valid_moves()
        return self.dice, self.invalid_dice, self.valid_moves
    
    def set_board(self, data):
        if "positions" in data:
            self.positions = []
            for pos in data["positions"]:
                if pos > 0:
                    self.positions.append([Color.WHITE for _ in range(pos)])
                else:
                    self.positions.append([Color.BLACK for _ in range(-pos)])
        if "dice" in data:
            self.dice = list(map(int, list(data["dice"])))
        if "turn" in data:
            self.turn = Color.WHITE if data["turn"] == 1 else Color.BLACK
        return self.convert()
            