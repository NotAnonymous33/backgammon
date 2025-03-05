# board.pyx
import cython
from pprint import pprint
from random import randint
from itertools import permutations
from copy import deepcopy

@cython.ccall
def sign(x: cython.int) -> cython.int:
    return -1 if x < 0 else (1 if x > 0 else 0)

@cython.ccall
def list_diff(a, b):
    a_copy = a[:]
    for i in b:
        if i in a_copy:
            a_copy.remove(i)
    return a_copy

cdef class Board:
    cdef public bint rolled
    cdef public int turn
    cdef public int white_off
    cdef public int black_off
    cdef public int white_bar
    cdef public int black_bar
    cdef public bint verbose
    cdef public list positions
    cdef public list dice
    cdef public list invalid_dice
    cdef public list valid_moves
    cdef public int white_left
    cdef public int black_left
    cdef public bint passed
    cdef public bint game_over
    

    def __init__(self, board_dict=None, board_db=None, copy=False, bint verbose=False):
        if copy:
            self.verbose = verbose
            self.positions = copy.positions[:]
            self.dice = copy.dice[:]
            self.invalid_dice = copy.invalid_dice[:]
            self.valid_moves = copy.valid_moves[:]
            self.rolled = copy.rolled
            self.turn = copy.turn
            self.white_off = copy.white_off
            self.black_off = copy.black_off
            self.white_bar = copy.white_bar
            self.black_bar = copy.black_bar
            self.white_left = copy.white_left
            self.black_left = copy.black_left
            self.passed = copy.passed
            self.game_over = copy.game_over
            return
            
        self.verbose = False
        if board_dict is None and board_db is None: # no params
            self.positions = [0 for _ in range(24)]
            initial_white = [(0, 2), (11, 5), (16, 3), (18, 5)]
            initial_black = [(5, 5), (7, 3), (12, 5), (23, 2)]
            
            for pos, count in initial_white:
                self.positions[pos] = count
            for pos, count in initial_black:
                self.positions[pos] = -count
            
            self.dice = []
            self.invalid_dice = []
            self.valid_moves = []
            self.rolled = False
            self.turn = 1
            
            self.white_off = 0
            self.black_off = 0
            
            self.white_bar = 0
            self.black_bar = 0
            
            self.white_left = self.calc_white_left()
            self.black_left = self.calc_black_left()
            self.passed = self.has_passed()
            self.game_over = False
        elif board_dict is not None: # board dict
            try:
                self.positions = board_dict["positions"]
                self.dice = list(map(int, list(board_dict["dice"])))
                self.rolled = board_dict["rolled"]
                self.turn = board_dict["turn"]
                self.white_bar = board_dict["white_bar"]
                self.black_bar = board_dict["black_bar"]
                self.white_off = board_dict["white_off"]
                self.black_off = board_dict["black_off"]
                self.valid_moves = self.set_valid_moves()
                self.invalid_dice = self.get_invalid_dice()
                self.white_left = self.calc_white_left()
                self.black_left = self.calc_black_left()
                self.passed = self.has_passed()
                self.game_over = board_dict["game_over"]
            except KeyError:
                self.__init__()
        else: # board db
            try:   
                self.positions = board_db.positions
                self.dice = list(map(int, list(board_db.dice)))
                self.rolled = board_db.rolled
                self.turn = board_db.turn
                self.white_bar = board_db.white_bar
                self.black_bar = board_db.black_bar
                self.white_off = board_db.white_off
                self.black_off = board_db.black_off
                self.game_over = board_db.game_over
                self.invalid_dice = self.get_invalid_dice()
                self.set_valid_moves()
                self.white_left = self.calc_white_left()
                self.black_left = self.calc_black_left()
                self.passed = self.has_passed()
            except AttributeError:
                self.__init__()

    def __deepcopy__(self, memo):
        return Board(copy=self)

    def __str__(self):
        """
        Returns the backgammon board as a formatted string.
        """
        def format_position(position):
            """Helper to format a position with its checkers."""
            if position == 0:
                return " . "  # Empty position
            color = "W" if position > 0 else "B"
            return f" {color}{abs(position)}"  # Format as 'W2' or 'B5'

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
        dice = f"\n\nDice: {self.dice}"
        invalid_dice = f"\n\nInvalid Dice: {self.invalid_dice}"
        start = "\n\nTurn: " + ("White" if self.turn == 1 else "Black") + "-----------------------------------------------\n"
        end = "\n\n-----------------------------------------------\n"
        left = f"\n\nWhite left: {self.white_left}, Black left: {self.black_left}"
        # Combine all parts
        return start + bar + bottom + top + off_board + dice + invalid_dice + left + end


    def get_invalid_dice(self):
        if self.verbose:
            print("Board:get_invalid_dice")
        invalid_dice = self.dice[:]
        max_length = 0
        max_die = 0
        
        # returns whether there is a valid move using all dice
        def verify_permutation(board, remaining_dice, move_sequence):
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
            if board.turn == 1 and board.white_bar:
                for i in range(6):
                    board_copy = deepcopy(board)
                    if board_copy.move(-1, i):
                        if verify_permutation(board_copy, board_copy.dice, move_sequence + list_diff(board.dice, board_copy.dice)):
                            return True
                return
            if board.turn == -1 and board.black_bar:
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
                    if board_copy.move(i, 100 * board.turn):
                        if verify_permutation(board_copy, remaining_dice[1:], move_sequence + [remaining_dice[0]]):
                            return True
                
            # normal moves
            for start in range(24):
                if sign(board.positions[start]) != self.turn:
                    continue
                end = start + remaining_dice[0] * self.turn
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
    
    @cython.ccall
    def get_single_moves(self):
        moves = set()
        if self.turn == 1:
            # reentering checkers
            if self.white_bar:
                for dice in self.dice:
                    if self.is_valid(-1, dice - 1):
                        moves.add((-1, dice - 1))
                return moves
            # bearing off
            for start in range(18, 24):
                if self.is_valid(start, 100):
                    moves.add((start, 100))

            # normal moves
            for start in range(24):
                for dice in self.dice:
                    if self.is_valid(start, start + dice):
                        moves.add((start, start + dice))

            return moves

        # reentering checkers
        if self.black_bar:
            for i in range(23, 17, -1):
                if self.is_valid(-1, i):
                    moves.add((-1, i))
            return moves

        # bearing off
        for start in range(5, -1, -1):
            if self.is_valid(start, -100):
                moves.add((start, -100))

        # normal moves
        for start in range(23, -1, -1):
            for dice in self.dice:
                if self.is_valid(start, start - dice):
                    moves.add((start, start - dice))
        return moves
        
    def set_valid_moves(self):
        if self.verbose:
            print("Board:get_valid_moves")

        def dfs(board, prev_moves):
            if not board.dice:
                if prev_moves:
                    return [prev_moves]
                return []
            moves = []
            if len(board.dice) == 1:
                for move in board.get_single_moves():
                    if board.is_valid(*move):
                        moves.append(prev_moves + [move])
                return moves
            for move in board.get_single_moves():
                board_copy = deepcopy(board)
                board_copy.move(*move, bypass=True)
                moves += dfs(board_copy, prev_moves + [move])
            return moves
        
        self.valid_moves = dfs(deepcopy(self), [])
        return self.valid_moves
    
    @cython.ccall
    def can_bearoff(self):
        if self.turn == 1:
            if self.white_bar:
                return False
            for i in range(18):
                if self.positions[i] > 0:
                    return False
        else:
            if self.black_bar:
                return False
            for i in range(6, 24):
                if self.positions[i] < 0:
                    return False
        return True
        

    def convert(self):
        return {
            "positions": self.positions, 
            "turn": self.turn, 
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
    
    @cython.ccall
    def has_passed(self):
        if self.white_bar > 0 or self.black_bar > 0:
            return False
        
        lowest_white = 24
        highest_black = -1
        
        # Single pass through positions for efficiency
        for i, pos in enumerate(self.positions):
            if pos > 0:
                lowest_white = i
                break
                
        for i, pos in enumerate(self.positions[::-1]):
            if pos < 0:
                highest_black = 23 - i
                break
        
        # If all pieces are borne off for one color, it's considered passed
        if lowest_white == -1 or highest_black == 24:
            return True
        
        # Check if all black pieces are past all white pieces
        return lowest_white > highest_black
    
    @cython.ccall
    def calc_white_left(self):
        cdef int total = 0
        cdef int i
        cdef int pos
        for i, pos in enumerate(self.positions):
            if pos > 0:
                total += (24 - i) * pos
        total += self.white_bar
        return total
            
    @cython.ccall
    def calc_black_left(self):
        cdef int total = 0
        cdef int i
        cdef int pos
        for i, pos in enumerate(self.positions):
            if pos < 0:
                total += (i + 1) * pos * -1
        total += self.black_bar
        return total
    
    @cython.ccall
    def move_from_sequence(self, sequence):
        if self.verbose:
            print("Board:move_from_sequence")
        # TODO ive had errors removing this before
        # seems to work fine now, no idea why
        # sequence = [tuple(move) for move in sequence] 
        if not self.valid_moves and not sequence:
            self.swap_turn()
        if sequence not in self.valid_moves:
            return False
        for move in sequence:
            self.move(*move, bypass=True)
        self.passed = self.has_passed()
        self.white_left = self.calc_white_left()
        self.black_left = self.calc_black_left()
        if self.has_won():
            self.game_over = True
            return True
        if self.rolled and len(self.dice) == 0:
            self.swap_turn()
        return True
    
    @cython.ccall
    def has_won(self):
        return self.white_off == 15 or self.black_off == 15
    
    @cython.ccall
    def move(self, current, next, bint bypass=False):
        if not bypass and not self.is_valid(current, next):
            return False
        
        # bearing off
        if next == 100 or next == -100:
            if self.turn == 1:
                self.white_off += 1
                if (24 - current) in self.dice:
                    self.dice.remove(24 - current)
                else:
                    self.dice.remove(max(self.dice))
                self.positions[current] -= 1
            else:
                self.black_off += 1
                if (current + 1) in self.dice:
                    self.dice.remove(current + 1)
                else:
                    self.dice.remove(max(self.dice))
                self.positions[current] += 1
            # if len(self.dice) == 0:
            #     self.swap_turn() TODO im pretty sure i dont want this 
            return True
        
        # capturing a piece
        if self.turn == 1:
            if self.positions[next] == -1:
                self.positions[next] = 0
                self.black_bar += 1
        else:
            if self.positions[next] == 1:
                self.positions[next] = 0
                self.white_bar += 1
        # reentering checkers
        if current == -1:
            if self.turn == 1:
                self.white_bar -= 1
                self.dice.remove(next+1)
                self.positions[next] += 1
            else:
                self.black_bar -= 1
                self.dice.remove(24-next)    
                self.positions[next] -= 1
        else: # not reentering
            self.positions[current] -= self.turn
            self.positions[next] += self.turn
            self.dice.remove((next - current) * self.turn)
        return True
    
    @cython.ccall
    def swap_turn(self):
        if self.verbose:
            print("Board:swap_turn")
        self.turn = self.turn * -1
        self.rolled = False
        self.dice = []
        self.invalid_dice = []
        self.valid_moves = []

    @cython.ccall
    def is_valid(self, current, next): 
        # TODO: unit tests
        # current can't be empty
        if current < -1 or current > 23:
            return False
        if (next < 0 or next > 23) and abs(next) != 100:
            return False
        # bearing off 
        if next == 100 and self.turn == 1 or next == -100 and self.turn == -1:
            if not self.can_bearoff():
                return False
            if sign(self.positions[current]) != self.turn:
                return False
            if self.turn == 1:
                for dice in self.dice:
                    if dice == 24 - current:
                        return True
                # no exact dice
                if 24 - current > max(self.dice):
                    return False
                for pos in range(18, 24):
                    if self.positions[pos] > 0:
                        if current == pos:
                            return True
                        else:
                            return False
                # maybe i need to return something here, check later TODO
            else: # self.turn == -1
                for dice in self.dice:
                    if dice == current + 1:
                        return True
                if current + 1 > max(self.dice):
                    return False
                for pos in range(5, -1, -1):
                    if self.positions[pos] < 0:
                        if current == pos:
                            return True
                        else:
                            return False
                return False
            assert False, "Should not reach here"

        # reentering checkers
        if self.white_bar and self.turn == 1:
            if not (next+1) in self.dice:
                return False
            if current != -1:
                return False
            if next > 5:
                return False
            if self.positions[next] < -1:
                return False
            return True
        
        if self.black_bar and self.turn == -1:
            if current != -1:
                return False
            if next < 18:
                return False
            if not (24-next) in self.dice:
                return False
            if self.positions[next] > 1:
                return False
            return True
        
        if sign(self.positions[current]) != self.turn:
            return False
        if self.positions[next] * self.turn < -1:
            return False
        
        for dice in self.dice:
            if (next - current) * self.turn == dice:
                return True
        return False
    
    @cython.ccall
    def roll_dice(self):
        if self.verbose:
            print("Board:roll_dice")
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
        self.set_valid_moves()
        return self.dice, self.invalid_dice, self.valid_moves
    
    @cython.ccall
    def set_dice(self, dice):
        if self.verbose:
            print("Board:roll_dice")
        if self.game_over:
            return False
        if self.rolled:
            return self.dice, self.invalid_dice, self.valid_moves
        self.dice = dice
        self.rolled = True
        self.invalid_dice = self.get_invalid_dice()
        self.set_valid_moves()
        return self.dice, self.invalid_dice, self.valid_moves
    
    @cython.ccall
    def set_board(self, data):
        if "positions" in data:
            self.positions = data["positions"]
        if "dice" in data:
            self.dice = list(map(int, list(data["dice"])))
        if "turn" in data:
            self.turn = data["turn"]
        return self.convert()