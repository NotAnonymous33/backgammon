from classes.Color import Color
from pprint import pprint
from random import randint

class Board:
    positions: list[list[Color]]
    dice: list[int]
    rolled: bool
    turn: Color
    white_off: int
    black_off: int
    white_bar: int
    black_bar: int
    
    def __init__(self, board_dict=None, board_db=None):
        if board_dict is None and board_db is None:
            self.positions = [[] for i in range(24)]
            initial_white = [(0, 2), (11, 5), (16, 3), (18, 5)]
            initial_black = [(5, 5), (7, 3), (12, 5), (23, 2)]       
            
            for pos, count in initial_white:
                self.positions[pos] = [Color.WHITE for i in range(count)]
            for pos, count in initial_black:
                self.positions[pos] = [Color.BLACK for i in range(count)]
                
            self.dice = []
            self.invalid_dice = []
            self.rolled = False
            self.turn = Color.WHITE
            
            self.white_off = 0
            self.black_off = 0
            
            self.white_bar = 0
            self.black_bar = 0
        elif board_dict is not None:
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
            except KeyError:
                self.__init__()
        else:
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
            # TODO if there is an attribute(?) error, call init

    def get_moves(self):
        valid_moves = []
        invalid_dice = []
        
        # ---------- reentering checkers ----------
        if self.turn == Color.WHITE and self.white_bar > 0:
            for die in self.dice:
                if self.is_valid(-1, die-1):
                    valid_moves.append((-1, die-1, die))
            if not valid_moves:
                return valid_moves, self.dice
            return valid_moves, invalid_dice
        
        if self.turn == Color.BLACK and self.black_bar > 0:
            for die in self.dice:
                if self.is_valid(-1, 24-die):
                    valid_moves.append((-1, 24-die, die))
            if not valid_moves:
                return valid_moves, self.dice
            return valid_moves, invalid_dice
        
        # ---------- normal moves ----------
        for start in range(24):
            if self.positions[start] and self.positions[start][0] == self.turn:
                for die in self.dice:
                    end = start + die * self.turn.value
                    if 0 <= end < 24 and self.is_valid(start, end):
                        valid_moves.append((start, end, die))
        
        if not valid_moves:
            return valid_moves, self.dice
        
        # ---------- prioritise higher moves then all dice ----------
        valid_moves.sort(key=lambda x: x[2], reverse=True)
        valid_dice = {move[2] for move in valid_moves}
        invalid_dice = [die for die in self.dice if die not in valid_dice]
        
        return valid_moves, invalid_dice
    
    
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
            "white_bar": self.white_bar,
            "black_bar": self.black_bar,
            "rolled": self.rolled,
            "white_off": self.white_off,
            "black_off": self.black_off
        }        
        return ret
    
    def move(self, current, next):
        if not self.is_valid(current, next):
            print("Invalid move")
            return False
        
        # bearing off
        if next == 100:
            if self.turn == Color.WHITE:
                self.white_off += 1
                self.dice.remove(24 - current)
            else:
                self.black_off += 1
                self.dice.remove(current + 1)
            self.positions[current].pop()
            if len(self.dice) == 0 or len(self.dice) == len(self.invalid_dice):
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
        if len(self.dice) == 0 or len(self.dice) == len(self.invalid_dice):
            self.swap_turn()
        return True
    
    def swap_turn(self):
        self.turn = Color.WHITE if self.turn == Color.BLACK else Color.BLACK
        self.rolled = False
        self.dice = []
        
    
    def is_valid(self, current, next): 
        #TODO: unit tests
        # current can't be empty
        
        # bearing off 
        if self.can_bearoff():
            if self.turn == Color.WHITE:
                if current < 18:
                    return False
                if len(self.positions[current]) == 0:
                    return False
                if self.positions[current][0] != Color.WHITE:
                    return False
                print(current)
                for dice in self.dice:
                    if dice == 24 - current:
                        return True
                return False
            else:
                if current > 5:
                    return False
                if len(self.positions[current]) == 0:
                    return False
                if self.positions[current][0] != Color.BLACK:
                    return False
                for dice in self.dice:
                    if dice == 24 - current:
                        return True
                return False
        
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
            if not (24-next) in self.dice:
                return False
            if current != -1:
                return False
            if next < 18:
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
    
    def roll_dice(self):
        if self.rolled:
            return self.dice
        self.dice = [randint(1, 6), randint(1, 6)]
        if self.dice[0] == self.dice[1]:
            self.dice.append(self.dice[0])
            self.dice.append(self.dice[0])
        self.rolled = True
        valid_moves, self.invalid_dice = self.get_moves()
        return self.dice, valid_moves
    
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
            