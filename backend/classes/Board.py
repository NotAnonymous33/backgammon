from classes.Color import Color
from pprint import pprint
from random import randint

class Board:
    def __init__(self):
        self.positions = [[] for i in range(24)]
        initial_white = [[0, 2], [11, 5], [16, 3], [18, 5]]
        initial_black = [[5, 5], [7, 3], [12, 5], [23, 2]]        
        
        for pos, count in initial_white:
            self.positions[pos] = [Color.WHITE for i in range(count)]
        for pos, count in initial_black:
            self.positions[pos] = [Color.BLACK for i in range(count)]
            
        self.dice = [0, 0]
        self.rolled = False
        self.turn = Color.WHITE
        
        self.white_off = 0
        self.black_off = 0
        
        self.white_bar = 0
        self.black_bar = 0
        
    def convert(self):
        positions = []
        for pos in self.positions:
            positions.append(pos.count(Color.WHITE) - pos.count(Color.BLACK))
        ret = {
            "positions": positions, 
            "turn": 1 if self.turn == Color.WHITE else -1, 
            "dice": self.dice,
            "white_bar": self.white_bar,
            "black_bar": self.black_bar,
            "rolled": self.rolled,
        }
        print(ret)
        
        return ret
    
    def move(self, current, next):
        if not self.is_valid(current, next):
            return False
        
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
            if len(self.dice) == 0:
                self.swap_turn()
            return True
        
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
        if len(self.dice) == 0:
            self.swap_turn()
        return True
    
    def swap_turn(self):
        self.turn = Color.WHITE if self.turn == Color.BLACK else Color.BLACK
        self.rolled = False
    
    def is_valid(self, current, next): 
        #TODO: unit tests
        # current can't be empty
        
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
        print(current, next, self.dice)
        return False
    
    def roll_dice(self):
        if self.rolled:
            return False
        self.dice = [randint(1, 6), randint(1, 6)]
        if self.dice[0] == self.dice[1]:
            self.dice.append(self.dice[0])
            self.dice.append(self.dice[0])
        self.rolled = True
        return self.dice
            