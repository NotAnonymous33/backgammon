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
        
        self.turn = Color.WHITE
        
    def convert(self):
        positions = []
        for pos in self.positions:
            positions.append(pos.count(Color.WHITE) - pos.count(Color.BLACK))
        return {
            "positions": positions, 
            "turn": 1 if self.turn == Color.WHITE else -1, 
            "dice": self.dice
        }
    
    def move(self, current, next):
        if not self.is_valid(current, next):
            return False
        if len(self.positions[next]) == 1 and self.positions[next][0] != self.turn:
            self.positions[next].pop()
            self.positions[next].append(self.positions[current].pop())
        else:
            self.positions[next].append(self.positions[current].pop())
        self.dice.remove((next - current) * self.turn.value)
        if len(self.dice) == 0:
            self.turn = Color.WHITE if self.turn == Color.BLACK else Color.BLACK
        return True
        
    def is_valid(self, current, next): 
        #TODO: unit tests
        # current can't be empty
        if len(self.positions[current]) == 0:
            return False
        # current must be type of current player
        if self.positions[current][0] != self.turn:
            return False
        if len(self.positions[next]) > 1 and self.positions[next][0] != self.turn:
            return False
        
        # TODO: change this to iterate over dice
        for dice in self.dice:
            if (next - current) * self.turn.value == dice:
                return True
        print(current, next, self.dice)
        return False
    
    def roll_dice(self):
        self.dice = [randint(1, 6), randint(1, 6)]
        if self.dice[0] == self.dice[1]:
            self.dice.append(self.dice[0])
            self.dice.append(self.dice[0])
        return self.dice
            