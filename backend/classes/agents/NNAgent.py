try:
    from classes.CBoard import Board # type: ignore
    from classes.agents.RandomAgent import RandomAgent
except:
    from CBoard import Board # type: ignore
    from RandomAgent import RandomAgent
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import multiprocessing as mp

# from random import randint
# from copy import deepcopy

# def sign(x: int) -> int:
#     return -1 if x < 0 else (1 if x > 0 else 0)

# def list_diff(a: list, b: list) -> list:
#     a_copy = a[:]
#     for i in b:
#         if i in a_copy:
#             a_copy.remove(i)
#     return a_copy

# class Board:
#     rolled: bool
#     turn: int
#     white_off: int
#     black_off: int
#     white_bar: int
#     black_bar: int
#     white_left: int
#     black_left: int
#     passed: bool
#     game_over: bool
#     positions: list[int]
#     dice: list[int]
#     invalid_dice: list[int]
#     valid_moves: list[list[tuple[int, int]]]
    
#     def __init__(self, board_dict=None, board_db=None, copy=None, verbose=False):
#         self.verbose = verbose
#         if copy:
#             self.positions = copy.positions[:]
#             self.dice = copy.dice[:]
#             self.invalid_dice = copy.invalid_dice[:]
#             self.valid_moves = copy.valid_moves[:]
#             self.rolled = copy.rolled
#             self.turn = copy.turn
#             self.white_off = copy.white_off
#             self.black_off = copy.black_off
#             self.white_bar = copy.white_bar
#             self.black_bar = copy.black_bar
#             self.white_left = copy.white_left
#             self.black_left = copy.black_left
#             self.passed = copy.passed
#             self.game_over = copy.game_over
#             return
#         if board_db:
#             try:   
#                 self.positions = board_db.positions
#                 self.dice = list(map(int, list(board_db.dice)))
#                 self.rolled = board_db.rolled
#                 self.turn = board_db.turn
#                 self.white_bar = board_db.white_bar
#                 self.black_bar = board_db.black_bar
#                 self.white_off = board_db.white_off
#                 self.black_off = board_db.black_off
#                 self.game_over = board_db.game_over
#                 self.invalid_dice = self.get_invalid_dice()
#                 self.set_valid_moves()
#                 self.white_left = self.calc_white_left()
#                 self.black_left = self.calc_black_left()
#                 self.passed = self.has_passed()
#             except AttributeError:
#                 self.__init__()
#             return
#         if board_dict:
#             try:
#                 self.positions = board_dict["positions"]
#                 self.dice = list(map(int, list(board_dict["dice"])))
#                 self.rolled = board_dict["rolled"]
#                 self.turn = board_dict["turn"]
#                 self.white_bar = board_dict["white_bar"]
#                 self.black_bar = board_dict["black_bar"]
#                 self.white_off = board_dict["white_off"]
#                 self.black_off = board_dict["black_off"]
#                 self.set_valid_moves()
#                 self.invalid_dice = self.get_invalid_dice()
#                 self.white_left = self.calc_white_left()
#                 self.black_left = self.calc_black_left()
#                 self.passed = self.has_passed()
#                 self.game_over = board_dict["game_over"]
#             except KeyError:
#                 self.__init__()# board db
#             return
#         self.positions = [0 for _ in range(24)]
#         initial_white = [(0, 2), (11, 5), (16, 3), (18, 5)]
#         initial_black = [(5, 5), (7, 3), (12, 5), (23, 2)]
        
#         for pos, count in initial_white:
#             self.positions[pos] = count
#         for pos, count in initial_black:
#             self.positions[pos] = -count
        
#         self.dice = []
#         self.invalid_dice = []
#         self.valid_moves = []
#         self.rolled = False
#         self.turn = 1
        
#         self.white_off = 0
#         self.black_off = 0
        
#         self.white_bar = 0
#         self.black_bar = 0
        
#         self.white_left = self.calc_white_left()
#         self.black_left = self.calc_black_left()
#         self.passed = self.has_passed()
#         self.game_over = False
        

#     def __deepcopy__(self, memo):
#         return Board(copy=self)

#     def __str__(self):
#         """
#         Returns the backgammon board as a formatted string.
#         """
#         def format_position(position):
#             """Helper to format a position with its checkers."""
#             if position == 0:
#                 return " . "  # Empty position
#             color = "W" if position > 0 else "B"
#             return f" {color}{abs(position)}"  # Format as 'W2' or 'B5'

#         # Top row (positions 12-23)
#         top_row = [format_position(self.positions[i]) for i in range(12, 24)]
#         top = (
#             "\n\n 13  14  15  16  17  18 | 19  20  21  22  23  24 \n"
#             "-------------------------------------------------\n"
#             + " ".join(top_row[:6]) + " | " + " ".join(top_row[6:])
#         )

#         # Middle section (bar)
#         bar = f"\n\nBar:\nWhite: {self.white_bar}, Black: {self.black_bar}\n"

#         # Bottom row (positions 11-0, reversed)
#         bottom_row = [format_position(self.positions[i]) for i in range(11, -1, -1)]
#         bottom = (
#             " 12  11  10   9   8   7 |  6   5   4   3   2   1 \n"
#             "-------------------------------------------------\n"
#             + " ".join(bottom_row[:6]) + " | " + " ".join(bottom_row[6:])
#         )

#         # Off-board area
#         off_board = f"\n\nOff-board:\nWhite: {self.white_off}, Black: {self.black_off}"
#         dice = f"\n\nDice: {self.dice}"
#         invalid_dice = f"\nInvalid Dice: {self.invalid_dice}"
#         start = "\n\nTurn: " + ("White" if self.turn == 1 else "Black") + "-----------------------------------------------\n"
#         end = "\n-----------------------------------------------\n"
#         left = f"\n\nWhite left: {self.white_left}, Black left: {self.black_left}"
#         # Combine all parts
#         return start + bar + bottom + top + off_board + dice + invalid_dice + left + end

#     def get_invalid_dice(self) -> list[int]:
#         self.verbose and print("Board:get_invalid_dice")
#         invalid_dice = self.dice[:]
#         max_length = 0
#         max_die = 0
        
#         # returns whether there is a valid move using all dice
#         # if there is one dice i can make some optimisations similar to the other function
#         def verify_permutation(board: Board, remaining_dice: list[int], move_sequence: list[int]) -> bool:
#             move_sequence = move_sequence[:]
#             nonlocal max_length
#             nonlocal max_die
#             nonlocal invalid_dice
#             # base case
#             # all dice are useable
#             if not remaining_dice:
#                 invalid_dice = []
#                 return True            
#             # there are dice remaining
            
#             # if you can move more dice than current best, replace
#             if len(move_sequence) > max_length:
#                 max_length = len(move_sequence)
#                 invalid_dice = remaining_dice

#             # if you can move the same number of dice but the max die is greater, replace
#             if len(move_sequence) == max_length and move_sequence and max(move_sequence) > max_die:
#                 max_die = max(move_sequence)
#                 invalid_dice = remaining_dice
            
#             # reentering moves
#             if board.turn == 1 and board.white_bar:
#                 for i in range(6):
#                     board_copy = deepcopy(board)
#                     if board_copy.move(-1, i):
#                         if verify_permutation(board_copy, board_copy.dice, move_sequence + list_diff(board.dice, board_copy.dice)):
#                             return True
#                 return False
#             if board.turn == -1 and board.black_bar:
#                 for i in range(23, 17, -1):
#                     board_copy = deepcopy(board)
#                     if board_copy.move(-1, i):
#                         if verify_permutation(board_copy, board_copy.dice, move_sequence + list_diff(board.dice, board_copy.dice)):
#                             return True
#                 return False
            
#             # bearing off
#             if board.can_bearoff():
#                 for i in range(24):
#                     board_copy = deepcopy(board)
#                     if board_copy.move(i, 100 * board.turn):
#                         if verify_permutation(board_copy, remaining_dice[1:], move_sequence + [remaining_dice[0]]):
#                             return True
                
#             # normal moves
#             for start in range(24):
#                 if sign(board.positions[start]) != self.turn:
#                     continue
#                 end = start + remaining_dice[0] * self.turn
#                 if board.is_valid(start, end):
#                     board_copy = deepcopy(board)
#                     if board_copy.move(start, end):
#                         if verify_permutation(board_copy, remaining_dice[1:], move_sequence + [remaining_dice[0]]):
#                             return True  
#             return False       
        
#         if verify_permutation(deepcopy(self), self.dice, []):
#             return []

#         for die in invalid_dice:
#             self.dice.remove(die)
#         return invalid_dice
    
#     def get_single_moves(self) -> set[tuple[int, int]]:
#         moves = set()
#         if self.turn == 1:
#             # reentering checkers
#             if self.white_bar:
#                 for dice in self.dice:
#                     if self.is_valid(-1, dice - 1):
#                         moves.add((-1, dice - 1))
#                 return moves
#             # bearing off
#             for start in range(18, 24):
#                 if self.is_valid(start, 100):
#                     moves.add((start, 100))

#             # normal moves
#             for start in range(24):
#                 for dice in self.dice:
#                     if self.is_valid(start, start + dice):
#                         moves.add((start, start + dice))

#             return moves

#         # reentering checkers
#         if self.black_bar:
#             for i in range(23, 17, -1):
#                 if self.is_valid(-1, i):
#                     moves.add((-1, i))
#             return moves

#         # bearing off
#         for start in range(5, -1, -1):
#             if self.is_valid(start, -100):
#                 moves.add((start, -100))

#         # normal moves
#         for start in range(23, -1, -1):
#             for dice in self.dice:
#                 if self.is_valid(start, start - dice):
#                     moves.add((start, start - dice))
#         return moves
        
#     def set_valid_moves(self) -> None:
#         self.verbose and print("Board:get_valid_moves")

#         def dfs(board: Board, prev_moves) -> list[list[tuple[int, int]]]:
#             if not board.dice:
#                 if prev_moves:
#                     return [prev_moves]
#                 return []
#             moves = []
#             if len(board.dice) == 1:
#                 for move in board.get_single_moves():
#                     if board.is_valid(*move):
#                         moves.append(prev_moves + [move])
#                 return moves
#             for move in board.get_single_moves():
#                 board_copy = deepcopy(board)
#                 board_copy.move(*move, bypass=True)
#                 moves += dfs(board_copy, prev_moves + [move])
#             return moves
        
#         self.valid_moves = dfs(deepcopy(self), [])
    
#     def can_bearoff(self) -> bool:
#         if self.turn == 1:
#             if self.white_bar:
#                 return False
#             for i in range(18):
#                 if self.positions[i] > 0:
#                     return False
#         else:
#             if self.black_bar:
#                 return False
#             for i in range(6, 24):
#                 if self.positions[i] < 0:
#                     return False
#         return True
        
#     def convert(self) -> dict:
#         ret = {
#             "positions": self.positions, 
#             "turn": self.turn, 
#             "dice": "".join(str(d) for d in self.dice),
#             "invalid_dice": "".join(str(d) for d in self.invalid_dice),
#             "white_bar": self.white_bar,
#             "black_bar": self.black_bar,
#             "rolled": self.rolled,
#             "white_off": self.white_off,
#             "black_off": self.black_off,
#             "valid_moves": self.valid_moves,
#             "game_over": self.game_over
#         }        
#         return ret
    
#     def has_passed(self) -> bool:
#         if self.white_bar > 0 or self.black_bar > 0:
#             return False
        
#         lowest_white = 24
#         highest_black = -1
        
#         # Single pass through positions for efficiency
#         for i, pos in enumerate(self.positions):
#             if pos > 0:
#                 lowest_white = i
#                 break
                
#         for i, pos in enumerate(self.positions[::-1]):
#             if pos < 0:
#                 highest_black = 23 - i
#                 break
        
#         # If all pieces are borne off for one color, it's considered passed
#         if lowest_white == -1 or highest_black == 24:
#             return True
        
#         # Check if all black pieces are past all white pieces
#         return lowest_white > highest_black
    
#     def calc_white_left(self) -> int:
#         total = 0
#         for i, pos in enumerate(self.positions):
#             if pos > 0:
#                 total += (24 - i) * pos
#         total += self.white_bar * 24
#         return total
            
    
#     def calc_black_left(self) -> int:
#         total = 0
#         for i, pos in enumerate(self.positions):
#             if pos < 0:
#                 total += (i + 1) * pos * -1
#         total += self.black_bar * 24
#         return total
    
#     def move_from_sequence(self, sequence: list[tuple[int, int]]) -> bool | None:
#         self.verbose and print("Board:move_from_sequence")
#         # TODO ive had errors removing this before
#         # seems to work fine now, no idea why
#         # sequence = [tuple(move) for move in sequence] 
#         if not self.valid_moves and not sequence:
#             self.swap_turn()
#         if sequence not in self.valid_moves:
#             return False
#         for move in sequence:
#             self.move(*move, bypass=True)
#         if not self.passed:
#             self.passed = self.has_passed()
#         self.white_left = self.calc_white_left()
#         self.black_left = self.calc_black_left()
#         if self.has_won():
#             self.game_over = True
#             return True
#         if self.rolled and len(self.dice) == 0:
#             self.swap_turn()
    
#     def has_won(self) -> bool:
#         return self.white_off == 15 or self.black_off == 15
    
#     def move(self, current: int, next: int, bypass: bool = False) -> bool:
#         if not bypass and not self.is_valid(current, next):
#             return False
        
#         # bearing off
#         if next == 100 or next == -100:
#             if self.turn == 1:
#                 self.white_off += 1
#                 if (24 - current) in self.dice:
#                     self.dice.remove(24 - current)
#                 else:
#                     self.dice.remove(max(self.dice))
#                 self.positions[current] -= 1
#             else:
#                 self.black_off += 1
#                 if (current + 1) in self.dice:
#                     self.dice.remove(current + 1)
#                 else:
#                     self.dice.remove(max(self.dice))
#                 self.positions[current] += 1
#             # if len(self.dice) == 0:
#             #     self.swap_turn() TODO im pretty sure i dont want this 
#             return True
        
#         # capturing a piece
#         if self.turn == 1:
#             if self.positions[next] == -1:
#                 self.positions[next] = 0
#                 self.black_bar += 1
#         else:
#             if self.positions[next] == 1:
#                 self.positions[next] = 0
#                 self.white_bar += 1
#         # reentering checkers
#         if current == -1:
#             if self.turn == 1:
#                 self.white_bar -= 1
#                 self.dice.remove(next+1)
#                 self.positions[next] += 1
#             else:
#                 self.black_bar -= 1
#                 self.dice.remove(24-next)    
#                 self.positions[next] -= 1
#         else: # not reentering
#             self.positions[current] -= self.turn
#             self.positions[next] += self.turn
#             self.dice.remove((next - current) * self.turn)
#         return True
    
#     def swap_turn(self) -> None:
#         self.verbose and print("Board:swap_turn")
#         self.turn = self.turn * -1
#         self.rolled = False
#         self.dice = []
#         self.invalid_dice = []
#         self.valid_moves = []

#     def is_valid(self, current: int, next: int) -> bool: 
#         # TODO: unit tests
#         # current can't be empty
#         if current < -1 or current > 23:
#             return False
#         if (next < 0 or next > 23) and abs(next) != 100:
#             return False
#         # bearing off 
#         if next == 100 and self.turn == 1 or next == -100 and self.turn == -1:
#             if not self.can_bearoff():
#                 return False
#             if sign(self.positions[current]) != self.turn:
#                 return False
#             if self.turn == 1:
#                 for dice in self.dice:
#                     if dice == 24 - current:
#                         return True
#                 # no exact dice
#                 if 24 - current > max(self.dice):
#                     return False
#                 for pos in range(18, 24):
#                     if self.positions[pos] > 0:
#                         if current == pos:
#                             return True
#                         else:
#                             return False
#                 # maybe i need to return something here, check later TODO
#             else: # self.turn == Color.BLACK
#                 for dice in self.dice:
#                     if dice == current + 1:
#                         return True
#                 if current + 1 > max(self.dice):
#                     return False
#                 for pos in range(5, -1, -1):
#                     if self.positions[pos] < 0:
#                         if current == pos:
#                             return True
#                         else:
#                             return False
#                 return False
#             assert False, "Should not reach here"

#         # reentering checkers
#         if self.white_bar and self.turn == 1:
#             if not (next+1) in self.dice:
#                 return False
#             if current != -1:
#                 return False
#             if next > 5:
#                 return False
#             if self.positions[next] < -1:
#                 return False
#             return True
        
#         if self.black_bar and self.turn == -1:
#             if current != -1:
#                 return False
#             if next < 18:
#                 return False
#             if not (24-next) in self.dice:
#                 return False
#             if self.positions[next] > 1:
#                 return False
#             return True
        
#         if sign(self.positions[current]) != self.turn:
#             return False
#         if self.positions[next] * self.turn < -1:
#             return False
        
#         for dice in self.dice:
#             if (next - current) * self.turn == dice:
#                 return True
#         return False
    
#     def roll_dice(self) -> tuple[list[int], list[int], list[list[tuple[int, int]]]] | bool:
#         self.verbose and print("Board:roll_dice")
#         if self.game_over:
#             return False
#         if self.rolled:
#             return self.dice, self.invalid_dice, self.valid_moves
#         self.dice = [randint(1, 6), randint(1, 6)]
#         if self.dice[0] == self.dice[1]:
#             self.dice.append(self.dice[0])
#             self.dice.append(self.dice[0])
#         self.rolled = True
#         self.invalid_dice = self.get_invalid_dice()
#         self.set_valid_moves()
#         return self.dice, self.invalid_dice, self.valid_moves
    
#     def set_dice(self, dice: list[int]) -> tuple[list[int], list[int], list[list[tuple[int, int]]]] | bool:
#         self.verbose and print("Board:roll_dice")
#         if self.game_over:
#             return False
#         if self.rolled:
#             return self.dice, self.invalid_dice, self.valid_moves
#         self.dice = dice
#         self.rolled = True
#         self.invalid_dice = self.get_invalid_dice()
#         self.set_valid_moves()
#         return self.dice, self.invalid_dice, self.valid_moves
    
#     def set_board(self, data: dict) -> dict:
#         if "positions" in data:
#             self.positions = data["positions"]
#         if "dice" in data:
#             self.dice = list(map(int, list(data["dice"])))
#         if "turn" in data:
#             self.turn = data["turn"]
#         return self.convert()
            

def extract_features(board):
    """    
    Args:
        board: A Board object representing the current game state
        
    Returns:
        List of numerical features representing the board state
    """
    features = []
    
    for pos in board.positions:
        features.append(pos / 15.0)
    
    features.append(board.white_bar / 15.0)
    features.append(board.black_bar / 15.0)
    

    features.append(board.white_off / 15.0)
    features.append(board.black_off / 15.0)
    
    white_singles = 0
    black_singles = 0
    for i in range(24):
        if board.positions[i] == 1:
            white_singles += 1
        elif board.positions[i] == -1:
            black_singles += 1
    features.append(white_singles / 15.0)
    features.append(black_singles / 15.0)
    
    features.append(longest_prime(board, 1) / 6.0)
    features.append(longest_prime(board, -1) / 6.0)
    
    white_anchor_points = 0
    for i in range(18, 24):
        if board.positions[i] >= 2:
            white_anchor_points += 1
    
    black_anchor_points = 0
    for i in range(0, 6):
        if board.positions[i] <= -2:
            black_anchor_points += 1
    
    features.append(white_anchor_points / 6.0)
    features.append(black_anchor_points / 6.0)

    
    return features

class BackgammonNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64]):
        """
        Neural network architecture for backgammon position evaluation.
        
        Args:
            input_size: Number of input features from the feature extractor
            hidden_sizes: List of hidden layer sizes
        """
        super(BackgammonNN, self).__init__()
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        
        # Output layer - single value representing winning probability
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())  # Bound output between 0 and 1
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)


class TDLambda:
    def __init__(self, model, learning_rate=0.01, lambda_param=0.7, gamma=0.99):
        """
        TD(λ) with explicit eligibility traces for updating a value function.
        
        Args:
            model: Neural network model.
            learning_rate: Step size for parameter updates.
            lambda_param: Decay rate for eligibility traces.
            gamma: Discount factor.
        """
        self.model = model
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.gamma = gamma
        # Using SGD because we update parameters manually with eligibility traces.
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        # Initialize eligibility traces for each parameter.
        self.eligibility = {}
        for name, param in self.model.named_parameters():
            self.eligibility[name] = torch.zeros_like(param, dtype=torch.float32)
    
    def update(self, states, reward):
        """
        Update model parameters using TD(λ) with eligibility traces.
        Assumes a single episode with final reward provided.
        
        Args:
            states: List of state representations (as tensors or convertible objects)
            reward: Final reward of the episode (e.g., 1 for win, 0 for loss)
        """
        if len(states) < 2:
            return
        
        # Convert states to tensors if necessary.
        state_tensors = [
            s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32)
            for s in states
        ]
        
        self.model.train()
        
        # Clear eligibility traces at the start of the episode.
        for name, param in self.model.named_parameters():
            self.eligibility[name].zero_()
        
        # Process each time step in the episode.
        for t in range(len(state_tensors) - 1):
            # Zero gradients for the current step.
            self.optimizer.zero_grad()
            
            # Compute the current state's value V(s_t).
            V_t = self.model(state_tensors[t])
            
            # Determine target value:
            # For all steps except the last, use V(s_{t+1}); at the final step, use the terminal reward.
            if t == len(state_tensors) - 2:
                # Last transition: use final reward as target.
                V_target = torch.tensor([[reward]], dtype=torch.float32)
            else:
                # Use the next state's value (no immediate reward assumed until terminal).
                with torch.no_grad():
                    V_target = self.model(state_tensors[t + 1])
            
            # Compute the TD error: δ = target - V(s_t)
            delta = V_target - V_t
            
            # Backpropagate to compute gradients of V(s_t) w.r.t. parameters.
            V_t.backward(retain_graph=True)
            
            # Update eligibility traces and parameters.
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        # Update trace: decay the old trace and add current gradient.
                        self.eligibility[name] = (
                            self.gamma * self.lambda_param * self.eligibility[name] + param.grad
                        )
                        # Update parameters using the TD error weighted by the eligibility trace.
                        param.add_(self.learning_rate * delta.item() * self.eligibility[name])
            
            # Zero gradients before next iteration.
            self.optimizer.zero_grad()


class NNAgent:
    def __init__(self, model, extract_features_fn, exploration_rate=0.1):
        """
        Neural network-based agent for backgammon.
        
        Args:
            model: Trained neural network model
            extract_features_fn: Function to extract features from a board
            exploration_rate: Rate of random exploration (0.0 to 1.0)
        """
        self.model = model
        self.extract_features = extract_features_fn
        self.exploration_rate = exploration_rate
        
    def select_move(self, board):
        """
        Select the best move based on model predictions.
        
        Args:
            board: Current board state
            
        Returns:
            Best move sequence
        """
        if not board.valid_moves:
            return []
            
        # Occasionally make a random move for exploration
        if random.random() < self.exploration_rate:
            return random.choice(board.valid_moves)
            
        best_move = None
        best_value = float('-inf') if board.turn == 1 else float('inf')
        
        # Evaluate each possible move
        for move in board.valid_moves:
            # Create a copy of the board and apply the move
            board_copy = deepcopy(board)
            board_copy.move_from_sequence(move)
            
            # Extract features from the resulting position
            features = self.extract_features(board_copy)
            state_tensor = torch.tensor([features], dtype=torch.float32)
            
            # Get model prediction
            with torch.no_grad():
                self.model.eval()
                value = self.model(state_tensor).item()
            
            # Select the best move based on the player's perspective
            if (board.turn == 1 and value > best_value) or (board.turn == -1 and value < best_value):
                best_value = value
                best_move = move
                
        return best_move


class BackgammonTrainer:
    def __init__(self, model, extract_features_fn, td_lambda, games_per_epoch=1000):
        """
        Training pipeline for backgammon AI.
        
        Args:
            model: Neural network model
            extract_features_fn: Function to extract features from a board
            td_lambda: TD(λ) learning implementation
            games_per_epoch: Number of games to play per training epoch
        """
        self.model = model
        self.extract_features = extract_features_fn
        self.td_lambda = td_lambda
        self.games_per_epoch = games_per_epoch
    
    def save_checkpoint(self, filename, epoch=0, optimizer_state=None):
        """
        Save a complete training checkpoint that can be used to resume training.
        
        Args:
            filename: Path to save the checkpoint
            epoch: Current epoch number
            optimizer_state: State of the optimizer (optional)
        """
        if optimizer_state is None and hasattr(self, 'td_lambda') and hasattr(self.td_lambda, 'optimizer'):
            optimizer_state = self.td_lambda.optimizer.state_dict()
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer_state,
            'win_rate': getattr(self, 'last_win_rate', 0.0),
            'training_results': getattr(self, 'training_results', [])
        }
        
        torch.save(checkpoint, filename)
        print(f"Model checkpoint saved to {filename}")
    
    def load_checkpoint(self, filename):
        """
        Load a training checkpoint to resume training.
        
        Args:
            filename: Path to the checkpoint file
            
        Returns:
            epoch: The epoch to resume from
            training_results: Previous training results
        """
        if not os.path.exists(filename):
            print(f"Checkpoint file {filename} not found. Starting fresh training.")
            return 0, []
        try:
            checkpoint = torch.load(filename)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state if it exists and we have an optimizer
            if 'optimizer_state_dict' in checkpoint and hasattr(self, 'td_lambda') and hasattr(self.td_lambda, 'optimizer'):
                self.td_lambda.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            # Store training results history
            self.training_results = checkpoint.get('training_results', [])
            self.last_win_rate = checkpoint.get('win_rate', 0.0)
            
            epoch = checkpoint.get('epoch', 0)
            print(f"Loaded checkpoint from epoch {epoch} with win rate {self.last_win_rate:.4f}")
            
            return epoch, self.training_results
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return 0, []
    
    def train_epoch(self):
        """Train the model for one epoch (multiple games)."""
        wins_white = 0
        
        for game_num in range(self.games_per_epoch):
            # Play a complete game
            winner, white_states, black_states = self.play_game()
            
            self.td_lambda.update(white_states, float(winner))
            self.td_lambda.update(black_states, float(winner))
            if winner == 1:  # White won
                wins_white += 1
                self.td_lambda.update(white_states, 1.0)
                self.td_lambda.update(black_states, 0.0)
            else:  # Black won
                self.td_lambda.update(white_states, 0.0)
                self.td_lambda.update(black_states, 1.0)
            
            if (game_num + 1) % 20 == 0:
                print(f"Completed game {game_num + 1}/{self.games_per_epoch}")
                
        win_rate = wins_white / self.games_per_epoch
        return win_rate
    
    def play_game(self):
        """
        Play a complete game of backgammon using the current model.
        
        Returns:
            winner: The winner of the game (1 for white, -1 for black)
            white_states: List of white player state tensors
            black_states: List of black player state tensors
        """
        board = Board()
        agent = NNAgent(self.model, self.extract_features)
        
        white_states = []
        black_states = []
        
        move_count = 0
        max_moves = 200  # Prevent infinite games
        
        while not board.game_over and move_count < max_moves:
            # Roll dice
            board.roll_dice()         
    
            # Extract features from current state
            features = self.extract_features(board)
            state_tensor = torch.tensor([features], dtype=torch.float32)
            
            # Store state
            if board.turn == 1:
                white_states.append(state_tensor)
            else:
                black_states.append(state_tensor)
            
            # Choose move
            move = agent.select_move(board)
            
            # Make move
            board.move_from_sequence(move)
            move_count += 1
        
        # Force an end if max moves reached
        if move_count >= max_moves:
            # The player with more pieces borne off wins
            winner = 1 if board.white_off > board.black_off else -1
        else:
            winner = 1 if board.white_off == 15 else -1
        
        if not white_states:
            white_states.append(torch.zeros([1, self.extract_features(board).shape[0]], dtype=torch.float32))
        if not black_states:
            black_states.append(torch.zeros([1, self.extract_features(board).shape[0]], dtype=torch.float32))
            
        return winner, white_states, black_states
    
    def train(self, num_epochs=100, resume_from=None, checkpoint_interval=5, name=""):
        """
        Train the model for multiple epochs with support for resuming training.
        
        Args:
            num_epochs: Total number of epochs to train
            resume_from: Path to checkpoint file to resume from (optional)
            checkpoint_interval: Save checkpoint every N epochs
            
        Returns:
            List of win rates for each epoch
        """
        start_epoch = 0
        results = []
        
        # Load checkpoint if provided
        if resume_from:
            start_epoch, prev_results = self.load_checkpoint(resume_from)
            results = prev_results
        
        # Training loop
        for epoch in range(start_epoch, start_epoch + num_epochs):
            print(f"\nEpoch {epoch+1}/{start_epoch + num_epochs}")
            win_rate = self.train_epoch()
            
            results.append(win_rate)
            self.last_win_rate = win_rate
            self.training_results = results
            
            print(f"Epoch {epoch+1}/{start_epoch + num_epochs}, Win Rate: {win_rate:.4f}")
            
            # Save checkpoint at regular intervals
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(f"backgammon_checkpoint_epoch_{name}{epoch+1}.pt", epoch=epoch+1)
            
            e = Evaluator(NNAgent(self.model, self.extract_features, exploration_rate=0.0), opponent_agent=RandomAgent(), num_games=100)
            eval_results = e.evaluate()                

            print("\nEvaluation Results:")
            for key, value in eval_results.items():
                print(f"{key}: {value}")
            with open(f"eval_results{name}.txt", "a") as f:
                f.write(f"{epoch}\t{eval_results['white_win_rate']}\t{eval_results['black_win_rate']}\t{eval_results['win_rate']}\n")
            
            # Always save latest model
            self.plot_learning_curve(save_path=f"learning_curve{name}.png")
            self.save_checkpoint(f"backgammon_{name}latest.pt", epoch=epoch+1)
        
        self.plot_eval_curve(f"eval_curve{name}.png")
        
        # Save final model
        self.save_model(f"backgammon_final_model{name}.pt")
        
        return results

    def plot_eval_curve(self, save_path=None):
        with open(save_path, "r") as f:
            lines = f.readlines()
            epochs = [int(line.split("\t")[0]) for line in lines]
            white_win_rates = [float(line.split("\t")[1]) for line in lines]
            black_win_rates = [float(line.split("\t")[2]) for line in lines]
            win_rates = [float(line.split("\t")[3]) for line in lines]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, white_win_rates, 'b-', label='White Win Rate')
        plt.plot(epochs, black_win_rates, 'r-', label='Black Win Rate')
        plt.plot(epochs, win_rates, 'g-', label='Total Win Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Win Rate')
        plt.title('Evaluation Progress / Win Rate against Random Agent')
        plt.grid(True)
        plt.legend()
        
        # Add trend lines
        x = np.arange(len(epochs))
        
        # Trend line for White Win Rate
        z_white = np.polyfit(x, white_win_rates, 1)
        p_white = np.poly1d(z_white)
        plt.plot(epochs, p_white(x), "b--", alpha=0.5, label='White Trend')
        
        # Trend line for Black Win Rate
        z_black = np.polyfit(x, black_win_rates, 1)
        p_black = np.poly1d(z_black)
        plt.plot(epochs, p_black(x), "r--", alpha=0.5, label='Black Trend')
        
        # Trend line for Total Win Rate
        z_total = np.polyfit(x, win_rates, 1)
        p_total = np.poly1d(z_total)
        plt.plot(epochs, p_total(x), "g--", alpha=0.5, label='Total Trend')
        
        plt.xlabel('Epoch')
        plt.ylabel('Win Rate')
        plt.title('Evaluation Progress')
        plt.grid(True)
        plt.legend()
        
        
        if save_path:
            plt.savefig(save_path)
            print(f"Evaluation curve saved to {save_path}")
    
    def plot_learning_curve(self, save_path=None):
        """
        Plot the learning curve showing win rate progression over epochs.
        
        Args:
            save_path: Path to save the plot image (optional)
        """        
        if not hasattr(self, 'training_results') or not self.training_results:
            print("No training results available to plot.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_results, 'b-')
        plt.plot(self.training_results, 'ro')
        plt.xlabel('Epoch')
        plt.ylabel('Win Rate as White')
        plt.title('Training Progress / Win Rate against itself')
        plt.grid(True)
        
        # Add trend line
        if len(self.training_results) > 1:
            import numpy as np
            x = np.arange(len(self.training_results))
            z = np.polyfit(x, self.training_results, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), "r--", alpha=0.5)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Learning curve saved to {save_path}")
            
    
    def evaluate(self):
        """Evaluate the model against a random agent."""
        nn_agent = NNAgent(self.model, self.extract_features, exploration_rate=0.0)
    
        # Evaluate against random agent
        print("Evaluating against random agent...")
        evaluator = Evaluator(nn_agent, opponent_agent=RandomAgent(), num_games=50)
        eval_results = evaluator.evaluate()
        
        # Print evaluation results
        print("\nEvaluation Results:")
        print("-" * 50)
        for key, value in eval_results.items():
            print(f"{key}: {value}")
        
    def save_model(self, filename):
        """Save the model to a file."""
        torch.save(self.model.state_dict(), filename)
        
    def load_model(self, filename):
        """Load model from a file."""
        try:
            model = torch.load(filename)
            return model
        except:
            model = BackgammonNN(input_size=198)
            model.load_state_dict(torch.load(filename))
            return model

class Evaluator:
    def __init__(self, nn_agent, opponent_agent=None, num_games=100, name=""):
        """
        Evaluation framework for backgammon AI.
        
        Args:
            nn_agent: Neural network agent to evaluate
            opponent_agent: Agent to play against (default: RandomAgent)
            num_games: Number of games to play for evaluation
        """
        self.nn_agent = nn_agent
        self.opponent_agent = opponent_agent if opponent_agent else RandomAgent()
        self.num_games = num_games
        self.move_limit = 200
        
    def evaluate(self):
        """
        Evaluate the neural network agent against the opponent.
        
        Returns:
            Dictionary of evaluation metrics
        """
        nn_as_white_wins = 0
        nn_as_black_wins = 0
        white_turn_counts = []
        black_turn_counts = []
        
        # Play games with NN as white
        for i in range(self.num_games // 2):
            board = Board()
            turn_count = 0
            
            while not board.game_over and turn_count < self.move_limit:
                board.roll_dice()
                
                if board.turn == 1:
                    move = self.nn_agent.select_move(board)
                else:
                    move = self.opponent_agent.select_move(board)
                
                board.move_from_sequence(move)
                turn_count += 1
            
            white_turn_counts.append(turn_count)
            
            if board.white_off == 15 or (turn_count >= self.move_limit and board.white_off > board.black_off):
                nn_as_white_wins += 1
                
        print("Completed games as white")
                
        # Play games with NN as black
        for i in range(self.num_games // 2):
            board = Board()
            turn_count = 0
            
            while not board.game_over and turn_count < 500:  # Limit to prevent infinite games
                board.roll_dice()
                
                if board.turn == 1:
                    move = self.opponent_agent.select_move(board)
                else:
                    move = self.nn_agent.select_move(board)
                
                board.move_from_sequence(move)
                turn_count += 1
            
            black_turn_counts.append(turn_count)
            
            if board.black_off == 15 or (turn_count >= 500 and board.black_off > board.white_off):
                nn_as_black_wins += 1
                
        print("Completed games as black")
                
        total_wins = nn_as_white_wins + nn_as_black_wins
        win_rate = total_wins / self.num_games
        
        return {
            "total_games": self.num_games,
            "total_wins": total_wins,
            "win_rate": win_rate,
            "white_wins": nn_as_white_wins,
            "black_wins": nn_as_black_wins,
            "white_win_rate": nn_as_white_wins / (self.num_games // 2),
            "black_win_rate": nn_as_black_wins / (self.num_games // 2),
            "avg_turns_as_white": sum(white_turn_counts) / len(white_turn_counts) if white_turn_counts else 0,
            "avg_turns_as_black": sum(black_turn_counts) / len(black_turn_counts) if black_turn_counts else 0
        }


class ModelComparator:
    def __init__(self, models, extract_features_fn, num_games=50):
        """
        Compare multiple trained models against each other.
        
        Args:
            models: Dictionary of {model_name: model} to compare
            extract_features_fn: Function to extract features from a board
            num_games: Number of games per match-up
        """
        self.models = models
        self.extract_features = extract_features_fn
        self.num_games = num_games
        self.results = {}
        
    def run_tournament(self):
        """Run a round-robin tournament between all models."""
        model_names = list(self.models.keys())
        results = {name: {"wins": 0, "games": 0} for name in model_names}
        
        # For each pair of models, play games
        for i, model1_name in enumerate(model_names):
            for model2_name in model_names[i+1:]:
                print(f"Match: {model1_name} vs {model2_name}")
                
                model1 = self.models[model1_name]
                model2 = self.models[model2_name]
                
                agent1 = NNAgent(model1, self.extract_features, exploration_rate=0)
                agent2 = NNAgent(model2, self.extract_features, exploration_rate=0)
                
                # Each model plays as white and black
                for _ in range(self.num_games // 2):
                    # Model1 as white
                    winner = self._play_game(agent1, agent2)
                    if winner == 1:
                        results[model1_name]["wins"] += 1
                    else:
                        results[model2_name]["wins"] += 1
                    results[model1_name]["games"] += 1
                    results[model2_name]["games"] += 1
                    
                    # Model2 as white
                    winner = self._play_game(agent2, agent1)
                    if winner == 1:
                        results[model2_name]["wins"] += 1
                    else:
                        results[model1_name]["wins"] += 1
                    results[model1_name]["games"] += 1
                    results[model2_name]["games"] += 1
        
        # Calculate win rates
        for name in results:
            results[name]["win_rate"] = results[name]["wins"] / results[name]["games"] if results[name]["games"] > 0 else 0
            
        self.results = results
        return results
    
    def _play_game(self, white_agent, black_agent):
        """Play a single game between two agents."""
        board = Board()
        turn_count = 0
        max_turns = 500  # Prevent infinite games
        
        while not board.game_over and turn_count < max_turns:
            board.roll_dice()
            
            if board.turn == 1:
                move = white_agent.select_move(board)
            else:
                move = black_agent.select_move(board)
            
            board.move_from_sequence(move)
            turn_count += 1
            
        if turn_count >= max_turns:
            return 1 if board.white_off > board.black_off else -1
        return 1 if board.white_off == 15 else -1
    
    def print_results(self):
        """Print tournament results in a readable format."""
        if not self.results:
            print("No tournament results available. Run run_tournament() first.")
            return
        
        print("\nTournament Results:")
        print("-" * 50)
        print(f"{'Model':<20} {'Wins':<10} {'Games':<10} {'Win Rate':<10}")
        print("-" * 50)
        
        # Sort by win rate
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]["win_rate"], reverse=True)
        
        for name, stats in sorted_results:
            print(f"{name:<20} {stats['wins']:<10} {stats['games']:<10} {stats['win_rate']:.4f}")

def longest_prime(board, color):
        longest = 0
        current = 0
        if color == 1:
            for i in range(24):
                if board.positions[i] >= 2:
                    current += 1
                    longest = max(longest, current)
                else:
                    current = 0
        else:
            for i in range(24):
                if board.positions[i] <= -2:
                    current += 1
                    longest = max(longest, current)
                else:
                    current = 0
        return longest

class FinalNNAgent:
    def __init__(self, model_path, extract_features_fn=extract_features):
        self.extract_features = extract_features_fn
        self.model = BackgammonNN(35)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def select_move(self, board):
        if not board.valid_moves:
            return []
        
        best_move = None
        best_value = float('-inf') if board.turn == 1 else float('inf')
        
        for move in board.valid_moves:
            board_copy = deepcopy(board)
            board_copy.move_from_sequence(move)
            
            features = self.extract_features(board_copy)
            state_tensor = torch.tensor([features], dtype=torch.float32)
            
            with torch.no_grad():
                value = self.model(state_tensor).item()
            
            if (board.turn == 1 and value > best_value) or (board.turn == -1 and value < best_value):
                best_value = value
                best_move = move
        
        return best_move
        


def main(resume=False, epochs=20, testing=False):
    torch.autograd.set_detect_anomaly(True)
    """Main function to train and evaluate the backgammon AI."""
    # Assuming the extract_features function already exists
    
    # Define input size based on feature extractor output
    input_size = len(extract_features(Board()))
    
    print(f"Feature vector size: {input_size}")
    
    # Create neural network model
    model = BackgammonNN(input_size=input_size)
    
    # Initialize TD-Lambda learner
    td_lambda = TDLambda(model, learning_rate=0.05, lambda_param=0.7)
    
    # Create trainer
    trainer = BackgammonTrainer(
        model=model,
        extract_features_fn=extract_features,
        td_lambda=td_lambda,
        games_per_epoch=100
    )
    
    print("Starting training")
    # Resume or start new training
    
    
    # Optional: Create multiple models with different parameters for comparison
    models = {
        "main_model": model,
    }
    
    # Train a smaller model for comparison
    small_model = BackgammonNN(input_size=input_size, hidden_sizes=[64, 64])
    small_td = TDLambda(small_model, learning_rate=0.01, lambda_param=0.7)
    small_trainer = BackgammonTrainer(small_model, extract_features, small_td, games_per_epoch=100)
    
    # Train a model with different lambda
    lambda_model = BackgammonNN(input_size=input_size)
    lambda_td = TDLambda(lambda_model, learning_rate=0.01, lambda_param=0.4)
    lambda_trainer = BackgammonTrainer(lambda_model, extract_features, lambda_td, games_per_epoch=100)
    
    low_lambda_model = BackgammonNN(input_size=input_size)
    low_lambda_td = TDLambda(low_lambda_model, learning_rate=0.01, lambda_param=0.1)
    low_lambda_trainer = BackgammonTrainer(low_lambda_model, extract_features, low_lambda_td, games_per_epoch=100)
    

    # Train a model with a higher learning rate
    high_lr_model = BackgammonNN(input_size=input_size)
    high_lr_td = TDLambda(high_lr_model, learning_rate=0.1, lambda_param=0.7)
    high_lr_trainer = BackgammonTrainer(high_lr_model, extract_features, high_lr_td, games_per_epoch=100)
    
    
    # Train a model with hidden sizes [128, 128], learning rate 0.05, and lambda 0.7
    hidden_128_model = BackgammonNN(input_size=input_size, hidden_sizes=[128, 128])
    hidden_128_td = TDLambda(hidden_128_model, learning_rate=0.05, lambda_param=0.7)
    hidden_128_trainer = BackgammonTrainer(hidden_128_model, extract_features, hidden_128_td, games_per_epoch=100)
    
    
    # Define a function to train a model
    def train_model(trainer, name, epochs):
        trainer.train(num_epochs=epochs, name=name)
        return trainer.model, name
    
    # Create a multiprocessing pool
    pool = mp.Pool(processes=mp.cpu_count())  # Use all available CPUs
    
    # Train models in parallel
    results = [
        pool.apply_async(train_model, args=(low_lambda_trainer, "low_lambda", 20)),
        pool.apply_async(train_model, args=(small_trainer, "small", 20)),
        pool.apply_async(train_model, args=(lambda_trainer, "lambda_0.4", 20)),
        pool.apply_async(train_model, args=(high_lr_trainer, "high_lr", 20)),
        pool.apply_async(train_model, args=(hidden_128_trainer, "hidden_128", 20))
    ]
    
    
    
    if resume:
        # Load the latest checkpoint
        results.append(pool.apply_async(train_model, args=(trainer, "main", 20)))
        # results = trainer.train(num_epochs=epochs, resume_from="backgammon_latest.pt")
    else:
        # Start fresh training
        print("Starting fresh training...")
        # results = trainer.train(num_epochs=epochs)
        results.append(pool.apply_async(train_model, args=(trainer, "main", 20)))
    
    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()
    print("after pool")

    trainer.plot_learning_curve(save_path="learning_curve.png")
    trainer.save_model("backgammon_final_model.pt")    
    nn_agent = NNAgent(model, extract_features, exploration_rate=0.0) # Create neural network agent with the trained model
    # Evaluate against random agent
    print("Evaluating against random agent...")
    evaluator = Evaluator(nn_agent, opponent_agent=RandomAgent(), num_games=200, name="final")
    eval_results = evaluator.evaluate()
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print("-" * 50)
    for key, value in eval_results.items():
        print(f"{key}: {value}")
    
    # Collect the trained models
    for result in results:
        model, name = result.get()
        models[f"{name}_model"] = model
    
    # Run tournament between models
    print("\nRunning model tournament...")
    comparator = ModelComparator(models, extract_features, num_games=100)
    comparator.run_tournament()
    comparator.print_results()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Backgammon AI')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--evaluate', action='store_true', help='Only evaluate the model without training')
    parser.add_argument('--model', type=str, default='backgammon_final_model.pt', help='Model path for evaluation')
    parser.add_argument('--testing', action='store_true', help='Run in testing mode')
    
    args = parser.parse_args()
    
    if args.evaluate:
        # Only run evaluation on a saved model
        input_size = len(extract_features(Board()))
        model = BackgammonNN(input_size=input_size)
        model.load_state_dict(torch.load(args.model))
        nn_agent = NNAgent(model, extract_features, exploration_rate=0.0)
        
        evaluator = Evaluator(nn_agent, opponent_agent=RandomAgent(), num_games=200)
        eval_results = evaluator.evaluate()
        
        print("\nEvaluation Results:")
        print("-" * 50)
        for key, value in eval_results.items():
            print(f"{key}: {value}")
    else:
        # Run training
        main(resume=args.resume, epochs=args.epochs, testing=args.testing)

