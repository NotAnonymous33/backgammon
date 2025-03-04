from copy import deepcopy
import random
import math
try:
    from Board import Board
    from Color import Color
except:
    from ..Board import Board
    from ..Color import Color


class HeuristicBackgammonAgent:
    def __init__(self, weights=None):
        if weights:
            self.off_weight = weights[0]
            self.race_weight = weights[1]
            self.home_weight = weights[2]
            self.inner_weight = weights[3]
            self.outer_weight = weights[4]
            self.single_checker_weight_home = weights[5]
            self.single_checker_weight_outer = weights[6]
            self.prime_weight = weights[7]
            self.anchor_weight = weights[8]
            self.bar_weight = weights[9]
        else:
            self.off_weight = 10
            self.race_weight = 0.5
            self.home_weight = 5
            self.inner_weight = 3
            self.outer_weight = 2
            self.single_checker_weight_home = 3
            self.single_checker_weight_outer = 1
            self.prime_weight = 5
            self.anchor_weight = 8
            self.bar_weight = 15
        
    def evaluate_board(self, board: Board):
        score = 0
        
        # 1. Material advantage (pieces off the board)
        score += self.off_weight * board.white_off
        score -= self.off_weight * board.black_off
        
        # 2. Race advantage (pip count differential)
        score += (board.white_left - board.black_left) * self.race_weight
        
        # 3. Checker distribution and point control
        for i in range(24):
            # Points made by white (2+ checkers)
            if board.positions[i] >= 2:
                # Value of made points depends on position
                # Home board points are most valuable
                if 0 <= i <= 5:
                    score += self.home_weight * min(board.positions[i], 3)
                # Next most valuable is blocking prime (consecutive points)
                elif 6 <= i <= 11:
                    score += self.inner_weight * min(board.positions[i], 3)
                # Outer board points
                else:
                    score += self.outer_weight * min(board.positions[i], 3)
                    
            # Points made by black
            elif board.positions[i] <= -2:
                # Same logic but for Black, subtract from score
                if 18 <= i <= 23:
                    score -= self.home_weight * min(abs(board.positions[i]), 3)
                elif 12 <= i <= 17:
                    score -= self.inner_weight * min(abs(board.positions[i]), 3)
                else:
                    score -= self.outer_weight * min(abs(board.positions[i]), 3)
                    
            # Blots (single checkers)
            elif board.positions[i] == 1:
                # Penalize more heavily for blots in opponent's home board
                if 18 <= i <= 23:
                    score -= self.single_checker_weight_home
                else:
                    score -= self.single_checker_weight_outer
            elif board.positions[i] == -1:
                if 0 <= i <= 5:
                    score += self.single_checker_weight_home
                else:
                    score += self.single_checker_weight_outer
        
        # 4. Primes (consecutive points)
        white_prime_length = self.longest_prime(board, Color.WHITE)
        black_prime_length = self.longest_prime(board, Color.BLACK)
        score += white_prime_length * self.prime_weight
        score -= black_prime_length * self.prime_weight
        
        # 5. Anchors in opponent's home board
        white_anchor_points = 0
        for i in range(18, 24):
            if board.positions[i] >= 2:
                white_anchor_points += 1
        
        black_anchor_points = 0
        for i in range(0, 6):
            if board.positions[i] <= -2:
                black_anchor_points += 1
        
        score += white_anchor_points * self.anchor_weight
        score -= black_anchor_points * self.anchor_weight
        
        # 6. Checkers on bar
        score -= self.bar_weight * board.white_bar  # Severe penalty for checkers on bar
        score += self.bar_weight * board.black_bar
        
        return score
    
    def longest_prime(self, board, color):
        longest = 0
        current = 0
        if color == Color.WHITE:
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
        
    def select_move(self, board: Board):
        """Select the best move based on heuristic evaluation."""
        # Return empty list if no valid moves
        if not board.valid_moves:
            return []
            
        best_move = None
        if board.turn == Color.WHITE:
            best_score = -1000000
        else:
            best_score = 1000000

        for move in board.valid_moves:
            # Clone the board and apply the move
            new_board = deepcopy(board)
            new_board.move_from_sequence(move)
            
            # Evaluate the new board
            score = self.evaluate_board(new_board)
            
            # Update best move if necessary
            if board.turn == Color.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
        
        return best_move
            
    
    