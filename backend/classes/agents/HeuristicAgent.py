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
    def __init__(self):
        # Strategy weights that adapt based on game state
        self.weights = {
            'safety': 0.4,
            'structure': 0.3,
            'race': 0.2,
            'special': 0.1
        }
        
        # Opening book moves for common dice rolls
        self.opening_book = {
            (1, 2): [((0, 2), (0, 1))],            # 2-1: 24/23, 24/22
            (1, 3): [((0, 3), (0, 1))],            # 3-1: 24/21, 24/23
            (1, 4): [((0, 4), (0, 1))],            # 4-1: 24/20, 24/23
            (1, 5): [((11, 16), (5, 6))],          # 5-1: 13/8, 6/5
            (1, 6): [((11, 17), (5, 6))],          # 6-1: 13/7, 6/5
            (2, 3): [((0, 3), (0, 2))],            # 3-2: 24/21, 24/22
            (2, 4): [((0, 4), (0, 2))],            # 4-2: 24/20, 24/22
            (2, 5): [((11, 16), (5, 7))],          # 5-2: 13/8, 6/4
            (2, 6): [((11, 17), (5, 7))],          # 6-2: 13/7, 6/4
            (3, 4): [((0, 4), (0, 3))],            # 4-3: 24/20, 24/21
            (3, 5): [((5, 10), (5, 8))],           # 5-3: 6/1, 6/3
            (3, 6): [((11, 17), (5, 8))],          # 6-3: 13/7, 6/3
            (4, 5): [((11, 16), (5, 10))],         # 5-4: 13/8, 6/1
            (4, 6): [((11, 17), (5, 9))],          # 6-4: 13/7, 6/2
            (5, 6): [((11, 17), (11, 16))],        # 6-5: 13/7, 13/8
            (1, 1, 1, 1): [((5, 6), (5, 6), (11, 12), (11, 12))],  # 1-1: 6/5, 6/5, 13/12, 13/12
            (2, 2, 2, 2): [((11, 13), (11, 13), (5, 7), (5, 7))],  # 2-2: 13/11, 13/11, 6/4, 6/4
            (3, 3, 3, 3): [((11, 14), (11, 14), (5, 8), (5, 8))],  # 3-3: 13/10, 13/10, 6/3, 6/3
            (4, 4, 4, 4): [((0, 4), (0, 4), (11, 15), (11, 15))],  # 4-4: 24/20, 24/20, 13/9, 13/9
            (5, 5, 5, 5): [((0, 5), (0, 5), (11, 16), (11, 16))],  # 5-5: 24/19, 24/19, 13/8, 13/8
            (6, 6, 6, 6): [((0, 6), (0, 6), (11, 17), (11, 17))]   # 6-6: 24/18, 24/18, 13/7, 13/7
        }
        
        # Point importance values (higher is more important)
        self.point_importance = {
            0: 4,   # 24-point: Opponent's ace-point
            1: 3,   # 23-point
            2: 3,   # 22-point
            3: 2,   # 21-point
            4: 2,   # 20-point
            5: 4,   # 19-point: Opponent's 5-point (key anchor)
            6: 4,   # 18-point: Opponent's bar-point
            7: 3,   # 17-point
            8: 2,   # 16-point
            9: 2,   # 15-point
            10: 2,  # 14-point
            11: 4,  # 13-point
            12: 2,  # 12-point
            13: 2,  # 11-point
            14: 2,  # 10-point
            15: 3,  # 9-point
            16: 3,  # 8-point
            17: 4,  # 7-point (bar-point)
            18: 4,  # 6-point (key defensive point)
            19: 2,  # 5-point
            20: 2,  # 4-point
            21: 3,  # 3-point
            22: 3,  # 2-point
            23: 4,  # 1-point (ace-point)
        }
        
    def select_move(self, board):
        """Select the best move based on heuristic evaluation."""
        # Return empty list if no valid moves
        if not board.valid_moves:
            return []
            
        # If only one valid move, return it immediately
        if len(board.valid_moves) == 1:
            return board.valid_moves[0]
            
        # Try opening book for early game
        opening_move = self.get_opening_book_move(board)
        if opening_move:
            return opening_move
        
        # Determine the current game phase and adjust weights
        self.adjust_weights_for_phase(board)
        
        # Evaluate all possible moves
        move_scores = []
        for move in board.valid_moves:
            score = self.evaluate_move(board, move)
            move_scores.append((move, score))
            
        # Sort by score (highest first)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select best move (with some randomization among close scores)
        return self.select_best_with_variation(move_scores)
    
    def get_opening_book_move(self, board):
        """Get a move from the opening book if available."""
        # Only use opening book on the first move of the game
        if board.white_off > 0 or board.black_off > 0 or board.white_bar > 0 or board.black_bar > 0:
            return None
            
        # Check if total pieces moved is 0 (first move)
        total_moved = 0
        for pos in board.positions:
            total_moved += abs(len(pos))  # Count pieces not in starting position
            
        if total_moved > 30:  # 15 pieces per player * 2 players = 30 total pieces
            return None
            
        # Try to find the current dice roll in the opening book
        dice_tuple = tuple(sorted(board.dice))
        
        if dice_tuple in self.opening_book:
            book_moves = self.opening_book[dice_tuple]
            
            # Convert book notation to actual board positions for the current player
            for book_move in book_moves:
                # Adjust move based on player color
                adjusted_move = []
                for move_pair in book_move:
                    if board.turn == Color.WHITE:
                        adjusted_move.append(move_pair)
                    else:
                        # Black moves in the opposite direction
                        adjusted_move.append((23 - move_pair[0], 23 - move_pair[1]))
                
                # Check if adjusted move is valid
                for valid_move in board.valid_moves:
                    if self.moves_equivalent(adjusted_move, valid_move):
                        return valid_move
        
        return None
        
    def moves_equivalent(self, move1, move2):
        """Check if two moves are equivalent (same positions, possibly different order)."""
        if len(move1) != len(move2):
            return False
        
        # Convert moves to sets of (from, to) tuples for comparison
        set1 = set(tuple(m) for m in move1)
        set2 = set(tuple(m) for m in move2)
        
        return set1 == set2
        
    def select_best_with_variation(self, move_scores):
        """Select the best move with some variation to avoid predictability."""
        if not move_scores:
            return []
            
        # Get the highest score
        best_score = move_scores[0][1]
        
        # Consider all moves within 5% of the best score as candidates
        candidates = []
        threshold = best_score * 0.95
        
        for move, score in move_scores:
            if score >= threshold:
                candidates.append(move)
                
        # If we have multiple candidates, occasionally choose randomly among them
        if len(candidates) > 1 and random.random() < 0.2:  # 20% chance of variation
            return random.choice(candidates)
        else:
            return move_scores[0][0]  # Return the highest-scoring move
    
    def evaluate_move(self, board, move):
        """Evaluate a potential move using multiple heuristics."""
        # Create a copy of the board and apply the move
        temp_board = deepcopy(board)
        temp_board.move_from_sequence(move)
        
        # Calculate component scores
        safety_score = self.evaluate_safety(temp_board)
        structure_score = self.evaluate_structure(temp_board)
        race_score = self.evaluate_race(temp_board)
        special_score = self.evaluate_special_situations(temp_board)
        
        # Combine scores with appropriate weights
        final_score = (
            self.weights['safety'] * safety_score +
            self.weights['structure'] * structure_score +
            self.weights['race'] * race_score +
            self.weights['special'] * special_score
        )
        
        return final_score
    
    def evaluate_safety(self, board):
        """Evaluate checker safety."""
        score = 0
        my_color = board.turn
        opp_color = Color.BLACK if my_color == Color.WHITE else Color.WHITE
        
        # Penalize vulnerable blots
        for pos in range(24):
            if self.is_blot(board, pos, my_color):
                vulnerability = self.calculate_vulnerability(board, pos)
                score -= vulnerability * 5  # High penalty for vulnerable blots
        
        # Reward making points
        for pos in range(24):
            if self.is_point(board, pos, my_color):
                point_val = self.calculate_point_value(board, pos)
                score += point_val
                
        # Reward anchors in opponent's home board
        opp_home = range(0, 6) if my_color == Color.WHITE else range(18, 24)
        for pos in opp_home:
            if self.is_point(board, pos, my_color):
                score += 3  # Anchors are valuable
                
        # Penalize checkers on the bar
        bar_count = board.white_bar if my_color == Color.WHITE else board.black_bar
        score -= bar_count * 5  # Heavy penalty for each checker on the bar
        
        # Reward having checkers off the board
        off_count = board.white_off if my_color == Color.WHITE else board.black_off
        score += off_count * 2
        
        return score
    
    def calculate_vulnerability(self, board, pos):
        """Calculate how vulnerable a blot is to being hit."""
        my_color = board.turn
        opp_color = Color.BLACK if my_color == Color.WHITE else Color.WHITE
        
        # Count how many opponent checkers can potentially hit this blot
        hit_ways = 0
        
        # Check direct hits
        for i in range(24):
            if not self.has_checkers(board, i, opp_color):
                continue
                
            # Check if opponent can hit with normal dice values
            for die in range(1, 7):
                target = i + die * opp_color.value
                if 0 <= target < 24 and target == pos:
                    # Weight by the probability (1/6 for each die value)
                    checker_count = len([c for c in board.positions[i] if c == opp_color])
                    hit_ways += checker_count * (1/6)
        
        # Check hits from the bar
        if (opp_color == Color.WHITE and board.white_bar > 0) or (opp_color == Color.BLACK and board.black_bar > 0):
            entry_point = 24 - pos - 1 if opp_color == Color.WHITE else pos
            if 1 <= entry_point <= 6:
                hit_ways += (board.white_bar if opp_color == Color.WHITE else board.black_bar) * (1/6)
        
        # Strategic position importance factor
        position_factor = 1.0
        if self.is_in_home_board(board, pos, my_color):
            position_factor = 1.5  # Blots in home board are more vulnerable
        
        # Higher risk for blots on key strategic points
        if pos in [5, 7, 12, 17, 19]:  # Bar point, 5-point, etc.
            position_factor *= 1.2
            
        return hit_ways * position_factor
    
    def calculate_point_value(self, board, pos):
        """Calculate the strategic value of making a point."""
        my_color = board.turn
        
        # Base value from point importance table
        base_value = self.point_importance[pos]
        
        # Adjust based on game phase
        if self.is_early_game(board):
            # In early game, middle points and home board points are most valuable
            if 6 <= pos <= 17:
                base_value *= 1.2
        elif self.is_middle_game(board):
            # In middle game, priming points are more valuable
            if (my_color == Color.WHITE and 12 <= pos <= 17) or (my_color == Color.BLACK and 6 <= pos <= 11):
                base_value *= 1.3
        elif self.is_endgame(board):
            # In endgame, home board points are most valuable
            if self.is_in_home_board(board, pos, my_color):
                base_value *= 1.5
        
        # Higher value for consecutive points (priming)
        if self.has_adjacent_point(board, pos, my_color):
            base_value *= 1.4
            
        return base_value
    
    def has_adjacent_point(self, board, pos, color):
        """Check if there's an adjacent point owned by the same player."""
        if pos > 0 and self.is_point(board, pos-1, color):
            return True
        if pos < 23 and self.is_point(board, pos+1, color):
            return True
        return False
        
    def evaluate_structure(self, board):
        """Evaluate board structure and control."""
        score = 0
        my_color = board.turn
        
        # Evaluate primes (consecutive made points)
        prime_length, prime_start, prime_end = self.find_longest_prime(board, my_color)
        score += prime_length ** 2  # Square for non-linear reward of longer primes
        
        # Evaluate blocking effectiveness
        if prime_length >= 2:
            blocking_value = self.evaluate_blocking(board, my_color, prime_start, prime_end)
            score += blocking_value
        
        # Value of controlling key points
        key_points = [5, 7, 12, 17, 19]  # 5-point, bar-point, etc.
        for pos in key_points:
            if self.is_point(board, pos, my_color):
                score += 3
        
        # Home board strength (# of points made in home)
        home_points = sum(1 for pos in self.get_home_board_range(my_color) if self.is_point(board, pos, my_color))
        score += home_points * 2
        
        # Evaluate checker distribution (avoid stacks)
        distribution_score = self.evaluate_distribution(board, my_color)
        score += distribution_score
        
        return score
    
    def find_longest_prime(self, board, color):
        """Find the longest prime (consecutive points) and its location."""
        longest = 0
        longest_start = -1
        longest_end = -1
        current = 0
        current_start = -1
        
        for i in range(24):
            if self.is_point(board, i, color):
                if current == 0:
                    current_start = i
                current += 1
                if current > longest:
                    longest = current
                    longest_start = current_start
                    longest_end = i
            else:
                current = 0
                current_start = -1
                
        return longest, longest_start, longest_end
    
    def evaluate_blocking(self, board, color, prime_start, prime_end):
        """Evaluate how effectively we're blocking opponent's checkers."""
        opp_color = Color.BLACK if color == Color.WHITE else Color.WHITE
        
        # Count opponent checkers trapped behind our prime
        trapped_count = 0
        
        # Define the "behind" range based on the direction of movement
        if color == Color.WHITE:
            behind_range = range(0, prime_start)
        else:
            behind_range = range(prime_end+1, 24)
        
        # Count opponent checkers that are trapped
        for pos in behind_range:
            trapped_count += len([c for c in board.positions[pos] if c == opp_color])
        
        # Count checkers on the bar
        if opp_color == Color.WHITE:
            trapped_count += board.white_bar
        else:
            trapped_count += board.black_bar
            
        # Value is higher for longer primes and more trapped checkers
        blocking_value = (prime_end - prime_start + 1) * trapped_count * 0.7
        
        return blocking_value
    
    def evaluate_distribution(self, board, color):
        """Evaluate checker distribution (penalize inefficient stacking)."""
        score = 0
        
        # Penalize having more than 3 checkers on non-home points
        home_range = self.get_home_board_range(color)
        
        for pos in range(24):
            if pos not in home_range:
                checker_count = len([c for c in board.positions[pos] if c == color])
                if checker_count > 3:
                    score -= (checker_count - 3) * 0.5  # Penalty for excessive stacking
                    
        # Reward connectivity (ability to make new points)
        score += self.evaluate_connectivity(board, color)
        
        return score
    
    def evaluate_connectivity(self, board, color):
        """Evaluate checker connectivity (ability to make new points)."""
        connectivity_score = 0
        
        for i in range(24):
            if not self.is_point(board, i, color) and not self.is_blot(board, i, color):
                # Check if we can potentially make this point
                for j in range(24):
                    if self.has_checkers(board, j, color):
                        for die in range(1, 7):
                            if (j + die * color.value) == i:
                                connectivity_score += 0.2  # Small bonus for connectivity
        
        return connectivity_score
    
    def evaluate_race(self, board):
        """Evaluate racing position (pip count and bearing off)."""
        my_color = board.turn
        
        # If not in a racing position, use a reduced weight for this component
        race_weight = 1.0 if self.is_racing_position(board) else 0.3
        
        # Calculate pip counts (lower is better)
        my_pip_count = self.calculate_pip_count(board, my_color)
        opp_pip_count = self.calculate_pip_count(board, self.opposite_color(my_color))
        
        # Race advantage (positive if ahead, negative if behind)
        pip_advantage = opp_pip_count - my_pip_count
        
        # Normalize to a reasonable range
        normalized_advantage = min(20, max(-20, pip_advantage)) / 20
        
        # Bearing off efficiency
        bearing_efficiency = 0
        if self.is_bearing_off_phase(board, my_color):
            bearing_efficiency = self.evaluate_bearing_off_distribution(board, my_color)
        
        return (normalized_advantage * 10 + bearing_efficiency) * race_weight
    
    def calculate_pip_count(self, board, color):
        """Calculate pip count (total distance to bear off)."""
        total = 0
        
        for pos in range(24):
            checkers = len([c for c in board.positions[pos] if c == color])
            
            if color == Color.WHITE:
                distance = 24 - pos  # Distance from position to off (for white)
            else:
                distance = pos + 1  # Distance from position to off (for black)
                
            total += checkers * distance
        
        # Add pip count for checkers on the bar
        bar_count = board.white_bar if color == Color.WHITE else board.black_bar
        bar_distance = 25  # Maximum distance
        total += bar_count * bar_distance
        
        return total
    
    def evaluate_bearing_off_distribution(self, board, color):
        """Evaluate how well checkers are distributed for bearing off."""
        score = 0
        home_board = self.get_home_board_range(color)
        
        # Count checkers in home board
        checker_counts = []
        for pos in home_board:
            count = len([c for c in board.positions[pos] if c == color])
            checker_counts.append(count)
        
        # Ideal distribution depends on the exact situation
        # This is a simplified approximation
        
        # Penalize gaps (empty points followed by checkers)
        has_gap = False
        for i in range(len(checker_counts)-1):
            if checker_counts[i] == 0 and sum(checker_counts[i+1:]) > 0:
                has_gap = True
                score -= 2  # Penalty for each gap
        
        # If no gaps, reward having checkers on higher points
        if not has_gap:
            for i, count in enumerate(checker_counts):
                position_weight = i + 1  # Higher points get higher weights
                score += count * position_weight * 0.2
        
        # Avoid stacking too many checkers on a single point
        stacking_penalty = sum((count - 3) ** 2 for count in checker_counts if count > 3)
        score -= stacking_penalty * 0.3
        
        return score
    
    def evaluate_special_situations(self, board):
        """Evaluate special strategic situations."""
        score = 0
        my_color = board.turn
        
        # Back game strategy (when significantly behind in the race)
        if self.is_back_game_position(board, my_color):
            score += self.evaluate_back_game(board, my_color)
        
        # Holding game (keeping anchor in opponent's home board)
        if self.is_holding_game(board, my_color):
            score += self.evaluate_holding_position(board, my_color)
        
        # Endgame bearing off optimization
        if self.is_endgame(board):
            score += self.evaluate_endgame_efficiency(board, my_color)
        
        # Blitz strategy (aggressive forward game)
        if self.is_blitz_position(board, my_color):
            score += self.evaluate_blitz(board, my_color)
        
        return score
    
    def is_back_game_position(self, board, color):
        """Check if we're in a back game position."""
        # Back games are typically when you're behind in the race 
        # but have 2+ anchors in opponent's home board
        opp_home = self.get_opponent_home_board_range(color)
        anchors_in_opp_home = sum(1 for pos in opp_home if self.is_point(board, pos, color))
        
        # Check pip count disadvantage
        my_pip_count = self.calculate_pip_count(board, color)
        opp_pip_count = self.calculate_pip_count(board, self.opposite_color(color))
        pip_disadvantage = my_pip_count - opp_pip_count
        
        return anchors_in_opp_home >= 2 and pip_disadvantage > 20
    
    def evaluate_back_game(self, board, color):
        """Evaluate back game strength."""
        score = 0
        opp_home = self.get_opponent_home_board_range(color)
        
        # Reward having multiple anchors in opponent's home board
        for pos in opp_home:
            if self.is_point(board, pos, color):
                # Lower points are more valuable for back games
                point_value = 6 - (pos % 6 if color == Color.WHITE else 5 - pos % 6)
                score += point_value * 1.5
        
        # Reward having builders (checkers to potentially make more anchors)
        builders = 0
        for pos in range(24):
            checker_count = len([c for c in board.positions[pos] if c == color])
            if pos not in opp_home and checker_count > 0:
                builders += min(2, checker_count)  # Count up to 2 builders per position
        
        score += builders * 0.5
        
        return score
    
    def is_holding_game(self, board, color):
        """Check if we're in a holding game position."""
        # Holding games involve keeping an anchor in opponent's home board
        # while bringing other checkers around
        opp_home = self.get_opponent_home_board_range(color)
        anchors_in_opp_home = sum(1 for pos in opp_home if self.is_point(board, pos, color))
        
        # We should have exactly one anchor
        if anchors_in_opp_home != 1:
            return False
            
        # Most other checkers should be advanced
        my_home = self.get_home_board_range(color)
        advanced_checkers = sum(len([c for c in board.positions[pos] if c == color]) for pos in my_home)
        
        return advanced_checkers >= 10  # At least 10 checkers advanced
    
    def evaluate_holding_position(self, board, color):
        """Evaluate holding game strength."""
        score = 0
        opp_home = self.get_opponent_home_board_range(color)
        
        # Reward the anchor (especially on key points)
        for pos in opp_home:
            if self.is_point(board, pos, color):
                # Lower points are better anchors
                anchor_value = 6 - (pos % 6 if color == Color.WHITE else 5 - pos % 6)
                score += anchor_value * 2
                break  # Only count one anchor
        
        # Reward advanced position of other checkers
        my_home = self.get_home_board_range(color)
        for pos in my_home:
            checker_count = len([c for c in board.positions[pos] if c == color])
            score += checker_count * 0.7
        
        return score
    
    def is_blitz_position(self, board, color):
        """Check if we're in a blitz position (aggressive attack)."""
        # Blitz involves attacking opponent's checkers while building a prime
        
        # We should have a strong prime
        prime_length, _, _ = self.find_longest_prime(board, color)
        if prime_length < 4:
            return False
            
        # Count attacked opponent blots
        opp_color = self.opposite_color(color)
        attacked_blots = 0
        for pos in range(24):
            if self.is_blot(board, pos, opp_color) and self.calculate_vulnerability(board, pos) > 0.5:
                attacked_blots += 1
                
        # Count opponent checkers on the bar
        bar_count = board.black_bar if opp_color == Color.BLACK else board.white_bar
        
        return attacked_blots + bar_count >= 2
    
    def evaluate_blitz(self, board, color):
        """Evaluate blitz position strength."""
        score = 0
        
        # Reward prime strength
        prime_length, prime_start, prime_end = self.find_longest_prime(board, color)
        score += prime_length ** 2
        
        # Reward attacking opponent blots
        opp_color = self.opposite_color(color)
        for pos in range(24):
            if self.is_blot(board, pos, opp_color):
                vulnerability = self.calculate_vulnerability(board, pos)
                score += vulnerability * 3
                
        # Reward having opponent checkers on the bar
        bar_count = board.black_bar if opp_color == Color.BLACK else board.white_bar
        score += bar_count * 4
        
        return score
    
    def evaluate_endgame_efficiency(self, board, color):
        """Evaluate efficiency during the endgame."""
        # Only relevant when bearing off
        if not self.is_bearing_off_phase(board, color):
            return 0
            
        score = 0
        home_board = self.get_home_board_range(color)
        
        # Reward crossover minimization
        crossovers = self.calculate_crossovers(board, color)
        score -= crossovers * 2  # Penalty for crossovers
        
        # Reward efficient use of dice
        # (This would ideally simulate different dice rolls and evaluate efficiency)
        
        # Simple approximation: reward having checkers on higher points
        weighted_sum = 0
        for i, pos in enumerate(home_board):
            checker_count = len([c for c in board.positions[pos] if c == color])
            pos_weight = i + 1  # 1 for the 1-point, 6 for the 6-point
            weighted_sum += checker_count * pos_weight
            
        # Normalize and add to score
        total_checkers = sum(len([c for c in board.positions[pos] if c == color]) for pos in home_board)
        if total_checkers > 0:
            normalized_weight = weighted_sum / (total_checkers * 3.5)  # 3.5 is average point value
            score += normalized_weight * 5
            
        return score
    
    def calculate_crossovers(self, board, color):
        """Calculate potential crossovers during bearing off."""
        home_board = list(self.get_home_board_range(color))
        
        # Count checkers that may need to "crossover" others
        crossovers = 0
        
        # For each position in home board
        for i, pos in enumerate(home_board):
            checker_count = len([c for c in board.positions[pos] if c == color])
            
            # Check if there are gaps before this position
            for j in range(i):
                earlier_pos = home_board[j]
                if len([c for c in board.positions[earlier_pos] if c == color]) == 0:
                    # Found a gap - all checkers at this position might need to crossover
                    crossovers += checker_count
                    break
                    
        return crossovers
    
    # Helper functions
    
    def is_point(self, board, pos, color):
        """Check if we have a made point at this position."""
        return len([c for c in board.positions[pos] if c == color]) >= 2
    
    def is_blot(self, board, pos, color):
        """Check if we have a blot (single checker) at this position."""
        return len([c for c in board.positions[pos] if c == color]) == 1
    
    def has_checkers(self, board, pos, color):
        """Check if there are any checkers of the given color at the position."""
        return len([c for c in board.positions[pos] if c == color]) > 0
    
    def is_racing_position(self, board):
        """Check if the game is in a racing position (no contact)."""
        # No checkers on the bar
        if board.white_bar > 0 or board.black_bar > 0:
            return False
            
        # Check if no checkers can hit each other
        white_furthest_back = self.furthest_back_checker(board, Color.WHITE)
        black_furthest_forward = self.furthest_forward_checker(board, Color.BLACK)
        
        # If white's furthest back checker is ahead of black's furthest forward,
        # then we're in a racing position
        return white_furthest_back > black_furthest_forward
    
    def furthest_back_checker(self, board, color):
        """Find the furthest back checker position."""
        if color == Color.WHITE:
            for pos in range(24):
                if len([c for c in board.positions[pos] if c == color]) > 0:
                    return pos
        else:
            for pos in range(23, -1, -1):
                if len([c for c in board.positions[pos] if c == color]) > 0:
                    return pos
        return -1
    
    def furthest_forward_checker(self, board, color):
        """Find the furthest forward checker position."""
        if color == Color.WHITE:
            for pos in range(23, -1, -1):
                if len([c for c in board.positions[pos] if c == color]) > 0:
                    return pos
        else:
            for pos in range(24):
                if len([c for c in board.positions[pos] if c == color]) > 0:
                    return pos
        return -1
    
    def is_early_game(self, board):
        """Check if the game is in the early phase."""
        # Early game is usually before significant pieces are borne off
        # or while there are still checkers far from home
        total_off = board.white_off + board.black_off
        return total_off < 5
    
    def is_middle_game(self, board):
        """Check if the game is in the middle phase."""
        total_off = board.white_off + board.black_off
        return 5 <= total_off < 10
    
    def is_endgame(self, board):
        """Check if the game is in the endgame phase."""
        total_off = board.white_off + board.black_off
        return total_off >= 10
    
    def is_bearing_off_phase(self, board, color):
        """Check if a player is in the bearing off phase."""
        # All checkers must be in home board or borne off
        for pos in range(24):
            if pos not in self.get_home_board_range(color):
                if len([c for c in board.positions[pos] if c == color]) > 0:
                    return False
        
        # No checkers on the bar
        if (color == Color.WHITE and board.white_bar > 0) or (color == Color.BLACK and board.black_bar > 0):
            return False
            
        return True
    
    def get_home_board_range(self, color):
        """Get the range of positions in a player's home board."""
        return range(18, 24) if color == Color.WHITE else range(0, 6)
    
    def get_opponent_home_board_range(self, color):
        """Get the range of positions in the opponent's home board."""
        return range(0, 6) if color == Color.WHITE else range(18, 24)
    
    def is_in_home_board(self, board, pos, color):
        """Check if a position is in a player's home board."""
        if color == Color.WHITE:
            return 18 <= pos < 24
        else:
            return 0 <= pos < 6
    
    def opposite_color(self, color):
        """Get the opposite color."""
        return Color.BLACK if color == Color.WHITE else Color.WHITE
    
    def adjust_weights_for_phase(self, board):
        """Adjust strategy weights based on game phase."""
        # Early game
        if self.is_early_game(board):
            self.weights = {'safety': 0.4, 'structure': 0.4, 'race': 0.1, 'special': 0.1}
        
        # Middle game
        elif self.is_middle_game(board):
            self.weights = {'safety': 0.3, 'structure': 0.4, 'race': 0.2, 'special': 0.1}
        
        # Racing position
        elif self.is_racing_position(board):
            self.weights = {'safety': 0.1, 'structure': 0.1, 'race': 0.7, 'special': 0.1}
        
        # Endgame
        elif self.is_endgame(board):
            self.weights = {'safety': 0.0, 'structure': 0.1, 'race': 0.8, 'special': 0.1}
        
        # Back game
        elif self.is_back_game_position(board, board.turn):
            self.weights = {'safety': 0.3, 'structure': 0.2, 'race': 0.1, 'special': 0.4}
        
        # Blitz position
        elif self.is_blitz_position(board, board.turn):
            self.weights = {'safety': 0.2, 'structure': 0.3, 'race': 0.1, 'special': 0.4}


# Usage example:
if __name__ == "__main__":
    # Create agents
    white_agent = HeuristicBackgammonAgent()
    black_agent = HeuristicBackgammonAgent()
    
    # Play a sample game
    board = Board()
    
    while not board.game_over:
        # Determine current agent
        current_agent = white_agent if board.turn == Color.WHITE else black_agent
        
        # Roll dice
        board.roll_dice()
        
        # Get and apply move
        move = current_agent.select_move(board)
        board.move_from_sequence(move)
        
        # Print current state
        print(f"Player: {board.turn}")
        print(f"Move: {move}")
        print(board)
        print("-" * 40)
    
    # Print winner
    if board.white_off == 15:
        print("White wins!")
    else:
        print("Black wins!")