from copy import deepcopy
import time
import random
import math
try:
    from Board import Board
except:
    from ..Board import Board


class Node:
    def __init__(self, state=None, move_sequence=None, parent=None):
        """Initialize a node in the MCTS tree."""
        self.state = state
        self.move_sequence = move_sequence  # The move that led to this state
        self.parent = parent
        self.children = {}  # Map move sequences to child nodes
        self.N = 0  # Visit count
        self.Q = 0  # Total reward
        self.Q2 = 0  # Sum of squared rewards (for UCB1 Tuned)

        # Get valid moves if state is provided
        if state:
            if not state.rolled:
                state.roll_dice()
            self.untried_moves = state.valid_moves
        else:
            self.untried_moves = []
            
    def get_board_hash(self):
        """Generate a hash representation of the board state for comparison."""
        if not self.state:
            return None
        
        # Create a tuple of all the important board state components
        board_tuple = (
            tuple(self.state.positions),
            self.state.turn,
            tuple(sorted(self.state.dice)),
            self.state.white_bar,
            self.state.black_bar,
            self.state.white_off,
            self.state.black_off
        )
        return hash(board_tuple)

    def value(self, exploration_weight):
        """Calculate the UCB1 Tuned value of this node."""
        if self.N == 0:
            return float('inf')

        # Exploitation term (average reward)
        exploitation = self.Q / self.N
        
        # Calculate variance estimate
        variance = (self.Q2 / self.N) - (exploitation ** 2)
        variance = max(0, variance)  # Ensure variance is non-negative
        
        # Calculate the upper bound on variance (V)
        log_parent = math.log(self.parent.N) if self.parent.N > 1 else 0
        V = variance + math.sqrt(2 * log_parent / self.N)
        
        # Use min(1/4, V) in the exploration term (UCB1 Tuned formula)
        exploration = exploration_weight * math.sqrt(log_parent / self.N * min(0.25, V))

        return exploitation + exploration

    def is_fully_expanded(self):
        """Check if all valid moves have been expanded."""
        return len(self.untried_moves) == 0

    def best_child(self, exploration_weight):
        """Return the child with the highest UCB1 Tuned value."""
        if not self.children:
            return None

        # Find child with highest UCB1 Tuned value
        best_value = float('-inf')
        best_children = []

        for child in self.children.values():
            value = child.value(exploration_weight)
            if value > best_value:
                best_value = value
                best_children = [child]
            elif value == best_value:
                best_children.append(child)

        # Return a random choice among the best children
        return random.choice(best_children) if best_children else None


class MCTSBackgammonAgent:
    def __init__(self, exploration_weight=1.0, simulation_depth=50):
        """Initialize the MCTS agent."""
        self.exploration_weight = exploration_weight
        self.simulation_depth = simulation_depth
        self.root = None
        self.player_color = 0
        self.sim_count = 0
        
        # Probability table for D²/S method (probability, ratio) pairs
        self.probability_table = [
            (0.50, 0.00000000), (0.51, 0.00137882), (0.52, 0.00551875), (0.53, 0.0124302),
            (0.54, 0.0221307), (0.55, 0.0346450), (0.56, 0.0500050), (0.57, 0.0682506),
            (0.58, 0.0894296), (0.59, 0.113598), (0.60, 0.140821), (0.61, 0.171174),
            (0.62, 0.204741), (0.63, 0.241618), (0.64, 0.281913), (0.65, 0.325747),
            (0.66, 0.373256), (0.67, 0.424591), (0.68, 0.479920), (0.69, 0.539433),
            (0.70, 0.603341), (0.71, 0.671879), (0.72, 0.745311), (0.73, 0.823934),
            (0.74, 0.908082), (0.75, 0.998131), (0.76, 1.09451), (0.77, 1.19769), 
            (0.78, 1.30824), (0.79, 1.42679), (0.80, 1.55407), (0.81, 1.69092), 
            (0.82, 1.83834), (0.83, 1.99749), (0.84, 2.16975), (0.85, 2.35678), 
            (0.86, 2.56060), (0.87, 2.78365), (0.88, 3.02902), (0.89, 3.30059), 
            (0.90, 3.60337), (0.91, 3.94399), (0.92, 4.33145), (0.93, 4.77844), 
            (0.94, 5.30360), (0.95, 5.93596), (0.96, 6.72439), (0.97, 7.76102), 
            (0.98, 9.25404), (0.99, 11.8737)
        ]
        
        # Reorganize the table for easier lookup - (ratio, probability) pairs sorted by ratio
        self.ratio_table = sorted([(ratio, prob) for prob, ratio in self.probability_table])

    def search(self, time_budget):
        """Run MCTS for the specified time."""
        start_time = time.time()

        while time.time() - start_time < time_budget:
            # 1. Selection: traverse tree until we reach a leaf node
            node = self.select_node()

            # 2. Expansion: add a child if not terminal, not passed, and not fully expanded
            if not node.is_fully_expanded() and not node.state.game_over and not node.state.passed:
                # Choose an untried move
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)

                # Create a new state by applying the move
                new_state = deepcopy(node.state)
                new_state.move_from_sequence(move)

                # Create a new child node
                move_key = tuple(tuple(m) for m in move)
                child = Node(state=new_state, move_sequence=move, parent=node)
                node.children[move_key] = child

                # Use the new node for simulation
                node = child

            # 3. Simulation: play out to terminal state, passed state, or depth limit
            result = self.simulate(node)

            # 4. Backpropagation: update statistics up the tree
            self.backpropagate(node, result)

            self.sim_count += 1

    def select_node(self):
        """Select a node to expand using UCB1 Tuned."""
        node = self.root

        # Keep selecting best child until reaching a leaf, unexpanded node, or passed state
        while node.children and node.is_fully_expanded() and not node.state.game_over and not node.state.passed:
            node = node.best_child(self.exploration_weight)

        return node

    def simulate(self, node):
        """Simulate a random game from node and return the result."""
        state = deepcopy(node.state)
        depth = 0
        
        # If the board is already in a passed state, use D²/S method
        if state.passed:
            return self.calculate_d2s_win_probability(state)

        # Simulate random play until terminal state, passed state, or depth limit
        while not state.game_over and not state.passed and depth < self.simulation_depth:
            # Get valid moves
            if not state.rolled:
                state.roll_dice()
            valid_moves = state.valid_moves

            # If no valid moves, roll dice or swap turn
            if not valid_moves:
                if state.rolled:
                    state.swap_turn()
                state.roll_dice()
                continue

            # Choose a random move and apply it
            move = random.choice(valid_moves)
            state.move_from_sequence(move)
            depth += 1

        # Return result for terminal state, D²/S probability for passed state, 
        # or heuristic evaluation for depth limit
        if state.game_over:
            if (self.player_color == 1 and state.white_off == 15) or \
               (self.player_color == 0 and state.black_off == 15):
                return 1  # Win for the current player
            return -1  # Loss for the current player
        elif state.passed:
            # Use D²/S method to estimate win probability
            return self.calculate_d2s_win_probability(state)
        else:
            # Use pip count ratio as a heuristic
            if self.player_color == 1:  # White player
                return 2 * (state.black_left / (state.white_left + state.black_left)) - 1
            else:  # Black player
                return 2 * (state.white_left / (state.white_left + state.black_left)) - 1

    def calculate_d2s_win_probability(self, state):
        """Calculate win probability using the adjusted D²/S method for a passed board."""
        white_pips = state.white_left
        black_pips = state.black_left
        
        # Determine which player has fewer pips (ahead) and which has more (behind)
        if white_pips <= black_pips:
            Y = white_pips  # White is ahead or tied
            X = black_pips  # Black is behind or tied
            white_is_ahead = True
        else:
            Y = black_pips  # Black is ahead
            X = white_pips  # White is behind
            white_is_ahead = False
        
        # Calculate Adjusted Difference (Δ)
        delta = Y - (X - 4)
        
        # Calculate Sum (S)
        S = Y + X
        
        # Calculate Numerator Value: Δ² + Δ/7
        numerator = delta**2 + delta/7
        
        # Calculate Denominator Value: S - 25
        denominator = max(1, S - 25)  # Ensure denominator is not zero
        
        # Calculate Ratio
        ratio = numerator / denominator
        
        # Get win probability for the player who is ahead using the lookup table
        win_prob_ahead = self.lookup_win_probability(ratio)
        
        # Convert to [-1, 1] value for MCTS based on player color
        if self.player_color == 1:  # White player
            if white_is_ahead:
                return 2 * win_prob_ahead - 1  # White is ahead
            else:
                return 2 * (1 - win_prob_ahead) - 1  # White is behind
        else:  # Black player
            if white_is_ahead:
                return 2 * (1 - win_prob_ahead) - 1  # Black is behind
            else:
                return 2 * win_prob_ahead - 1  # Black is ahead

    def lookup_win_probability(self, ratio):
        """Lookup win probability based on D²/S ratio using the provided table."""
        # If ratio is below the minimum in the table, return the minimum probability
        if ratio <= self.ratio_table[0][0]:
            return self.ratio_table[0][1]
        
        # If ratio is above the maximum in the table, return the maximum probability
        if ratio >= self.ratio_table[-1][0]:
            return self.ratio_table[-1][1]
        
        # Find the two closest ratios in the table
        for i in range(len(self.ratio_table) - 1):
            if self.ratio_table[i][0] <= ratio < self.ratio_table[i+1][0]:
                r1, p1 = self.ratio_table[i]
                r2, p2 = self.ratio_table[i+1]
                
                # Linear interpolation between the two closest probabilities
                return p1 + (p2 - p1) * (ratio - r1) / (r2 - r1)
        
        # This should not happen, but return 0.5 as a fallback
        return 0.5

    def backpropagate(self, node, result):
        """Update statistics in all nodes along path from node to root."""
        current = node
        current_result = result

        while current is not None:
            current.N += 1
            current.Q += current_result
            current.Q2 += current_result ** 2  # Update sum of squared rewards for variance calculation
            current = current.parent
            current_result = -current_result  # Negate result for parent (other player)

    def best_move(self):
        """Return move with highest visit count from root's children."""
        if not self.root.children:
            return []

        # Find children with highest visit count
        max_visits = max(child.N for child in self.root.children.values())
        best_children = [child for child in self.root.children.values() if child.N == max_visits]

        # Choose one randomly if there are multiple
        best_child = random.choice(best_children)

        return best_child.move_sequence
        
    def find_matching_child(self, board):
        """Find a child node that matches the given board state."""
        if not self.root:
            return None
            
        # Create a hash of the current board state
        board_hash = hash((
            tuple(board.positions),
            board.turn,
            tuple(sorted(board.dice)),
            board.white_bar,
            board.black_bar,
            board.white_off,
            board.black_off
        ))
        
        # Check all children for a matching state
        for child in self.root.children.values():
            if child.get_board_hash() == board_hash:
                return child
                
        return None


class BackgammonMCTSAgent:
    def __init__(self, exploration_weight=1.0, simulation_depth=50, time_budget=2.0):
        self.mcts = MCTSBackgammonAgent(
            exploration_weight=exploration_weight,
            simulation_depth=simulation_depth
        )
        self.time_budget = time_budget
        self.previous_tree = None  # Store the MCTS tree between moves

    def select_move(self, board):
        if not board.valid_moves:
            return []

        # If we only have one valid move, no need to run MCTS
        if len(board.valid_moves) == 1:
            # Reset the tree since we didn't use MCTS
            self.previous_tree = None
            return board.valid_moves[0]
            
        # If the board is already in a passed state, pick the move that minimizes pip count
        if board.passed:
            # Reset the tree for passed boards (using direct evaluation)
            self.previous_tree = None
            return self.select_move_for_passed_board(board)

        # Set the player color for the MCTS agent
        self.mcts.player_color = board.turn
        
        # Check if we can reuse the previous tree
        if self.previous_tree:
            # Look for a matching child node from the previous root
            matching_child = self.mcts.find_matching_child(board)
            
            if matching_child:
                # We found a matching child, use it as the new root
                matching_child.parent = None  # Detach from parent
                self.mcts.root = matching_child
                print("Reusing subtree")
            else:
                # No matching child found, create a new tree
                self.mcts.root = Node(state=deepcopy(board))
                print("Creating new tree - no matching child")
        else:
            # No previous tree, create a new one
            self.mcts.root = Node(state=deepcopy(board))
            print("Creating new tree - no previous tree")

        # Run MCTS search
        self.mcts.search(self.time_budget)
        
        # Store the current tree for next time
        self.previous_tree = self.mcts.root
        
        return self.mcts.best_move()
        
    def select_move_for_passed_board(self, board):
        """Select a move for a board that's already in a passed state.
        Choose the move that minimizes the player's pip count."""
        best_move = None
        min_pips = float('inf')
        
        for move in board.valid_moves:
            # Create a copy and apply the move
            test_board = deepcopy(board)
            test_board.move_from_sequence(move)
            
            # Get the player's pip count after the move
            if board.turn == 1:  # White
                pips = test_board.white_left
            else:  # Black
                pips = test_board.black_left
                
            # Keep track of the move that minimizes pips
            if pips < min_pips:
                min_pips = pips
                best_move = move
                
        return best_move