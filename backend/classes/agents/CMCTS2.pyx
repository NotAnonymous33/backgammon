import time
import random
import math
import cython
from .HeuristicAgent import HeuristicAgent
from .NNAgent import FinalNNAgent
import torch

cdef class Node:
    cdef public object state
    cdef public list move_sequence
    cdef public object parent
    cdef public dict children
    cdef public int N
    cdef public float Q
    cdef public float Q2  # Sum of squared rewards for variance calculation
    cdef public list untried_moves
    
    def __init__(self, state=None, move_sequence=None, parent=None):
        """Initialize a node in the MCTS tree."""
        self.state = state
        self.move_sequence = move_sequence  # The move that led to this state
        self.parent = parent
        self.children = {}  # Map move sequences to child nodes
        self.N = 0  # Visit count
        self.Q = 0  # Total reward
        self.Q2 = 0  # Sum of squared rewards for UCB1 Tuned

        # Get valid moves if state is provided
        if state:
            if not state.rolled:
                state.roll_dice()
            self.untried_moves = state.valid_moves
        else:
            self.untried_moves = []

    @cython.ccall
    def get_board_hash(self):
        """Generate a hash representation of the board state for comparison."""
        if not self.state:
            return None
        
        # Create a tuple of all the important board state components
        board_tuple = (
            tuple(self.state.positions),
            self.state.turn,
            tuple(sorted(self.state.dice)) if self.state.dice else tuple(),
            self.state.white_bar,
            self.state.black_bar,
            self.state.white_off,
            self.state.black_off
        )
        return hash(board_tuple)
        
    @cython.ccall
    @cython.cdivision(True)
    def value(self, float exploration_weight):
        """Calculate the UCB1 Tuned value of this node."""
        if self.N == 0:
            return float('inf')

        # Exploitation term (average reward)
        cdef float exploitation = self.Q / self.N
        
        # Calculate variance estimate
        cdef float variance = (self.Q2 / self.N) - (exploitation * exploitation)
        variance = max(0, variance)  # Ensure variance is non-negative
        
        # Calculate the upper bound on variance (V)
        cdef float log_parent = math.log(max(1, self.parent.N))
        cdef float V = variance + math.sqrt(2 * log_parent / self.N)
        
        # Use min(1/4, V) in the exploration term (UCB1 Tuned formula)
        cdef float min_term = min(0.25, V)
        cdef float exploration = exploration_weight * math.sqrt(log_parent / self.N * min_term)

        return exploitation + exploration

    @cython.ccall
    def is_fully_expanded(self):
        """Check if all valid moves have been expanded."""
        return not self.untried_moves

    @cython.ccall
    def best_child(self, float exploration_weight):
        """Return the child with the highest UCT value."""
        if not self.children:
            return None
        
        # Find child with highest UCT value
        cdef float best_value = float('-inf')
        cdef list best_children = []
        cdef object child
        cdef float value
        for child in self.children.values():
            value = child.value(exploration_weight)
            if value > best_value:
                best_value = value
                best_children = [child]
            elif value == best_value:
                best_children.append(child)

        # Return a random choice among the best children
        return random.choice(best_children) if best_children else None


cdef class MCTSBackgammonAgent:
    cdef public float exploration_weight
    cdef public int simulation_depth
    cdef public int eval_depth
    cdef public str mode
    cdef public object heuristic_agent
    cdef public object nn_agent
    cdef public object root
    cdef public int player_color
    cdef public int sim_count
    cdef public list probability_table
    cdef public list ratio_table

    def __init__(self,
                 float exploration_weight=1.0,
                 int simulation_depth=50,
                 mode="default",
                 int eval_depth=-1,
                 heuristic_agent=None,
                 nn_agent=None):
        """Initialize the MCTS agent with a mode and evaluation depth."""
        self.exploration_weight = exploration_weight
        self.simulation_depth = simulation_depth
        # when eval_depth < 0, just use simulation_depth
        self.eval_depth = eval_depth if eval_depth > 0 else simulation_depth
        self.mode = mode  # "default", "heuristic", or "neural"
        # allow injection of your own agents, else build defaults
        if mode == "heuristic":
            self.heuristic_agent = heuristic_agent or HeuristicAgent()
        else:
            self.heuristic_agent = None
        
        if mode == "neural":
            self.nn_agent = nn_agent or FinalNNAgent()
        else:
            self.nn_agent = None


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

    @cython.ccall
    def search(self, float time_budget):
        cdef object node
        cdef list move
        cdef object new_state
        cdef object child
        cdef float result
        cdef tuple tuple_move
        start_time = time.time()
        self.sim_count = 0
        """Run MCTS for the specified time."""

        while time.time() - start_time < time_budget:
            # 1. Selection: traverse tree until we reach a leaf node
            node = self.select_node()

            # 2. Expansion: add a child if not terminal, not passed, and not fully expanded
            if not node.is_fully_expanded() and not node.state.game_over and not node.state.passed:
                # Choose an untried move
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)

                # Create a new state by applying the move
                new_state = node.state.clone()
                new_state.move_from_sequence(move)

                # Create a new child node
                child = Node(state=new_state, move_sequence=move, parent=node)
                
                # Convert move to tuple for dict key
                tuple_move = self.move_to_tuple(move)
                node.children[tuple_move] = child

                # Use the new node for simulation
                node = child

            # 3. Simulation: play out to terminal state, passed state, or depth limit
            result = self.simulate(node)

            # 4. Backpropagation: update statistics up the tree
            self.backpropagate(node, result)

            self.sim_count += 1
        return self.best_move()
            
    @cython.ccall
    def move_to_tuple(self, list move):
        """Convert a move (list of lists) to a tuple format suitable for dictionary keys."""
        result = []
        for m in move:
            result.append(tuple(m))
        return tuple(result)
        
    @cython.ccall
    def select_node(self):
        """Select a node to expand using UCB1 Tuned."""
        cdef object node = self.root

        # Keep selecting best child until reaching a leaf, unexpanded node, or passed state
        while node.children and node.is_fully_expanded() and not node.state.game_over and not node.state.passed:
            node = node.best_child(self.exploration_weight)

        return node

    @cython.ccall
    @cython.cdivision(True)
    cdef float simulate(self, object node):
        """Simulate a game, stopping at game over, passed, or eval_depth."""
        cdef object state = node.state.clone()
        cdef int depth = 0

        # If already passed, use D²/S
        if state.passed:
            return self.calculate_d2s_win_probability(state)

        # play random until one of the stopping criteria
        while not state.game_over and not state.passed and depth < self.eval_depth:
            if not state.rolled:
                state.roll_dice()
            valid_moves = state.valid_moves
            if not valid_moves:
                if state.rolled:
                    state.swap_turn()
                state.roll_dice()
                continue
            move = random.choice(valid_moves)
            state.move_from_sequence(move)
            depth += 1

        # 1) terminal win/loss
        if state.game_over:
            if (self.player_color == 1 and state.white_off == 15) \
               or (self.player_color == -1 and state.black_off == 15):
                return 1.0
            else:
                return -1.0

        # 2) passed -> D²/S
        if state.passed:
            return self.calculate_d2s_win_probability(state)

        # 3) depth reached: branch on mode
        if self.mode == "heuristic":
            # get raw heuristic score and squash to (-1,1)
            score = self.heuristic_agent.evaluate_board(state)
            return math.tanh(score)

        elif self.mode == "neural":
            # get network‐predicted win‐probability in [0,1]
            feats = torch.tensor(self.nn_agent.extract_features(state),
                                 dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                p = self.nn_agent.model(feats).item()
            # map to [-1,1]
            return 2.0 * p - 1.0

        else:
            # default: pip‐count heuristic as before
            if self.player_color == 1:
                return 2 * (state.black_left /
                            (state.white_left + state.black_left)) - 1
            else:
                return 2 * (state.white_left /
                            (state.white_left + state.black_left)) - 1
    
    @cython.ccall
    @cython.cdivision(True)
    def calculate_d2s_win_probability(self, object state):
        """Calculate win probability using the adjusted D²/S method for a passed board."""
        cdef int white_pips = state.white_left
        cdef int black_pips = state.black_left
        cdef int X, Y
        cdef bint white_is_ahead
        cdef float delta, S, numerator, denominator, ratio, win_prob_ahead
        
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
        numerator = delta*delta + delta/7
        
        # Calculate Denominator Value: S - 25
        denominator = max(1, S - 24.72588)  # Ensure denominator is not zero
        
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
    
    @cython.ccall
    @cython.cdivision(True)
    def lookup_win_probability(self, float ratio):
        """Lookup win probability based on D²/S ratio using the provided table."""
        cdef int i
        cdef float r1, p1, r2, p2
        
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

    @cython.ccall
    def backpropagate(self, object node, float result):
        """Update statistics in all nodes along path from node to root."""
        cdef object current = node
        cdef float current_result = result
        cdef float squared_result

        while current is not None:
            current.N += 1
            current.Q += current_result
            squared_result = current_result * current_result
            current.Q2 += squared_result  # Update sum of squared results for variance calculation
            current = current.parent
            current_result = -current_result  # Negate result for parent (other player)

    def best_move(self):
        """Return move with highest visit count from root's children."""
        if not self.root.children:
            return []

        max_visits = max(child.N for child in self.root.children.values())
        # print([child.N for child in self.root.children.values()])
        best_children = [child for child in self.root.children.values() if child.N == max_visits]

        return_move = random.choice(best_children).move_sequence
        self.root = self.root.children[self.move_to_tuple(return_move)]

        return return_move
        


cdef class MCTSAgent2:
    cdef public object mcts
    cdef public float time_budget
    
    def __init__(self,
                 float exploration_weight=1.0,
                 int simulation_depth=50,
                 mode="default",
                 int eval_depth=-1,
                 float time_budget=2.0,
                 heuristic_agent=None,
                 nn_agent=None):
        """High‐level wrapper: pass mode and eval_depth through."""
        self.mcts = MCTSBackgammonAgent(
            exploration_weight=exploration_weight,
            simulation_depth=simulation_depth,
            mode=mode,
            eval_depth=eval_depth,
            heuristic_agent=heuristic_agent,
            nn_agent=nn_agent
        )
        self.time_budget = time_budget


    @cython.ccall
    @cython.boundscheck(False)
    def select_move(self, object board):
        if not board.rolled:
            board.roll_dice()
        if not board.valid_moves:
            return []

        # If we only have one valid move, no need to run MCTS
        if len(board.valid_moves) == 1:
            return board.valid_moves[0]

        # If the board is already in a passed state, pick the move that minimizes pip count
        if board.passed:
            return self.select_move_for_passed_board(board)

        # Set the player color for the MCTS agent
        self.mcts.player_color = board.turn

        cdef object temp_node = Node(state=board.clone())
        cdef object target_hash = temp_node.get_board_hash()
        del temp_node
        cdef bint found = False
        if self.mcts.root is not None:
            found = False
            for child in self.mcts.root.children.values():
                if child.get_board_hash() == target_hash:
                    self.mcts.root = child
                    child.parent = None
                    found = True
                    break
            
            if not found:
                self.mcts.root = Node(state=board.clone())
        
        else:
            self.mcts.root = Node(state=board.clone())
        


        # search for new root has to see if it has been expanded already
        # Run MCTS search
        return self.mcts.search(self.time_budget)
    
    @cython.ccall
    def select_move_for_passed_board(self, object board):
        """Select a move for a board that's already in a passed state.
        Choose the move that minimizes the player's pip count."""
        cdef list best_move = None
        cdef int min_pips = 999999 
        cdef int pips
        cdef list move
        cdef object test_board
        
        for move in board.valid_moves:
            # Create a copy and apply the move
            test_board = board.clone()
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