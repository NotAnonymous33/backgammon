from copy import deepcopy
import time
import random
import math
try:
    from Board import Board
    from Color import Color
except:
    from ..Board import Board
    from ..Color import Color

class BackgammonState:
    def __init__(self, board=None):
        """Initialize the backgammon state with a board."""
        self.board = deepcopy(board) if board else Board()
    
    def clone(self):
        """Create a deep copy of this state."""
        return BackgammonState(self.board)
    
    def apply_move(self, move_sequence):
        """Apply a sequence of moves to this state."""
        return self.board.move_from_sequence(move_sequence)
    
    def get_valid_moves(self):
        """Return all valid move sequences for the current state."""
        if not self.board.rolled:
            self.board.roll_dice()
        return self.board.valid_moves
    
    def is_terminal(self):
        """Check if the game is over."""
        return self.board.game_over
    
    def get_result(self, player_color):
        """Return the result (1 for win, -1 for loss) from perspective of player_color."""
        if not self.is_terminal():
            return 0
        

        if player_color == Color.WHITE:
            return 1 if self.board.white_off == 15 else -1
        else:  # player_color == Color.BLACK
            return 1 if self.board.black_off == 15 else -1


class Node:
    def __init__(self, state=None, move_sequence=None, parent=None):
        """Initialize a node in the MCTS tree."""
        self.state = state
        self.move_sequence = move_sequence  # The move that led to this state
        self.parent = parent
        self.children = {}  # Map move sequences to child nodes
        self.N = 0  # Visit count
        self.Q = 0  # Total reward
        
        # Get valid moves if state is provided
        if state:
            self.untried_moves = state.get_valid_moves()
        else:
            self.untried_moves = []
    
    def value(self, exploration_weight):
        """Calculate the UCT value of this node."""
        if self.N == 0:
            return float('inf')
        
        exploitation = self.Q / self.N
        exploration = exploration_weight * math.sqrt(2 * math.log(self.parent.N) / self.N)
        
        return exploitation + exploration
    
    def is_fully_expanded(self):
        """Check if all valid moves have been expanded."""
        return len(self.untried_moves) == 0
    
    def best_child(self, exploration_weight):
        """Return the child with the highest UCT value."""
        if not self.children:
            return None
        
        # Find child with highest UCT value
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
        self.player_color = None
        self.sim_count = 0
    
    # def select_move(self, board):
    #     """Select the best move using MCTS given the current board state."""
    #     if len(board.valid_moves) == 0:
    #         return []
        
    #     # Initialize root with current board state
    #     self.player_color = board.turn
    #     self.root = Node(state=BackgammonState(board))
        
    #     # Run search with time budget
    #     time_budget = 2.0  # 2 seconds
    #     self.search(time_budget)
        
    #     # Return best move
    #     return self.best_move()
    
    def search(self, time_budget):
        """Run MCTS for the specified time."""
        start_time = time.time()
        
        while time.time() - start_time < time_budget:
            # 1. Selection: traverse tree until we reach a leaf node
            node = self.select_node()
            
            # 2. Expansion: add a child if not terminal and not fully expanded
            if not node.is_fully_expanded() and not node.state.is_terminal():
                # Choose an untried move
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                
                # Create a new state by applying the move
                new_state = node.state.clone()
                new_state.apply_move(move)
                
                # Create a new child node
                move_key = tuple(tuple(m) for m in move)
                child = Node(state=new_state, move_sequence=move, parent=node)
                node.children[move_key] = child
                
                # Use the new node for simulation
                node = child
            
            # 3. Simulation: play out to terminal state or depth limit
            result = self.simulate(node)
            
            # 4. Backpropagation: update statistics up the tree
            self.backpropagate(node, result)
            
            self.sim_count += 1
        
        # print(f"Performed {num_simulations} simulations in {time.time() - start_time:.2f} seconds")
        print(f"{self.sim_count=}")
        
    
    def select_node(self):
        """Select a node to expand using UCT."""
        node = self.root
        
        # Keep selecting best child until reaching a leaf or unexpanded node
        while node.children and node.is_fully_expanded() and not node.state.is_terminal():
            node = node.best_child(self.exploration_weight)
        
        return node
    
    def simulate(self, node):
        """Simulate a random game from node and return the result."""
        state = node.state.clone()
        depth = 0
        
        # Simulate random play until terminal state or depth limit
        while not state.is_terminal() and depth < self.simulation_depth:
            # Get valid moves
            valid_moves = state.get_valid_moves()
            
            # If no valid moves, roll dice or swap turn
            if not valid_moves:
                if state.board.rolled:
                    state.board.swap_turn()
                state.board.roll_dice()
                continue
            
            # Choose a random move and apply it
            move = random.choice(valid_moves)
            state.apply_move(move)
            depth += 1
        
        # Return result or heuristic evaluation
        if state.is_terminal():
            return state.get_result(self.player_color)
        else:
            if self.player_color == Color.WHITE:
                return 2 * (state.board.black_left / (state.board.white_left + state.board.black_left) - 1)
            return 2 * (state.board.white_left / (state.board.white_left + state.board.black_left) - 1)
            # # Heuristic: compare pieces borne off and on bar
            # if self.player_color == Color.WHITE:
            #     relative_progress = state.board.white_left - state.board.black_left
            # else:
            #     relative_progress = state.board.black_left - state.board.white_left
            
            
            # white_progress = state.board.white_off - state.board.white_bar
            # black_progress = state.board.black_off - state.board.black_bar
            # if self.player_color == Color.WHITE:
            #     relative_progress = white_progress - black_progress
            # else:
            #     relative_progress = black_progress - white_progress
            
            # # Normalize to [-1, 1]
            # return max(-1, min(1, relative_progress / 15))
    
    def backpropagate(self, node, result):
        """Update statistics in all nodes along path from node to root."""
        current = node
        current_result = result
        
        while current is not None:
            current.N += 1
            current.Q += current_result
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


class BackgammonMCTSAgent:
    def __init__(self, exploration_weight=1.0, simulation_depth=50, time_budget=2.0):
        self.mcts = MCTSBackgammonAgent(
            exploration_weight=exploration_weight,
            simulation_depth=simulation_depth
        )
        self.time_budget = time_budget
    
    def select_move(self, board):
        if not board.valid_moves:
            return []
        
        # If we only have one valid move, no need to run MCTS
        if len(board.valid_moves) == 1:
            return board.valid_moves[0]
        
        # Use MCTS to find the best move
        self.mcts.player_color = board.turn
        self.mcts.root = Node(state=BackgammonState(board))
        
        self.mcts.search(self.time_budget)
        
        return self.mcts.best_move()

# TODO: try different exploration weights and simulation depths
# could do self play to find best params