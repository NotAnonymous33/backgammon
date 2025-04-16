from copy import deepcopy
import time
import random
import math
import cython
from .HeuristicAgent import HeuristicAgent


cdef class Node:
    cdef public object state
    cdef public list move_sequence
    cdef public object parent
    cdef public dict children
    cdef public int N
    cdef public float Q
    cdef public list untried_moves
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
            if not state.rolled:
                state.roll_dice()
            self.untried_moves = state.valid_moves
        else:
            self.untried_moves = []

    @cython.ccall
    @cython.cdivision(True)
    def value(self, float exploration_weight):
        """Calculate the UCT value of this node."""
        if self.N == 0:
            return float('inf')

        cdef float exploitation = self.Q / self.N
        cdef float exploration = exploration_weight * math.sqrt(2 * math.log(self.parent.N) / self.N)

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
    cdef public object root
    cdef public int player_color
    cdef public int sim_count
    cdef public object heuristic_agent

    def __init__(self, float exploration_weight=1.0, simulation_depth=50):
        """Initialize the MCTS agent."""
        self.exploration_weight = exploration_weight
        self.simulation_depth = simulation_depth
        self.root = None
        self.player_color = 0
        self.sim_count = 0
        self.heuristic_agent = HeuristicAgent()

    @cython.ccall
    def search(self, float time_budget):
        cdef object node
        cdef list move
        cdef object new_state
        cdef object child
        cdef tuple move_key
        cdef float result
        start_time = time.time()
        self.sim_count = 0
        """Run MCTS for the specified time."""

        while time.time() - start_time < time_budget:
            # 1. Selection: traverse tree until we reach a leaf node
            node = self.select_node()

            # 2. Expansion: add a child if not terminal and not fully expanded
            if not node.is_fully_expanded() and not node.state.game_over:
                # Choose an untried move
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)

                # Create a new state by applying the move
                new_state = deepcopy(node.state)
                new_state.move_from_sequence(move)

                # Create a new child node
                child = Node(state=new_state, move_sequence=move, parent=node)
                node.children[tuple(move)] = child

                # Use the new node for simulation
                node = child

            # 3. Simulation: play out to terminal state or depth limit
            result = self.simulate(node)

            # 4. Backpropagation: update statistics up the tree
            self.backpropagate(node, result)

            self.sim_count += 1

        # print(f"Performed {num_simulations} simulations in {time.time() - start_time:.2f} seconds")
        # print(f"{self.sim_count=}")

    @cython.ccall
    def select_node(self):
        """Select a node to expand using UCT."""
        cdef object node = self.root

        # Keep selecting best child until reaching a leaf or unexpanded node
        while node.children and node.is_fully_expanded() and not node.state.game_over:
            node = node.best_child(self.exploration_weight)

        return node

    @cython.ccall
    @cython.cdivision(True)
    def simulate(self, object node):
        """Simulate a random game from node and return the result."""
        cdef list move
        cdef object state = deepcopy(node.state)
        cdef int depth = 0

        # Simulate random play until terminal state or depth limit
        while not state.game_over and depth < self.simulation_depth:
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

        # Return result or heuristic evaluation
        if state.game_over:
            if (self.player_color == 1 and state.white_off == 15) or (self.player_color == -1 and state.black_off == 15):
                return 1
            return -1
        else:
            if self.player_color == 1:
                return 2 * (state.black_left / (state.white_left + state.black_left)) - 1
            return 2 * (state.white_left / (state.white_left + state.black_left)) - 1

    @cython.ccall
    def backpropagate(self, object node, float result):
        """Update statistics in all nodes along path from node to root."""
        cdef object current = node
        cdef float current_result = result

        while current is not None:
            current.N += 1
            current.Q += current_result
            current = current.parent
            current_result = -current_result  # Negate result for parent (other player)

    # @cython.ccall
    def best_move(self):
        """Return move with highest visit count from root's children."""
        if not self.root.children:
            return []

        max_visits = max(child.N for child in self.root.children.values())
        best_children = [child for child in self.root.children.values() if child.N == max_visits]

        # choose random child
        return random.choice(best_children).move_sequence


cdef class MCTSAgent:
    cdef public object mcts
    cdef public float time_budget
    def __init__(self, float exploration_weight=1.0, int simulation_depth=50, float time_budget=2.0):
        self.mcts = MCTSBackgammonAgent(
            exploration_weight=exploration_weight,
            simulation_depth=simulation_depth
        )
        self.time_budget = time_budget

    @cython.ccall
    @cython.boundscheck(False)
    def select_move(self, object board):
        if not board.valid_moves:
            return []

        # If we only have one valid move, no need to run MCTS
        if len(board.valid_moves) == 1:
            return board.valid_moves[0]

        # Use MCTS to find the best move
        self.mcts.player_color = board.turn
        self.mcts.root = Node(state=deepcopy(board))

        self.mcts.search(self.time_budget)

        return self.mcts.best_move()
