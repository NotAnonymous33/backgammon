import time
import random
import math
import cython
from .HeuristicAgent import HeuristicAgent

# Leaf evaluations:
#   neural     the value network's P(win) of the leaf (default)
#   heuristic  HeuristicAgent's score of the leaf, squashed with tanh
#   rollout    random play, then the pip-count ratio (the original "default")
MODES = ("neural", "heuristic", "rollout")

# UCB1 exploration constant per mode, for values that are win probabilities
# in [0, 1]. Move values differ by a few points of winning chance, so c must
# be small for visits to follow them. Tuned on the bench skill positions at
# 400 and 1,600 iterations: the network's evaluations are smooth, so it needs
# far less exploration than the noisy rollout and heuristic leaves.
DEFAULT_EXPLORATION = {"neural": 0.01, "heuristic": 0.2, "rollout": 0.2}

# One entry per ordered outcome of two dice, so a uniform pick has the right
# odds: 1/36 for each double and 2/36 for each non-double. Higher die first.
ROLLS = tuple((max(a, b), min(a, b)) for a in range(1, 7) for b in range(1, 7))
PASS = "pass"  # child key when the roll has no legal play


@cython.ccall
def afterstate_key(object state):
    """Identifies a resulting position, so move orders reaching it share a child."""
    return (tuple(state.positions), state.white_bar, state.black_bar,
            state.white_off, state.black_off, state.turn)


cdef class Node:
    """A position with ``state.turn`` about to roll; the root has already rolled.

    W sums the mover's winning chance over the N backups through this node,
    where the mover is the player whose play led here. A parent therefore picks
    the child with the highest mean W / N, whichever side is choosing.
    """
    cdef public object state
    cdef public list move_sequence
    cdef public int mover
    cdef public int N
    cdef public double W
    cdef public dict rolls  # (high, low) -> Roll

    def __init__(self, state, move_sequence, int mover):
        self.state = state
        self.move_sequence = move_sequence
        self.mover = mover
        self.N = 0
        self.W = 0.0
        self.rolls = {}


cdef class Roll:
    """The decision at a node once its roll is known.

    Dice are drawn afresh on every visit to a node, so each roll a node has met
    gets its own Roll, and the node's value averages over the dice instead of
    depending on one roll fixed when the node was made.
    """
    cdef public object state      # the node's board with these dice rolled
    cdef public dict children     # afterstate_key -> Node
    cdef public list untried      # plays not yet expanded
    cdef public int N             # visits with this roll

    def __init__(self, state):
        self.state = state
        self.children = {}
        self.untried = list(state.valid_moves)
        self.N = 0


cdef class MCTSBackgammonAgent:
    cdef public float exploration_weight
    cdef public int simulation_depth
    cdef public int eval_depth
    cdef public str mode
    cdef public object heuristic_agent
    cdef public object nn_agent
    cdef public object root
    cdef public object root_roll
    cdef public int player_color
    cdef public int sim_count
    cdef public int max_iterations
    cdef public list probability_table
    cdef public list ratio_table

    def __init__(self,
                 float exploration_weight=-1.0,
                 int simulation_depth=50,
                 mode="neural",
                 int eval_depth=-1,
                 heuristic_agent=None,
                 nn_agent=None,
                 int max_iterations=0):
        """Initialize the MCTS agent with a leaf evaluation mode.

        eval_depth is the number of random plies played from a leaf before it
        is evaluated; 0 evaluates the leaf itself. Below 0 it defaults to
        simulation_depth for "rollout" and to 0 otherwise. A negative
        exploration_weight uses the mode's DEFAULT_EXPLORATION.

        max_iterations > 0 also stops each search after that many iterations,
        which makes searches reproducible and independent of machine speed.
        """
        if mode == "default":  # the original name for random rollouts
            mode = "rollout"
        if mode not in MODES:
            raise ValueError(f"unknown MCTS mode {mode!r}; expected one of {', '.join(MODES)}")
        self.mode = mode
        self.exploration_weight = exploration_weight if exploration_weight >= 0 else DEFAULT_EXPLORATION[mode]
        self.max_iterations = max_iterations
        self.simulation_depth = simulation_depth
        if eval_depth >= 0:
            self.eval_depth = eval_depth
        else:
            self.eval_depth = simulation_depth if mode == "rollout" else 0
        # allow injection of your own agents, else build defaults
        if mode == "heuristic":
            self.heuristic_agent = heuristic_agent or HeuristicAgent()
        else:
            self.heuristic_agent = None

        if mode == "neural":
            if nn_agent is None:
                from .NNAgent import FinalNNAgent  # torch is only needed here
                nn_agent = FinalNNAgent()
            self.nn_agent = nn_agent
        else:
            self.nn_agent = None

        self.root = None
        self.root_roll = None
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

    def set_root(self, object board):
        """Start a fresh tree at ``board``, whose dice are already rolled."""
        self.player_color = board.turn
        self.root = Node(board.clone(), None, -board.turn)
        self.root_roll = Roll(board.clone())
        self.root.rolls[tuple(sorted(board.dice[:2], reverse=True))] = self.root_roll

    @cython.ccall
    def search(self, float time_budget):
        start_time = time.time()
        self.sim_count = 0
        """Run MCTS for the specified time."""

        while time.time() - start_time < time_budget and \
                (self.max_iterations <= 0 or self.sim_count < self.max_iterations):
            self.iterate()
            self.sim_count += 1
        return self.best_move()

    @cython.ccall
    def iterate(self):
        """One selection, expansion, evaluation and backup."""
        cdef object node = self.root
        cdef list path = [node]
        cdef object roll
        cdef object child
        cdef object state
        cdef double value

        # 1. Selection, drawing the dice at every node on the way down
        while node is self.root or not (node.state.game_over or node.state.passed):
            roll = self.root_roll if node is self.root else self.roll_at(node)
            roll.N += 1
            if not roll.state.valid_moves:
                # No legal play: the turn passes to the other side.
                child = roll.children.get(PASS)
                if child is None:
                    state = roll.state.clone()
                    state.swap_turn()
                    child = Node(state, [], roll.state.turn)
                    roll.children[PASS] = child
                    path.append(child)
                    break
                node = child
                path.append(node)
                continue

            # 2. Expansion: add one new afterstate while any remain
            child = self.expand(roll)
            if child is not None:
                path.append(child)
                break
            node = self.best_child(roll)
            path.append(node)

        # 3. Evaluation, as the searching player's winning chance
        value = self.evaluate(path[-1].state)

        # 4. Backup: each node scores the result for the player who moved into it
        for node in path:
            node.N += 1
            node.W += value if node.mover == self.player_color else 1.0 - value

    @cython.ccall
    def roll_at(self, object node):
        """Draw dice for ``node`` and return the decision they give."""
        cdef tuple dice = ROLLS[random.randrange(36)]
        cdef object roll = node.rolls.get(dice)
        cdef object state
        if roll is None:
            state = node.state.clone()
            state.set_dice([dice[0]] * 4 if dice[0] == dice[1] else list(dice))
            roll = Roll(state)
            node.rolls[dice] = roll
        return roll

    @cython.ccall
    def expand(self, object roll):
        """Add a child for an untried play reaching a new position, or return None."""
        cdef list untried = roll.untried
        cdef int i
        cdef list move
        cdef object state
        cdef object key
        cdef object child
        while untried:
            i = random.randrange(len(untried))
            move = untried[i]
            untried[i] = untried[-1]
            untried.pop()
            state = roll.state.clone()
            state.move_from_sequence(move)
            key = afterstate_key(state)
            if key in roll.children:
                continue  # another order of the same checker moves
            child = Node(state, move, roll.state.turn)
            roll.children[key] = child
            return child
        return None

    @cython.ccall
    @cython.cdivision(True)
    def best_child(self, object roll):
        """The child with the highest UCB1 value for the player choosing."""
        cdef double log_n = math.log(max(1, roll.N))
        cdef double c = self.exploration_weight
        cdef double best_value = -1e300
        cdef double value
        cdef list best_children = []
        cdef object child
        for child in roll.children.values():
            value = child.W / child.N + c * math.sqrt(log_n / child.N)
            if value > best_value:
                best_value = value
                best_children = [child]
            elif value == best_value:
                best_children.append(child)
        return random.choice(best_children)

    @cython.ccall
    def terminal_value(self, object state):
        if (self.player_color == 1 and state.white_off == 15) \
           or (self.player_color == -1 and state.black_off == 15):
            return 1.0
        return 0.0

    @cython.ccall
    @cython.cdivision(True)
    def evaluate(self, object leaf):
        """The searching player's winning chance at ``leaf``, in [0, 1]."""
        cdef object state = leaf
        cdef int depth = 0
        cdef double score

        if state.game_over:
            return self.terminal_value(state)

        # Optional random plies before evaluating (always for "rollout")
        if self.eval_depth > 0 and not state.passed:
            state = leaf.clone()
            while not state.game_over and not state.passed and depth < self.eval_depth:
                if not state.rolled:
                    state.roll_dice()
                valid_moves = state.valid_moves
                if not valid_moves:
                    state.swap_turn()
                    continue
                state.move_from_sequence(random.choice(valid_moves))
                depth += 1
            if state.game_over:
                return self.terminal_value(state)

        if self.mode == "neural":
            # The network's chance that the searching player wins, whatever its
            # feature schema (white-34 or canonical).
            return self.nn_agent.win_probability(state, self.player_color)

        if state.passed:
            return self.calculate_d2s_win_probability(state)

        if self.mode == "heuristic":
            # The heuristic scores the board for white.
            score = self.heuristic_agent.evaluate_board(state)
            if self.player_color == -1:
                score = -score
            return (math.tanh(score) + 1.0) / 2.0

        # rollout: pip-count ratio
        if self.player_color == 1:
            return state.black_left / (state.white_left + state.black_left)
        return state.white_left / (state.white_left + state.black_left)

    @cython.ccall
    @cython.cdivision(True)
    def calculate_d2s_win_probability(self, object state):
        """The searching player's winning chance on a passed board, by the adjusted D²/S method."""
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

        if white_is_ahead == (self.player_color == 1):
            return win_prob_ahead
        return 1 - win_prob_ahead

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

    def best_move(self):
        """Return move with highest visit count from root's children."""
        if not self.root_roll.children:
            return []

        children = list(self.root_roll.children.values())
        max_visits = max(child.N for child in children)
        best_children = [child for child in children if child.N == max_visits]
        return random.choice(best_children).move_sequence



cdef class MCTSAgent2:
    cdef public object mcts
    cdef public float time_budget

    def __init__(self,
                 float exploration_weight=-1.0,
                 int simulation_depth=50,
                 mode="neural",
                 int eval_depth=-1,
                 float time_budget=2.0,
                 heuristic_agent=None,
                 nn_agent=None,
                 int max_iterations=0):
        """High‐level wrapper: pass mode and eval_depth through."""
        self.mcts = MCTSBackgammonAgent(
            exploration_weight=exploration_weight,
            simulation_depth=simulation_depth,
            mode=mode,
            eval_depth=eval_depth,
            heuristic_agent=heuristic_agent,
            nn_agent=nn_agent,
            max_iterations=max_iterations
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

        # Each decision searches a fresh tree. Children are reached through
        # sampled dice, so a tree kept from the previous move rarely contains
        # the position actually reached.
        self.mcts.set_root(board)
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
