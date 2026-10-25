try:
    from classes.board_cpp import Board # type: ignore
    from classes.agents.RandomAgent import RandomAgent
    from classes.agents.HeuristicAgent import HeuristicAgent
    from classes.agents import features as feature_schemas
except ImportError:
    
    from board_cpp import Board # type: ignore
    from .RandomAgent import RandomAgent
    from .HeuristicAgent import HeuristicAgent
    from . import features as feature_schemas
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import random
from copy import deepcopy
import os
from multiprocessing import Pool, cpu_count
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.set_float32_matmul_precision('medium')

def extract_features(board):
    """The original ``white-34`` features (see features.py), kept as the reference implementation.

    New networks use ``features.CANONICAL``, which takes the side to roll's point of view.

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
        layers.append(nn.Sigmoid()) 
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class TDLambda:
    def __init__(self, model, learning_rate=0.1, lambda_param=0.7, gamma=1.0):
        """
        Online TD(λ) with eligibility traces for a value network.

        Value convention (``features.CANONICAL``): V(s) is the probability that
        the side about to roll in s wins, so targets lie in [0, 1]:

        * between turns, the target for V(s_t) is 1 - V(s_{t+1}) when the next
          state belongs to the opponent (it always does in backgammon, even
          after a pass), or V(s_{t+1}) for the same player;
        * the last state of a finished game gets the outcome itself: 1 if its
          player won (the side that bears off last always wins on its own
          turn), else 0;
        * the last state of a game cut off by the move limit gets no target,
          because nobody won it.

        Because the point of view flips each turn, the eligibility trace flips
        sign with it (see ``update``); this is exactly TD(λ) on the white-win
        probability, written in the network's own perspective.

        Args:
            model: Neural network model.
            learning_rate: Step size for parameter updates.
            lambda_param: Decay rate for eligibility traces.
            gamma: Discount factor. Backgammon has no intermediate rewards and
                the value is a probability, so the default is 1 (undiscounted).
        """
        self.model = model
        self.initial_learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.gamma = gamma
        # Using SGD because we update parameters manually with eligibility traces.
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        # Initialize eligibility traces for each parameter.
        self.eligibility = {}
        for name, param in self.model.named_parameters():
            self.eligibility[name] = torch.zeros_like(param, dtype=torch.float32)
    
    def update(self, states, players, winner, epoch_num=0):
        """
        Update model parameters from one self-play game.

        Args:
            states: Feature rows, one per turn, each from the point of view of
                the side about to roll (array-like of shape [turns, inputs]).
            players: The side to roll for each state (1 white, -1 black).
            winner: 1 or -1, or None if the game hit the move limit.
            epoch_num: Epoch number, for learning-rate decay.

        In white's fixed frame, W_t = V_t when white is on roll and 1 - V_t
        otherwise, and ordinary TD(λ) applies to W. Substituting gives the
        update below: the TD error is target - V_t and the trace is decayed
        by +γλ when consecutive states belong to the same player and -γλ
        when the player changes.
        """
        x = torch.as_tensor(np.asarray(states, dtype=np.float32)).reshape(len(players), -1)
        n = len(players)
        if n == 0:
            return
        lr = self.initial_learning_rate * 0.999 ** epoch_num
        self.model.train()
        params = [(name, p) for name, p in self.model.named_parameters()]
        for name, _ in params:
            self.eligibility[name].zero_()

        for t in range(n):
            if t == n - 1:
                if winner is None:
                    break  # cut off by the move limit: no outcome to learn from
                target = 1.0 if winner == players[t] else 0.0
            else:
                with torch.no_grad():
                    v_next = self.model(x[t + 1:t + 2]).item()
                target = v_next if players[t + 1] == players[t] else 1.0 - v_next

            self.optimizer.zero_grad()
            v = self.model(x[t:t + 1])
            v.backward()
            delta = target - v.item()
            decay = self.gamma * self.lambda_param
            if t > 0 and players[t] != players[t - 1]:
                decay = -decay
            with torch.no_grad():
                for name, param in params:
                    if param.grad is None:
                        continue
                    trace = self.eligibility[name]
                    trace.mul_(decay).add_(param.grad)
                    param.add_(trace, alpha=lr * delta)
        self.optimizer.zero_grad()


def as_schema(features_or_schema):
    """A feature schema, accepting the legacy ``extract_features`` function for old call sites."""
    if features_or_schema is None:
        return feature_schemas.DEFAULT
    if isinstance(features_or_schema, feature_schemas.Schema):
        return features_or_schema
    if features_or_schema is extract_features:
        return feature_schemas.LEGACY
    raise TypeError("pass a features.Schema (e.g. features.CANONICAL), or extract_features for white-34 networks")


Episode = namedtuple("Episode", ["states", "players", "winner"])
Episode.__doc__ = """One self-play game: per-turn feature rows, who was on roll, and the winner (None if cut off)."""


class NNAgent:
    def __init__(self, model, schema=None, exploration_rate=0.1):
        """
        Neural network-based agent for backgammon, as used in training.
        
        Args:
            model: Trained neural network model
            schema: Feature schema the model was trained with (default
                features.CANONICAL); ``extract_features`` means white-34
            exploration_rate: Rate of random exploration (0.0 to 1.0)
        """
        self.model = model
        self.schema = as_schema(schema)
        self.extract_features = self.schema.features
        self.exploration_rate = exploration_rate
        
        
    def select_move(self, board):
        if not board.valid_moves:
            return []
            
        # Occasionally make a random move for exploration
        if random.random() < self.exploration_rate:
            return random.choice(board.valid_moves)
        
        moves = board.valid_moves
        children = [board.clone() for _ in moves]
        for child, move in zip(children, moves):
            child.move_from_sequence(move)
        
        rows = self.schema.afterstate_rows(board.turn, children)
        with torch.no_grad():
            self.model.eval()
            values = self.model(torch.tensor(rows, dtype=torch.float32)).reshape(-1).tolist()
        return moves[self.schema.best_index(board.turn, children, values)]


class BackgammonTrainer:
    def __init__(self, model, schema, td_lambda, games_per_epoch=1000, eval_games=100, num_workers=None):
        """
        Training pipeline for backgammon AI.
        
        Args:
            model: Neural network model
            schema: Feature schema to train with; only features.CANONICAL
                matches the value convention TDLambda trains towards
            td_lambda: TD(λ) learning implementation
            games_per_epoch: Number of games to play per training epoch
            eval_games: Number of games to evaluate the model
            num_workers: Number of workers for data loading (optional)
        """
        self.model = model
        self.schema = as_schema(schema)
        if not self.schema.canonical:
            raise ValueError("TDLambda trains P(side to roll wins); use features.CANONICAL")
        self.extract_features = self.schema.features
        self.td_lambda = td_lambda
        self.games_per_epoch = games_per_epoch
        self.eval_games = eval_games
        self.num_workers = num_workers or cpu_count()
    
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
            'training_results': getattr(self, 'training_results', []),
            # Self-describing: loaders need these to use the network correctly.
            'feature_schema': self.schema.describe(),
            'architecture': {'input_size': self.schema.size, 'hidden_sizes': hidden_sizes_of(self.model.state_dict())},
        }
        
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        tmp = filename + ".tmp"
        torch.save(checkpoint, tmp)
        os.replace(tmp, filename)  # never leave a half-written checkpoint behind
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
            schema = feature_schemas.from_description(checkpoint.get('feature_schema'))
            if schema is not self.schema:
                raise feature_schemas.SchemaError(
                    f"{filename} was trained with {schema.name} v{schema.version}, not {self.schema.name} v{self.schema.version}")
            
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
    
    def simulate_one_game(self, idx):
        return self.play_game()
    
    def train_epoch(self, epoch_num=0):
        """Simulate games in parallel and update as each one completes."""
        wins_white = 0

        # Spawn workers
        with Pool(self.num_workers) as pool:
            # pool.imap_unordered returns results as soon as they're ready
            game_results = pool.imap_unordered(
                self.simulate_one_game,
                range(self.games_per_epoch)
            )

            # Iterate and update immediately
            for episode in tqdm(
                game_results,
                total=self.games_per_epoch,
                desc=f"Epoch {epoch_num+1} rollout & learn"
            ):
                # on‑the‑fly TD‑λ update
                self.td_lambda.update(episode.states, episode.players, episode.winner, epoch_num)
                if episode.winner == 1:
                    wins_white += 1

        return wins_white / self.games_per_epoch
    
    def play_game(self):
        """
        Play a complete game of backgammon using the current model.
        
        Returns:
            Episode(states, players, winner): one feature row per turn, from
            the point of view of the side about to roll, as a float32 array
            (cheap to send back from worker processes); who was on roll at
            each turn; and the winner (1 white, -1 black), or None if the game
            reached the move limit.
        """
        board = Board()
        agent = NNAgent(self.model, self.schema)
        
        states = []
        players = []
        
        move_count = 0
        max_moves = 200  # Prevent infinite games
        
        while not board.game_over and move_count < max_moves:
            # Roll dice
            board.roll_dice()         
    
            # Record the state from the roller's point of view
            states.append(self.schema.features(board))
            players.append(board.turn)
            # Choose move
            move = agent.select_move(board)
            
            # Make move
            board.move_from_sequence(move)
            move_count += 1
        
        winner = (1 if board.white_off == 15 else -1) if board.game_over else None
        return Episode(np.asarray(states, dtype=np.float32).reshape(len(players), self.schema.size), players, winner)
    
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
        for epoch in tqdm(range(start_epoch, start_epoch + num_epochs)):
            print(f"\nEpoch {start_epoch + epoch + 1}/{start_epoch + num_epochs}")
            win_rate = self.train_epoch(start_epoch + epoch)
            
            results.append(win_rate)
            self.last_win_rate = win_rate
            self.training_results = results
            
            print(f"Epoch {epoch+1}/{start_epoch + num_epochs}, Win Rate: {win_rate:.4f}")
            
            # Save checkpoint at regular intervals
            if (epoch + 1) % checkpoint_interval == 0:
                if name != "":
                    self.save_checkpoint(f"models/{name}/checkpoint_epoch_{name}{epoch+1}.pt", epoch=epoch+1)
                else:
                    self.save_checkpoint(f"models/checkpoint_epoch_{epoch+1}.pt", epoch=epoch+1)
            
            e = Evaluator(NNAgent(self.model, self.schema, exploration_rate=0.0), opponent_agent=HeuristicAgent(), num_games=self.eval_games)
            eval_results = e.evaluate()                

            print("\nEvaluation Results:")
            for key, value in eval_results.items():
                print(f"{key}: {value}")
            if not os.path.exists("eval_results"):
                os.makedirs("eval_results")
            with open(f"eval_results/results_{name}.txt", "a") as f:
                f.write(f"{epoch}\t{eval_results['white_win_rate']}\t{eval_results['black_win_rate']}\t{eval_results['win_rate']}\n")
            
            # Always save latest model
            self.plot_learning_curve(save_path=f"learning_curve_{name}.png")
            # if models does not exist create
            if not os.path.exists("models"):
                os.makedirs("models")
            if name != "":
                if not os.path.exists("models/" + name):
                    os.makedirs("models/" + name)
                self.save_checkpoint(f"models/{name}/{name}_checkpoint_latest.pt", epoch=epoch+1)
                self.save_model(f"models/{name}/{name}_model_latest.pt")
            else:
                self.save_checkpoint("models/latest.pt", epoch=epoch+1)
                self.save_model("models/latest.pt")

        
        self.plot_eval_curve(f"eval_results/results_{name}.txt", f"graph_{name}.png")
        
        # Save final model
        if name != "":
            # create dir if it does not exist
            if not os.path.exists("models/" + name):
                os.makedirs("models/" + name)
            self.save_model(f"{name}/{name}_final.pt")
        else:
            self.save_model(f"final_model.pt")
        
        return results

    def plot_eval_curve(self, txt_path, save_path=None):
        with open(txt_path, "r") as f:
            lines = f.readlines()
            epochs = [int(line.split("\t")[0]) for line in lines]
            white_win_rates = [float(line.split("\t")[1]) for line in lines]
            black_win_rates = [float(line.split("\t")[2]) for line in lines]
            win_rates = [float(line.split("\t")[3]) for line in lines]
        
        plt.style.use('seaborn-v0_8-talk')
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, white_win_rates, '-o', color='#1f77b4', alpha=0.6, label='White Win Rate') 
        plt.plot(epochs, black_win_rates, '-o', color='#d62728', alpha=0.6, label='Black Win Rate') 
        plt.plot(epochs, win_rates, '-o', color='#2ca02c', label='Total Win Rate')   
        
        plt.ylim(0, 1.1)
        plt.xlabel('Epoch')
        plt.ylabel('Win Rate vs Heuristic Agent')
        plt.title(f'Evaluation Progress / Win Rate of {txt_path[21:-4]} against Heuristic Agent')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(f"eval_results/{save_path}")
            print(f"Evaluation curve saved to {save_path}")
        plt.close()
    
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

        
        if save_path:
            plt.savefig(f"learning_curves/{save_path}")
            print(f"Learning curve saved to {save_path}")
        plt.close()

        
    def save_model(self, filename):
        """Save the model to a file."""
        # create file if it does not exist
        with open(filename, 'w') as f:
            pass
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
    def __init__(self, nn_agent, opponent_agent=None, num_games=100, num_workers=None, move_limit=200):
        """
        Evaluation framework for backgammon AI.
        
        Args:
            nn_agent: Neural network agent to evaluate
            opponent_agent: Agent to play against (default: RandomAgent)
            num_games: Number of games to play for evaluation
            num_workers: How many parallel processes to spawn (defaults to cpu_count())
            move_limit: Max turns per game
        """
        self.nn_agent = nn_agent
        self.opponent_agent = opponent_agent or RandomAgent()
        self.num_games = num_games
        self.move_limit = move_limit
        self.num_workers = num_workers or cpu_count()

    def _play_one(self, idx):
        """
        Play a single game.
        idx < half → NN plays as white.
        idx ≥ half → NN plays as black.
        
        Returns:
          (win_flag:int, turn_count:int, is_white:bool)
        """
        is_white = idx < (self.num_games // 2)
        board = Board()
        turn_count = 0

        while not board.game_over and turn_count < self.move_limit:
            board.roll_dice()
            if (board.turn == 1) == is_white:
                move = self.nn_agent.select_move(board)
            else:
                move = self.opponent_agent.select_move(board)
            board.move_from_sequence(move)
            turn_count += 1

        # Determine winner
        if turn_count >= self.move_limit:
            # tie‐break by borne‐off pieces
            if board.white_off == board.black_off:
                winner = 1 if board.white_left < board.black_left else -1
            else:
                winner = 1 if board.white_off > board.black_off else -1
        else:
            winner = 1 if board.white_off == 15 else -1

        # Convert to win flag for NN
        if is_white:
            win_flag = 1 if winner == 1 else 0
        else:
            win_flag = 1 if winner == -1 else 0

        return win_flag, turn_count, is_white

    def evaluate(self):
        """
        Evaluate the neural network agent against the opponent in parallel.
        
        Returns:
            Dictionary of evaluation metrics
        """
        half = self.num_games // 2
        wins_white = wins_black = 0
        white_turns = []
        black_turns = []

        with Pool(self.num_workers) as pool:
            # imap gives results as they complete; tqdm for progress
            for win_flag, turns, is_white in tqdm(
                pool.imap(self._play_one, range(self.num_games)),
                total=self.num_games,
                desc="Evaluating"
            ):
                if is_white:
                    white_turns.append(turns)
                    wins_white += win_flag
                else:
                    black_turns.append(turns)
                    wins_black += win_flag

        total_wins = wins_white + wins_black
        return {
            "total_games": self.num_games,
            "total_wins": total_wins,
            "win_rate": total_wins / self.num_games,
            "white_wins": wins_white,
            "black_wins": wins_black,
            "white_win_rate": wins_white / half if half else 0,
            "black_win_rate": wins_black / half if half else 0,
            "avg_turns_as_white": sum(white_turns) / len(white_turns) if white_turns else 0,
            "avg_turns_as_black": sum(black_turns) / len(black_turns) if black_turns else 0
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
            if board.white_off > board.black_off:
                return 1
            if board.black_off > board.white_off:
                return -1
            return 1 if board.white_left < board.black_left else -1
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

def hidden_sizes_of(state_dict):
    """Hidden layer widths of a BackgammonNN state dict (Linear layers sit at model.0, model.2, ...)."""
    weights = sorted((int(k.split(".")[1]), v) for k, v in state_dict.items() if k.endswith(".weight"))
    return [int(w.shape[0]) for _, w in weights[:-1]]


# The network the server, neural MCTS and the benchmark use unless told otherwise.
DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                  "models", "canonical", "canonical_128x128_100k.pt")


def load_network(path):
    """(model in eval mode, feature schema) for a checkpoint or bare state dict.

    The schema comes from the checkpoint; files saved before schemas existed
    are ``white-34``. Loading is strict: a partial load would leave layers at
    random initialisation.
    """
    ckpt = torch.load(path)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        schema = feature_schemas.from_description(ckpt.get("feature_schema"))
    else:
        state_dict = ckpt
        schema = feature_schemas.LEGACY
    model = BackgammonNN(schema.size, hidden_sizes=hidden_sizes_of(state_dict))
    model.load_state_dict(state_dict)
    model.eval()
    return model, schema


class FinalNNAgent:
    def __init__(self, model_path: str = None, checkpoint_path: str = None, extract_features_fn=None,
                 hidden_sizes=None):
        """
        Args:
            model_path (str): Path to a raw model state_dict (.pt/.pth).
            checkpoint_path (str, optional): Full checkpoint file (with optimizer, epoch, etc.).
                                             If provided, this is used instead of `model_path`.
                                             With neither, `DEFAULT_CHECKPOINT` is loaded.
            extract_features_fn: override the features of a white-34 network
                                 (the checkpoint's schema decides otherwise).
            hidden_sizes: hidden layer widths; read from the checkpoint when omitted,
                          so 34-40-1 and 34-128-128-1 networks both load.
        """
        load_path = checkpoint_path or model_path or DEFAULT_CHECKPOINT
        self.model, self.schema = load_network(load_path)
        if hidden_sizes is not None and list(hidden_sizes) != hidden_sizes_of(self.model.state_dict()):
            raise ValueError(f"{load_path} has hidden layers {hidden_sizes_of(self.model.state_dict())}, not {hidden_sizes}")
        if extract_features_fn is not None and self.schema.canonical:
            raise ValueError(f"{load_path} uses the {self.schema.name} schema; its features cannot be overridden")
        self.extract_features = extract_features_fn or self.schema.features
    
    def select_move(self, board):
        if not board.valid_moves:
            return []
        if self.schema.canonical:
            moves = board.valid_moves
            children = [board.clone() for _ in moves]
            for child, move in zip(children, moves):
                child.move_from_sequence(move)
            rows = self.schema.afterstate_rows(board.turn, children)
            with torch.no_grad():
                values = self.model(torch.tensor(rows, dtype=torch.float32)).reshape(-1).tolist()
            return moves[self.schema.best_index(board.turn, children, values)]
        
        board_copies = [deepcopy(board) for _ in board.valid_moves]
        for i, move in enumerate(board.valid_moves):
            board_copies[i].move_from_sequence(move)
        
        features_batch = torch.stack([torch.tensor(self.extract_features(b), dtype=torch.float32) for b in board_copies])
        
        with torch.no_grad():
            self.model.eval()
            values = self.model(features_batch).squeeze()
        
        if board.turn == 1:
            best_idx = values.argmax().item()
        else:
            best_idx = values.argmin().item()
        return board.valid_moves[best_idx] 

    def win_probability(self, board, player):
        """The network's P(``player`` wins) for ``board`` with ``board.turn`` about to roll."""
        if board.game_over:
            return self.schema.win_probability(board, None, player)
        with torch.no_grad():
            output = self.model(torch.tensor([self.extract_features(board)], dtype=torch.float32)).item()
        return self.schema.win_probability(board, output, player)

def main(resume=False, epoch_count=20, name="canonical"):
    """Main function to train and evaluate the backgammon AI."""
    schema = feature_schemas.CANONICAL
    input_size = schema.size
    
    print(f"Feature vector size: {input_size}")
    
    # 9k games / 2 hours
    # 4.5k games / 1 hours
    
    num_hours = 12
    games_per_epoch = 500
    epoch_per_hour = 37
    epoch_count = 260 # epoch_per_hour * num_hours
    eval_games = 500
    
    # Create neural network model
    model = BackgammonNN(input_size=input_size, hidden_sizes=[40])
    
    # Initialize TD-Lambda learner
    td_lambda = TDLambda(model, learning_rate=0.1, lambda_param=0.7)
    
    # Create trainer
    trainer = BackgammonTrainer(
        model=model,
        schema=schema,
        td_lambda=td_lambda,
        games_per_epoch=games_per_epoch,
        eval_games=eval_games
    )
    
    print("Starting training")    

    # models = {
    #     "main_model": model,
    # }
    
    # Define a function to train a model
    def train_model(trainer: BackgammonTrainer, name, num_epochs, resume=False):
        # The same path train() writes the latest checkpoint to.
        resume_from = f"models/{name}/{name}_checkpoint_latest.pt" if resume else None
        return trainer.train(num_epochs=num_epochs, name=name, resume_from=resume_from)
    
    # A new name, so a canonical run never overwrites the committed white-34 checkpoints.
    train_model(trainer, name, epoch_count, resume=resume)
    

    # trainer.plot_learning_curve(save_path="learning_curve.png")
    # trainer.save_model("final_model.pt")    
    
    # Run tournament between models
    # print("\nRunning model tournament...")
    # comparator = ModelComparator(models, extract_features, num_games=100)
    # comparator.run_tournament()
    # comparator.print_results()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Backgammon AI')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--evaluate', action='store_true', help='Only evaluate the model without training')
    parser.add_argument('--model', type=str, default='final_model.pt', help='Model path for evaluation')
    parser.add_argument('--name', type=str, default='canonical', help='Run name: checkpoints go to models/<name>/')
    
    args = parser.parse_args()
    
    if args.evaluate:
        # Only run evaluation on a saved model
        input_size = len(extract_features(Board()))
        model = BackgammonNN(input_size=input_size)
        model.load_state_dict(torch.load(args.model))
        nn_agent = NNAgent(model, extract_features, exploration_rate=0.0)
        
        evaluator = Evaluator(nn_agent, opponent_agent=RandomAgent(), num_games=50)
        eval_results = evaluator.evaluate()
        
        print("\nEvaluation Results:")
        print("-" * 50)
        for key, value in eval_results.items():
            print(f"{key}: {value}")
    else:
        # Run training
        main(resume=args.resume, epoch_count=args.epochs, name=args.name)

