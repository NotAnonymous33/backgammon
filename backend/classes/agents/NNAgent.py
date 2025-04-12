try:
    from classes.board_cpp import Board # type: ignore
    from classes.agents.RandomAgent import RandomAgent
    from classes.agents.HeuristicAgent import HeuristicAgent
except ImportError:
    
    from board_cpp import Board # type: ignore
    from .RandomAgent import RandomAgent
    from .HeuristicAgent import HeuristicAgent
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
from time import perf_counter
from tqdm import tqdm

import torch._dynamo as dynamo

# Enable verbose logging to understand why compilation fails
dynamo.config.verbose = True

# Raise an error instead of falling back to eager mode
dynamo.config.suppress_errors = False

torch.set_float32_matmul_precision('medium')

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
    
    @torch.compile(mode="reduce-overhead")
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
        if not board.valid_moves:
            return []
            
        # Occasionally make a random move for exploration
        if random.random() < self.exploration_rate:
            return random.choice(board.valid_moves)
        
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


class BackgammonTrainer:
    def __init__(self, model, extract_features_fn, td_lambda, games_per_epoch=1000, eval_games=100):
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
        self.eval_games = eval_games
    
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
        
        for game_num in tqdm(range(self.games_per_epoch)):
            # Play a complete game
            winner, states = self.play_game()
            
            self.td_lambda.update(states, float(winner))
            if winner == 1:  # White won
                wins_white += 1
                
        win_rate = wins_white / self.games_per_epoch
        return win_rate
    
    
    def play_game(self):
        """
        Play a complete game of backgammon using the current model.
        
        Returns:
            winner: The winner of the game (1 for white, -1 for black)
            states: List of player state tensors
        """
        board = Board()
        agent = NNAgent(self.model, self.extract_features)
        
        states = []
        
        move_count = 0
        max_moves = 200  # Prevent infinite games
        
        while not board.game_over and move_count < max_moves:
            # Roll dice
            board.roll_dice()         
    
            # Extract features from current state
            features = self.extract_features(board)
            state_tensor = torch.tensor([features], dtype=torch.float32)
            
            # Store state
            states.append(state_tensor)
            # Choose move
            move = agent.select_move(board)
            
            # Make move
            board.move_from_sequence(move)
            move_count += 1
        
        # Force an end if max moves reached
        if move_count >= max_moves:
            # The player with more pieces borne off wins
            if board.white_off == board.black_off:
                winner = 1 if board.white_left < board.black_left else -1
            else:
                winner = 1 if board.white_off > board.black_off else -1
        else:
            winner = 1 if board.white_off == 15 else -1
        
        if not states:
            states.append(torch.zeros([1, self.extract_features(board).shape[0]], dtype=torch.float32))
            
        return winner, states
    
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
            win_rate = self.train_epoch()
            
            results.append(win_rate)
            self.last_win_rate = win_rate
            self.training_results = results
            
            print(f"Epoch {epoch+1}/{start_epoch + num_epochs}, Win Rate: {win_rate:.4f}")
            
            # Save checkpoint at regular intervals
            if (epoch + 1) % checkpoint_interval == 0:
                if name != "":
                    self.save_checkpoint(f"models/{name}/backgammon_checkpoint_epoch_{name}{epoch+1}.pt", epoch=epoch+1)
                else:
                    self.save_checkpoint(f"models/backgammon_checkpoint_epoch_{epoch+1}.pt", epoch=epoch+1)
            
            e = Evaluator(NNAgent(self.model, self.extract_features, exploration_rate=0.0), opponent_agent=HeuristicAgent(), num_games=self.eval_games)
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
                self.save_checkpoint(f"models/{name}/backgammon_{name}_latest.pt", epoch=epoch+1)
            else:
                self.save_checkpoint("models/backgammon_latest.pt", epoch=epoch+1)

        
        self.plot_eval_curve(f"eval_results/results_{name}.txt", f"graph_{name}.png")
        
        # Save final model
        if name != "":
            # create dir if it does not exist
            if not os.path.exists("models/" + name):
                os.makedirs("models/" + name)
            self.save_model(f"{name}/backgammon_{name}_final.pt")
        else:
            self.save_model(f"backgammon_final_model.pt")
        
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
            
    # TODO: remove
    # def evaluate(self):
    #     """Evaluate the model against a random agent."""
    #     nn_agent = NNAgent(self.model, self.extract_features, exploration_rate=0.0)
    
    #     # # Evaluate against random agent
    #     # print("Evaluating against random agent...")
    #     # evaluator = Evaluator(nn_agent, opponent_agent=RandomAgent(), num_games=50)
    #     # eval_results = evaluator.evaluate()
        
    #     # Evaluate against heuristic agent
    #     print("Evaluating against heuristic agent...")
    #     evaluator = Evaluator(nn_agent, opponent_agent=HeuristicAgent(), num_games=50)
    #     eval_results = evaluator.evaluate()
        
    #     # Print evaluation results
    #     print("\nEvaluation Results:")
    #     print("-" * 50)
    #     for key, value in eval_results.items():
    #         print(f"{key}: {value}")
        
    def save_model(self, filename):
        filename = "models/" + filename
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
            
            while not board.game_over and turn_count < self.move_limit:  # Limit to prevent infinite games
                board.roll_dice()
                
                if board.turn == 1:
                    move = self.opponent_agent.select_move(board)
                else:
                    move = self.nn_agent.select_move(board)
                
                board.move_from_sequence(move)
                turn_count += 1
            
            black_turn_counts.append(turn_count)
            
            if board.black_off == 15 or (turn_count >= self.move_limit and board.black_off > board.white_off):
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

class FinalNNAgent:
    def __init__(self, model_path, extract_features_fn=extract_features):
        self.extract_features = extract_features_fn
        self.model = BackgammonNN(len(extract_features(Board())))
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def select_move(self, board):
        if not board.valid_moves:
            return []
        
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

def main(resume=False, epoch_count=20):
    torch.autograd.set_detect_anomaly(True)
    """Main function to train and evaluate the backgammon AI."""
    # Assuming the extract_features function already exists
    
    # Define input size based on feature extractor output
    input_size = len(extract_features(Board()))
    
    print(f"Feature vector size: {input_size}")
    
    # Create neural network model
    # model = BackgammonNN(input_size=input_size)
    
    # Initialize TD-Lambda learner
    # td_lambda = TDLambda(model, learning_rate=0.05, lambda_param=0.9)
    
    # # Create trainer
    # trainer = BackgammonTrainer(
    #     model=model,
    #     extract_features_fn=extract_features,
    #     td_lambda=td_lambda,
    #     games_per_epoch=games_per_epoch,
    #     eval_games=eval_games
    # )
    
    print("Starting training")    

    models = {
        # "main_model": model,
    }
    
    games_per_epoch = 300 # 100 10 100
    epoch_count = 30
    eval_games = 200
    
    
    # Train a model with hidden sizes [128, 128], learning rate 0.05, and lambda 0.7
    hidden_128_model = BackgammonNN(input_size=input_size, hidden_sizes=[128, 128])
    hidden_128_td = TDLambda(hidden_128_model, learning_rate=0.05, lambda_param=0.7)
    hidden_128_trainer = BackgammonTrainer(hidden_128_model, extract_features, hidden_128_td, games_per_epoch=games_per_epoch, eval_games=eval_games)
    
    # Train a model with hidden sizes [128, 128], learning rate 0.05, and lambda 0.9
    hidden_128_model_lambda = BackgammonNN(input_size=input_size, hidden_sizes=[128, 128])
    hidden_128_td_lambda = TDLambda(hidden_128_model_lambda, learning_rate=0.05, lambda_param=0.9)
    hidden_128_trainer_lambda = BackgammonTrainer(hidden_128_model_lambda, extract_features, hidden_128_td_lambda, games_per_epoch=games_per_epoch, eval_games=eval_games)

    # Train a model with hidden sizes [128, 128], learning rate 0.1, and lambda 0.7
    hidden_128_model_learning = BackgammonNN(input_size=input_size, hidden_sizes=[128, 128])
    hidden_128_td_learning = TDLambda(hidden_128_model_learning, learning_rate=0.1, lambda_param=0.7)
    hidden_128_trainer_learning = BackgammonTrainer(hidden_128_model_learning, extract_features, hidden_128_td_learning, games_per_epoch=games_per_epoch, eval_games=eval_games)

    # Train a model with hidden sizes [128, 128], learning rate 0.1, and lambda 0.9
    hidden_128_model_learning_lambda = BackgammonNN(input_size=input_size, hidden_sizes=[128, 128])
    hidden_128_td_learning_lambda = TDLambda(hidden_128_model_learning_lambda, learning_rate=0.1, lambda_param=0.9)
    hidden_128_trainer_learning_lambda = BackgammonTrainer(hidden_128_model_learning_lambda, extract_features, hidden_128_td_learning_lambda, games_per_epoch=games_per_epoch, eval_games=eval_games)
    
    
    # Define a function to train a model
    def train_model(trainer: BackgammonTrainer, name, num_epochs, resume=False):
        return trainer.train(num_epochs=num_epochs, name=name, resume_from="backgammon_latest.pt" if resume else None)
    
    models["hidden_128"] = hidden_128_model
    models["hidden_128_lambda"] = hidden_128_model_lambda
    models["hidden_128_learning"] = hidden_128_model_learning
    models["hidden_128_learning_lambda"] = hidden_128_model_learning
    
    trainer_list = [
        (hidden_128_trainer, "hidden_128"),
        (hidden_128_trainer_lambda, "hidden_128_lambda"),
        (hidden_128_trainer_learning, "hidden_128_learning"),
        (hidden_128_trainer_learning_lambda, "hidden_128_learning_lambda"),
    ]
    
    for trainer, name in tqdm(trainer_list, desc="Training models"):
        train_model(trainer, name, epoch_count)
    
    # if resume:
    #     train_model(trainer, "main", epoch_count, resume=True)
    # else:
    #     train_model(trainer, "main", epoch_count)
    

    # trainer.plot_learning_curve(save_path="learning_curve.png")
    # trainer.save_model("backgammon_final_model.pt")    
    
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
        main(resume=args.resume, epoch_count=args.epochs)

