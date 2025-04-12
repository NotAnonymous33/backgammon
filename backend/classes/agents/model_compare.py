try:
    from classes.board_cpp import Board # type: ignore
    from classes.agents.NNAgent import NNAgent, extract_features, BackgammonNN
except ImportError:
    from board_cpp import Board # type: ignore
    from .NNAgent import NNAgent, extract_features, BackgammonNN

import torch

def checkpoint_to_model(checkpoint_path, input_size=None, hidden_sizes=[128, 128]):
    """
    Convert a checkpoint to a standalone model.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        input_size: Input size for the model (required if creating a new model)
        hidden_sizes: Hidden layer sizes for the model architecture
        
    Returns:
        model: The extracted model with loaded weights
    """
    try:
        # Load the checkpoint dictionary
        checkpoint = torch.load(checkpoint_path)
        
        # Check if it contains a model_state_dict
        if 'model_state_dict' not in checkpoint:
            raise ValueError("Checkpoint does not contain model_state_dict")
        
        # Create a new model with the same architecture
        if input_size is None:
            # Try to infer input size from the first layer weight
            first_layer_name = [k for k in checkpoint['model_state_dict'].keys() if 'model.0.weight' in k][0]
            inferred_input_size = checkpoint['model_state_dict'][first_layer_name].shape[1]
            print(f"Inferred input size: {inferred_input_size}")
            model = BackgammonNN(input_size=inferred_input_size, hidden_sizes=hidden_sizes)
        else:
            model = BackgammonNN(input_size=input_size, hidden_sizes=hidden_sizes)
        
        # Load just the model state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode
        model.eval()
        
        # Optionally print some info about the checkpoint
        if 'epoch' in checkpoint:
            print(f"Model extracted from checkpoint at epoch {checkpoint['epoch']}")
        if 'win_rate' in checkpoint:
            print(f"Checkpoint win rate: {checkpoint['win_rate']:.4f}")
            
        return model
        
    except Exception as e:
        print(f"Error extracting model from checkpoint: {e}")
        raise
    
    
    
    
class ModelComparator:
    def __init__(self, model_names, extract_features_fn, num_games=50):
        """
        Compare multiple trained models against each other.
        
        Args:
            model_paths: List of model paths to compare
            extract_features_fn: Function to extract features from a board
            num_games: Number of games per match-up
        """
        self.model_names = model_names
        self.extract_features = extract_features_fn
        self.num_games = num_games
        self.results = {}
        
    def run_tournament(self):
        """Run a round-robin tournament between all models."""
        results = {name: {"wins": 0, "games": 0} for name in self.model_names}
        
        # For each pair of models, play games
        for i, model1_name in enumerate(self.model_names):
            for model2_name in self.model_names[i+1:]:
                print(f"Match: {model1_name} vs {model2_name}")
                
                # model_path=f"models/{model1_name}/backgammon_{model1_name}_latest.pt"
                
                agent1 = NNAgent(checkpoint_to_model(f"models/{model1_name}/backgammon_{model1_name}_latest.pt"), self.extract_features, exploration_rate=0)
                agent2 = NNAgent(checkpoint_to_model(f"models/{model2_name}/backgammon_{model2_name}_latest.pt"), self.extract_features, exploration_rate=0)
                
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

if __name__ == "__main__":
    model_names = ["hidden_128", "hidden_128_learning", "hidden_128_learning_lambda"]
    num_games = 500
    comparator = ModelComparator(model_names, extract_features_fn=extract_features, num_games=num_games)
    results = comparator.run_tournament()
    comparator.print_results()