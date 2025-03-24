import matplotlib.pyplot as plt
from random import choice

next_round = ['seaborn-v0_8', 'fivethirtyeight', 'ggplot', 'seaborn-v0_8-dark', 'seaborn-v0_8-deep',
                      'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks']
def plot_eval_curve(txt_path, save_path=None):
        with open(txt_path, "r") as f:
            lines = f.readlines()
            epochs = [int(line.split("\t")[0]) for line in lines]
            white_win_rates = [float(line.split("\t")[1]) for line in lines]
            black_win_rates = [float(line.split("\t")[2]) for line in lines]
            win_rates = [float(line.split("\t")[3]) for line in lines]
        
        
        style = choice(next_round)
        next_round.remove(style)
        plt.style.use(style)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, white_win_rates, '-o', label='White Win Rate')
        plt.plot(epochs, black_win_rates, '-o', label='Black Win Rate')
        plt.plot(epochs, win_rates, '-o', label='Total Win Rate')
        
        plt.ylim(0.2, 1.1)
        plt.xlabel('Epoch')
        plt.ylabel('Win Rate vs Random Agent')
        plt.title(f'Evaluation Progress / Win Rate of {txt_path[:-4]} against Random Agent')
        plt.grid(True)
        plt.legend()
        
        plt.xlabel('Epoch')
        plt.ylabel('Win Rate')
        plt.title('Evaluation Progress')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(f"eval_results/{save_path}")
            print(f"Evaluation curve saved to {save_path}, style: {style}")
    
    
names = ["low_lambda", "small", "lambda_0.4", "high_lr", "hidden_128", "main"]
for name in names:
    plot_eval_curve(f"eval_results/results_{name}.txt", f"graph_{name}.png")

