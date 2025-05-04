import matplotlib.pyplot as plt


def plot_eval_curve(txt_path, save_path=None):
        with open(txt_path, "r") as f:
            lines = f.readlines()
            epochs = [int(line.split("\t")[0]) for line in lines]
            white_win_rates = [float(line.split("\t")[1]) for line in lines]
            black_win_rates = [float(line.split("\t")[2]) for line in lines]
            win_rates = [float(line.split("\t")[3]) for line in lines]
        
        
        plt.style.use('seaborn-v0_8-talk')
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, white_win_rates, '-', color='#1f77b4', alpha=0.3, label='White Win Rate') 
        plt.plot(epochs, black_win_rates, '-', color='#d62728', alpha=0.3, label='Black Win Rate') 
        plt.plot(epochs, win_rates, '-', color='#2ca02c', label='Total Win Rate')     
        
        plt.ylim(0, 1.1)
        plt.xlabel('Epoch')
        plt.ylabel('Win Rate vs Heuristic Agent')
        plt.title(f'Evaluation Progress / Win Rate of {txt_path[21:-4]} against Heuristic Agent')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(f"eval_results/{save_path}")
            print(f"Evaluation curve saved to {save_path}")
    
    
names = ["tdgammon", "momentum"]
for name in names:
    plot_eval_curve(f"eval_results/results_{name}.txt", f"graph_{name}.png")

