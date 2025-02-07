import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# n samples at x and loss at y
def stdd_entropy_losses_plot_over_all_samples(best_of_n_completions_dir):
    temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    available_temperatures = []
    stdd_l1 = []
    stdd_l2 = []

    for temp in temperatures:
        dir_name = f"best_of_n_completions_temp_{temp}_analysis"
        json_path = best_of_n_completions_dir / dir_name / "entropy_losses.json"
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
                available_temperatures.append(temp)
                stdd_l1.append(data["stddentropy_l1"])  # Collect all values
                stdd_l2.append(data["stddentropy_l2"])  # Collect all values
        else:
            print(f"Warning: {json_path} does not exist.")

    plt.figure(figsize=(10, 6))
    for i, temp in enumerate(available_temperatures):
        plt.plot(stdd_l1[i], label=f'stddentropy_l1 (T={temp})', marker='o')
        plt.plot(stdd_l2[i], label=f'stddentropy_l2 (T={temp})', marker='o')
    
    plt.xlabel('Sample Index')
    plt.ylabel('Standard Deviation of Entropy')
    plt.title('Standard Deviation of Entropy Losses vs Temperature')
    plt.legend()
    plt.grid(True)
    # save the plot instead of showing it
    plt.savefig(best_of_n_completions_dir / "stdd_entropy_losses.png")

# temp at x and loss at y
def stdd_entropy_losses_plot_over_temperature(best_of_n_completions_dir):
    temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    l1_losses = []
    l2_losses = []

    for temp in temperatures:
        dir_name = f"best_of_n_completions_temp_{temp}_analysis"
        json_path = best_of_n_completions_dir / dir_name / "stddentropy_l1_l2_loss.json"
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
                l1_losses.append(data["l1_loss"])
                l2_losses.append(data["l2_loss"])
        else:
            print(f"Warning: {json_path} does not exist.")
            l1_losses.append(None)
            l2_losses.append(None)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('L1 Loss', color='tab:blue')
    ax1.plot(temperatures, l1_losses, label='L1 Loss', color='tab:blue', marker='o')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('L2 Loss', color='tab:red')
    ax2.plot(temperatures, l2_losses, label='L2 Loss', color='tab:red', marker='x')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title('L1 and L2 Losses vs Temperature')
    plt.grid(True)
    plt.savefig(best_of_n_completions_dir / "stdd_entropy_losses_temperature.png")

if __name__ == "__main__":
    # add argument parser
    parser = argparse.ArgumentParser(description="Plot entropy losses for best of n completions")
    parser.add_argument("--best_of_n_completions_dir", type=str, help="Path to the best of n completions directory")
    args = parser.parse_args()

    best_of_n_completions_dir = Path(args.best_of_n_completions_dir)
    # stdd_entropy_losses_plot_over_all_samples(best_of_n_completions_dir)
    stdd_entropy_losses_plot_over_temperature(best_of_n_completions_dir)
