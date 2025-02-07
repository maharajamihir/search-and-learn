import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def stdd_entropy_losses_plot(best_of_n_completions_dir):
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

if __name__ == "__main__":
    # add argument parser
    parser = argparse.ArgumentParser(description="Plot entropy losses for best of n completions")
    parser.add_argument("--best_of_n_completions_dir", type=str, help="Path to the best of n completions directory")
    args = parser.parse_args()

    best_of_n_completions_dir = Path(args.best_of_n_completions_dir)
    stdd_entropy_losses_plot(best_of_n_completions_dir)
