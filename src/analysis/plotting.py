# TODO @mihir
import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import numpy as np
import csv


class Plotter:
    """
    A class to handle plotting of empirical difficulty against level labels.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.problems = []
        self.plots_dir = Path(self.file_path).parent.parent / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, n_problems):
        """
        Load a JSONL file containing problems and return them as a list of dictionaries.
        """

        # Load the JSONL file and extract data
        with open(self.file_path, 'r') as f:
            for idx, line in tqdm(enumerate(f), desc="Loading data", total=n_problems):
                if idx > n_problems:
                    break

                problem = json.loads(line)
                self.problems.append(problem)

    def plot_difficulty_comparisons(self):
        """
        Compare per-problem difficulties using Pearson and Spearman correlations, and calculate L1 and L2 losses.
        Additionally, plot each difficulty type against empirical difficulties.
        """
        empirical_difficulties = np.array([result['empirical_difficulty'] for result in self.problems])
        comparison_results = {}

        for difficulty_type in ['dataset_difficulty', 'prm_based_difficulty', 'varentropy_difficulty', 
                                'verbal_difficulty', 'probability_of_difficult', 'difficulty_probe_difficulty']:
            # Check if the difficulty type exists in any of the results
            if any(difficulty_type in result for result in self.problems):
                # Handle different structures for difficulties
                if isinstance(self.problems[0].get(difficulty_type, {}), dict):
                    for sub_type in self.problems[0][difficulty_type]:
                        sub_difficulties = np.array([result[difficulty_type][sub_type] for result in self.problems if difficulty_type in result])
                        # Normalize difficulties
                        sub_difficulties_normalized = (sub_difficulties - np.min(sub_difficulties)) / (np.max(sub_difficulties) - np.min(sub_difficulties))
                        empirical_difficulties_normalized = (empirical_difficulties - np.min(empirical_difficulties)) / (np.max(empirical_difficulties) - np.min(empirical_difficulties))
                        
                        # Calculate correlations
                        pearson_corr, pearson_p = pearsonr(empirical_difficulties, sub_difficulties)
                        spearman_corr, spearman_p = spearmanr(empirical_difficulties, sub_difficulties)
                        
                        # Calculate L1 and L2 losses
                        l1_loss = np.sum(np.abs(empirical_difficulties_normalized - sub_difficulties_normalized))
                        l2_loss = np.sqrt(np.sum((empirical_difficulties_normalized - sub_difficulties_normalized) ** 2))
                        
                        comparison_results[f"{difficulty_type}_{sub_type}"] = {
                            "pearson_corr": pearson_corr,
                            "pearson_p": pearson_p,
                            "spearman_corr": spearman_corr,
                            "spearman_p": spearman_p,
                            "l1_loss": l1_loss,
                            "l2_loss": l2_loss
                        }

                        # Plot sub_difficulties against empirical difficulties
                        plt.figure(figsize=(10, 6))
                        plt.scatter(empirical_difficulties, sub_difficulties, alpha=0.7)
                        plt.title(f'Empirical vs {difficulty_type}_{sub_type} Difficulty')
                        plt.xlabel('Empirical Difficulty')
                        plt.ylabel(f'{difficulty_type}_{sub_type} Difficulty')
                        plt.grid(True)
                        plt.show()
                        plot_file_path = self.plots_dir / f"empirical_vs_{difficulty_type}_{sub_type}_difficulty.png"
                        plt.savefig(plot_file_path)
                        print(f"Plot saved to {plot_file_path}")

                else:
                    difficulties = np.array([result[difficulty_type] for result in self.problems if difficulty_type in result])
                    # Normalize difficulties
                    difficulties_normalized = (difficulties - np.min(difficulties)) / (np.max(difficulties) - np.min(difficulties))
                    empirical_difficulties_normalized = (empirical_difficulties - np.min(empirical_difficulties)) / (np.max(empirical_difficulties) - np.min(empirical_difficulties))
                    
                    # Calculate correlations
                    pearson_corr, pearson_p = pearsonr(empirical_difficulties, difficulties)
                    spearman_corr, spearman_p = spearmanr(empirical_difficulties, difficulties)
                    
                    # Calculate L1 and L2 losses
                    l1_loss = np.sum(np.abs(empirical_difficulties_normalized - difficulties_normalized))
                    l2_loss = np.sqrt(np.sum((empirical_difficulties_normalized - difficulties_normalized) ** 2))
                    
                    comparison_results[difficulty_type] = {
                        "pearson_corr": pearson_corr,
                        "pearson_p": pearson_p,
                        "spearman_corr": spearman_corr,
                        "spearman_p": spearman_p,
                        "l1_loss": l1_loss,
                        "l2_loss": l2_loss
                    }

                    # Plot difficulties against empirical difficulties
                    plt.figure(figsize=(10, 6))
                    plt.scatter(empirical_difficulties, difficulties, alpha=0.7)
                    plt.title(f'Empirical vs {difficulty_type} Difficulty')
                    plt.xlabel('Empirical Difficulty')
                    plt.ylabel(f'{difficulty_type} Difficulty')
                    plt.grid(True)
                    plt.show()
                    plot_file_path = self.plots_dir / f"empirical_vs_{difficulty_type}_difficulty.png"
                    plt.savefig(plot_file_path)
                    print(f"Plot saved to {plot_file_path}")

        output_json_path = self.plots_dir / "comparison_results.json"
        with open(output_json_path, 'w') as outfile:
            json.dump(comparison_results, outfile, indent=4)
        print(f"Saved comparison results to {output_json_path}")

        # Save as CSV
        output_csv_path = self.plots_dir / "comparison_results.csv"
        with open(output_csv_path, 'w', newline='') as csvfile:
            fieldnames = ["difficulty_type", "pearson_corr", "pearson_p", "spearman_corr", "spearman_p", "l1_loss", "l2_loss"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for difficulty_type, metrics in comparison_results.items():
                row = {"difficulty_type": difficulty_type, **metrics}
                writer.writerow(row)
        print(f"Saved comparison results to {output_csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot empirical difficulty against level.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the JSONL file containing accumulated difficulties.')
    parser.add_argument('--num_problems', type=int, default=float('inf'), help='Number of problems to analyze.')
    args = parser.parse_args()

    plotter = Plotter(args.file_path)
    plotter.load_data(args.num_problems)

    plotter.plot_difficulty_comparisons()


if __name__ == "__main__":
    main()
