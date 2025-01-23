import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import List, Dict, Any
from pathlib import Path


def plot_per_problem_logprobs(results: List[Dict[str, Any]], output_dir: Path, show_std: bool = True):
    # Group results by unique_id
    problems = []
    means = []
    stds = []
    for r in results:
        problems.append(r['unique_id'])
        means.append(-r['avg_by_rank'][0])  # rank 1 logprobs
        stds.append(r['stds_by_rank'][0])  # rank 1 standard deviations
    # Create the plot
    plt.figure(figsize=(12, 6))
    x = range(len(problems))
    if show_std:
        plt.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color='blue', 
                label='Mean with ±1 std dev')
    else:
        plt.bar(x, means, alpha=0.7, color='blue', label='Mean')
    

    
    plt.title('Average Negative Log Probability per Problem')
    plt.xlabel('Problem')
    plt.ylabel('Average Negative Log Probability')
    plt.grid(True)
    plt.legend()
    
    # Set problem names as x-tick labels
    plt.xticks(x, problems, rotation=45, ha='right')
    if len(problems) > 20:
        # Show only every nth label to avoid overcrowding
        plt.gca().set_xticks(x[::len(x)//20])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_problem_logprobs.png')
    plt.close()

def plot_per_problem_entropy(results: List[Dict[str, Any]], output_dir: Path):
    # Group results by unique_id
    problems = []
    means = []
    for r in results:
        problems.append(r['unique_id'])
        means.append(r['avg_entropy'])
    
    # Calculate mean and std for each problem
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    x = range(len(problems))
    plt.bar(x, means, alpha=0.7, color='red', label='Mean')
    
    plt.title('Average Entropy per Problem')
    plt.xlabel('Problem')
    plt.ylabel('Average Entropy')
    plt.grid(True)
    plt.legend()
    
    # Set problem names as x-tick labels
    plt.xticks(x, problems, rotation=45, ha='right')
    if len(problems) > 20:
        # Show only every nth label to avoid overcrowding
        plt.gca().set_xticks(x[::len(x)//20])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_problem_entropy.png')
    plt.close()

def plot_per_problem_level(results: List[Dict[str, Any]], output_dir: Path):
    # Group results by unique_id and get level
    problems = [r['unique_id'] for r in results]
    levels = [r['level'] for r in results]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    x = range(len(problems))
    plt.bar(x, levels, alpha=0.7, color='green')
    
    plt.title('Problem Level per Problem')
    plt.xlabel('Problem')
    plt.ylabel('Level')
    plt.grid(True)
    
    # Set problem names as x-tick labels
    plt.xticks(x, problems, rotation=45, ha='right')
    if len(problems) > 20:
        # Show only every nth label to avoid overcrowding
        plt.gca().set_xticks(x[::len(x)//20])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_problem_level.png')
    plt.close()

def plot_pass_at_1(results: List[Dict[str, Any]], output_dir: Path):
    # Extract pass@1 data
    problems = [r['unique_id'] for r in results]
    pass_at_1 = [r['pass@1'] for r in results]
    print(f"Number of pass@1 values greater than 0: {sum(1 for p in pass_at_1 if p > 0.)}")
    # Create the plot
    plt.figure(figsize=(12, 6))
    x = range(len(problems))
    plt.bar(x, pass_at_1, alpha=0.7, color='purple')
    
    plt.title('Pass@1 per Problem')
    plt.xlabel('Problem')
    plt.ylabel('Pass@1')
    plt.grid(True)
    
    # Set problem names as x-tick labels
    plt.xticks(x, problems, rotation=45, ha='right')
    if len(problems) > 20:
        plt.gca().set_xticks(x[::len(x)//20])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pass_at_1.png')
    plt.close()

def plot_mean_score(results: List[Dict[str, Any]], output_dir: Path):
    # Extract mean score data
    problems = [r['unique_id'] for r in results]
    mean_scores = [r['mean_score'] for r in results]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    x = range(len(problems))
    plt.bar(x, mean_scores, alpha=0.7, color='orange')
    
    plt.title('Mean Score per Problem')
    plt.xlabel('Problem')
    plt.ylabel('Mean Score')
    plt.grid(True)
    
    # Set problem names as x-tick labels
    plt.xticks(x, problems, rotation=45, ha='right')
    if len(problems) > 20:
        plt.gca().set_xticks(x[::len(x)//20])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mean_score.png')
    plt.close()

def plot_level_mean(results: List[Dict[str, Any]], output_dir: Path):
    # Extract level mean data
    problems = [r['unique_id'] for r in results]
    level_means = [r['level_mean'] for r in results]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    x = range(len(problems))
    plt.bar(x, level_means, alpha=0.7, color='cyan')
    
    plt.title('Level Mean per Problem')
    plt.xlabel('Problem')
    plt.ylabel('Level Mean')
    plt.grid(True)
    
    # Set problem names as x-tick labels
    plt.xticks(x, problems, rotation=45, ha='right')
    if len(problems) > 20:
        plt.gca().set_xticks(x[::len(x)//20])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'level_mean.png')
    plt.close()

def plot_level_pass(results: List[Dict[str, Any]], output_dir: Path):
    # Extract level pass data
    problems = [r['unique_id'] for r in results]
    level_pass = [r['level_pass'] for r in results]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    x = range(len(problems))
    plt.bar(x, level_pass, alpha=0.7, color='magenta')
    
    plt.title('Level Pass per Problem')
    plt.xlabel('Problem')
    plt.ylabel('Level Pass')
    plt.grid(True)
    
    # Set problem names as x-tick labels
    plt.xticks(x, problems, rotation=45, ha='right')
    if len(problems) > 20:
        plt.gca().set_xticks(x[::len(x)//20])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'level_pass.png')
    plt.close()

def plot_entropy_vs_level_mean(results: List[Dict[str, Any]], output_dir: Path):
    entropies = [r['avg_entropy'] for r in results]
    level_means = [r['level_mean'] for r in results]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(level_means, entropies, alpha=0.7, color='blue')
    
    plt.title('Entropy vs Level Mean')
    plt.xlabel('Level Mean')
    plt.ylabel('Entropy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_vs_level_mean.png')
    plt.close()

def plot_entropy_vs_level_pass(results: List[Dict[str, Any]], output_dir: Path):
    entropies = [r['avg_entropy'] for r in results]
    level_pass = [r['level_pass'] for r in results]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(level_pass, entropies, alpha=0.7, color='green')
    
    plt.title('Entropy vs Level Pass')
    plt.xlabel('Level Pass')
    plt.ylabel('Entropy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_vs_level_pass.png')
    plt.close()

def plot_logprobs_vs_level_mean(results: List[Dict[str, Any]], output_dir: Path):
    avg_logprobs = [-r['avg_logprob_per_token'] for r in results]
    level_means = [r['level_mean'] for r in results]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(level_means, avg_logprobs, alpha=0.7, color='red')
    
    plt.title('Neg Logprobs vs Level Mean')
    plt.xlabel('Level Mean')
    plt.ylabel('Average Neg Logprob per Token')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'logprobs_vs_level_mean.png')
    plt.close()

def plot_logprobs_vs_level_pass(results: List[Dict[str, Any]], output_dir: Path):
    avg_logprobs = [-r['avg_logprob_per_token'] for r in results]
    level_pass = [r['level_pass'] for r in results]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(level_pass, avg_logprobs, alpha=0.7, color='purple')
    
    plt.title('Neg Logprobs vs Level Pass')
    plt.xlabel('Level Pass')
    plt.ylabel('Average Neg Logprob per Token')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'logprobs_vs_level_pass.png')
    plt.close()



def plot_entropy_vs_mean_score(results: List[Dict[str, Any]], output_dir: Path):
    entropies = [r['avg_entropy'] for r in results]
    mean_scores = [r['mean_score'] for r in results if r['mean_score'] is not None]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(mean_scores, entropies, alpha=0.7, color='orange')
    
    plt.title('Entropy vs Mean Score')
    plt.xlabel('Mean Score')
    plt.ylabel('Entropy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_vs_mean_score.png')
    plt.close()

def plot_entropy_vs_pass_at_1(results: List[Dict[str, Any]], output_dir: Path):
    entropies = [r['avg_entropy'] for r in results]
    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(pass_at_1, entropies, alpha=0.7, color='brown')
    
    plt.title('Entropy vs Pass@1')
    plt.xlabel('Pass@1')
    plt.ylabel('Entropy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_vs_pass_at_1.png')
    plt.close()

def plot_logprobs_vs_mean_score(results: List[Dict[str, Any]], output_dir: Path):
    avg_logprobs = [r['avg_logprob_per_token'] for r in results]
    mean_scores = [r['mean_score'] for r in results if r['mean_score'] is not None]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(mean_scores, avg_logprobs, alpha=0.7, color='pink')
    
    plt.title('Logprobs vs Mean Score')
    plt.xlabel('Mean Score')
    plt.ylabel('Average Logprob per Token')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'logprobs_vs_mean_score.png')
    plt.close()

def plot_logprobs_vs_pass_at_1(results: List[Dict[str, Any]], output_dir: Path):
    avg_logprobs = [r['avg_logprob_per_token'] for r in results]
    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(pass_at_1, avg_logprobs, alpha=0.7, color='grey')
    
    plt.title('Logprobs vs Pass@1')
    plt.xlabel('Pass@1')
    plt.ylabel('Average Logprob per Token')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'logprobs_vs_pass_at_1.png')
    plt.close()

def plot_mean_score_vs_pass_at_1(results: List[Dict[str, Any]], output_dir: Path):
    mean_scores = [r['mean_score'] for r in results if r['mean_score'] is not None]
    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(pass_at_1, mean_scores, alpha=0.7, color='blue')
    
    plt.title('Mean Score vs Pass@1')
    plt.xlabel('Pass@1')
    plt.ylabel('Mean Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mean_score_vs_pass_at_1.png')
    plt.close()


def create_plots(results: List[Dict[str, Any]], output_dir: Path):
    """Create all plots and save them to the output directory."""
    # plot_per_problem_logprobs(results, output_dir, False)
    # plot_per_problem_entropy(results, output_dir)
    # plot_per_problem_level(results, output_dir)
    # plot_pass_at_1(results, output_dir)
    # plot_mean_score(results, output_dir)
    # plot_level_mean(results, output_dir)
    # plot_level_pass(results, output_dir)
    # plot_entropy_vs_level_mean(results, output_dir)
    # plot_entropy_vs_level_pass(results, output_dir)
    # plot_logprobs_vs_level_mean(results, output_dir)
    # plot_logprobs_vs_level_pass(results, output_dir)
    # plot_entropy_vs_mean_score(results, output_dir)
    # plot_entropy_vs_pass_at_1(results, output_dir)
    # plot_logprobs_vs_mean_score(results, output_dir)
    # plot_logprobs_vs_pass_at_1(results, output_dir)
    plot_mean_score_vs_pass_at_1(results, output_dir)

def analyze_logprobs(file_path: str, num_tokens_to_analyse: int) -> List[Dict[str, Any]]:
    results = []
    
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Processing lines"):
            data = json.loads(line)
            
            # Skip if no log_probs
            if 'log_probs' not in data:
                continue
            if len(data['log_probs'][0]) < num_tokens_to_analyse:
                raise ValueError(f"log_probs list ({len(data['log_probs'])}) is shorter than num_tokens_to_analyse ({num_tokens_to_analyse})")
            log_probs = data['log_probs'][:][:num_tokens_to_analyse]
            
            # Get dimensions
            num_generations = len(log_probs)
            
            # Calculate average for each rank position across all generations and tokens
            avg_by_rank = []
            stds_by_rank = []

            for rank in range(len(log_probs[0][0])):  # For each rank position
                rank_probs = []
                for gen in range(num_generations):
                    for token in range(len(log_probs[gen])):
                        if rank < len(log_probs[gen][token]):
                            rank_probs.append(log_probs[gen][token][rank])
                avg = float(np.mean(rank_probs))
                std = float(np.std(rank_probs))
                avg_by_rank.append(avg)
                stds_by_rank.append(std)
                
            
            # Calculate entropy for each token in each generation
            entropies = []
            for gen in range(num_generations):
                for token_logprobs in log_probs[gen]:
                    # Convert log probs to probabilities
                    probs = np.exp(token_logprobs)
                    # Calculate entropy
                    entropy = float(-np.sum(probs * token_logprobs))
                    entropies.append(entropy)
            
            analysis = {
                'unique_id': data['unique_id'],
                'level': data['level'],
                'original_logprobs': log_probs,
                'max_logprobs_per_token': [max(token_probs) for gen in log_probs for token_probs in gen],
                'avg_by_rank': avg_by_rank,
                'stds_by_rank': stds_by_rank,
                'avg_logprob_per_token': float(np.mean([np.mean(probs) for gen in log_probs for probs in gen])),
                'entropies': entropies,
                'avg_entropy': float(np.mean(entropies)),
                'pass@1': data.get('pass@1', None),
                'mean_score': data.get('mean_score', None),
                'level_mean': data.get('level_mean', None),
                'level_pass': data.get('level_pass', None)
            }
            
            results.append(analysis)
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyse_logprobs.py <path_to_jsonl>")
        sys.exit(1)

    file_path = sys.argv[1]

    num_tokens_to_analyse = 15  # Default value

    if len(sys.argv) == 3:
        try:
            num_tokens_to_analyse = int(sys.argv[2])
        except ValueError:
            print("Invalid number of tokens specified. Using default value of 15.")

    if file_path.endswith('.jsonl'):
        results = analyze_logprobs(file_path, num_tokens_to_analyse)
    else:
        with open(file_path, 'r') as f:
            results = json.load(f)
    
    # Create output directory based on input file name
    output_dir = Path(file_path).parent / (Path(file_path).stem + '_analysis')
    output_dir.mkdir(exist_ok=True)
    
    # Save results to JSON file
    output_json = output_dir / 'analysis.json'
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    # Create and save plots
    create_plots(results, output_dir)
    
    # Print summary of first result as example
    if results:
        print(f"\nResults saved to: {output_json}")
        print("\nPlots saved in: {output_dir}")
        print("\nAnalysis of first entry:")
        for key, value in results[0].items():
            if key in ['original_logprobs', 'max_logprobs_per_token', 'entropies']:
                print(f"{key}: [...]")  # Skip printing
            else:
                print(f"{key}: {value}")
