import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import List, Dict, Any
import msgpack
from pathlib import Path
from scipy.interpolate import make_interp_spline

from sal.utils.score import aggregate_scores

def plot_per_problem_logprobs(results: List[Dict[str, Any]], output_dir: Path, show_std: bool = False):
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
    plt.scatter(entropies, level_means, alpha=0.7, color='blue')
    
    plt.title('Level Mean vs Entropy')
    plt.xlabel('Entropy')
    plt.ylabel('Level Mean')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'level_mean_vs_entropy.png')
    plt.close()

def plot_entropy_vs_level_pass(results: List[Dict[str, Any]], output_dir: Path):
    entropies = [r['avg_entropy'] for r in results]
    level_pass = [r['level_pass'] for r in results]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(entropies, level_pass, alpha=0.7, color='green')
    
    plt.title('Level Pass vs Entropy')
    plt.xlabel('Entropy')
    plt.ylabel('Level Pass')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'level_pass_vs_entropy.png')
    plt.close()

def plot_logprobs_vs_level_mean(results: List[Dict[str, Any]], output_dir: Path):
    avg_logprobs = [-r['avg_by_rank'][0] for r in results]
    level_means = [r['level_mean'] for r in results]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(avg_logprobs, level_means, alpha=0.7, color='red')
    
    plt.title('Level Mean vs Neg Logprobs')
    plt.xlabel('Average Neg Logprob per Token')
    plt.ylabel('Level Mean')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'level_mean_vs_logprobs.png')
    plt.close()

def plot_logprobs_vs_level_pass(results: List[Dict[str, Any]], output_dir: Path):
    avg_logprobs = [-r['avg_by_rank'][0] for r in results]
    level_pass = [r['level_pass'] for r in results]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(avg_logprobs, level_pass, alpha=0.7, color='purple')
    
    plt.title('Level Pass vs Neg Logprobs')
    plt.xlabel('Average Neg Logprob per Token')
    plt.ylabel('Level Pass')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'level_pass_vs_logprobs.png')
    plt.close()
