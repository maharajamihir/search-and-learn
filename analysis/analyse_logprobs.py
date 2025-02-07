import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Any
import msgpack
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression

from sal.utils.score import aggregate_scores
from sklearn.metrics import mean_squared_error, mean_absolute_error
import csv 

##########################################################################################################################



######### THESE ARE THE MOST RELEVANT PLOTTING FUNCTIONS FOR OUR CURRENT EVALS ###########################################

def moving_average(data, window_size):
    """Calculate the simple moving average of a list of numbers."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot_entropy_vs_pass_at_1(results: List[Dict[str, Any]], output_dir: Path):
    entropies = [np.mean([item for sublist in r['entropies'] for item in sublist]) for r in results]
    min_entropy = min(entropies)
    max_entropy = max(entropies)
    normalized_entropies = [(e - min_entropy) / (max_entropy - min_entropy) for e in entropies]
    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(pass_at_1, normalized_entropies, alpha=0.7, color='brown')

    # Calculate mean of entropies for each unique pass_at_1
    unique_pass_at_1 = np.unique(pass_at_1)
    mean_entropies_per_pass = [np.mean([entropies[i] for i in range(len(pass_at_1)) if pass_at_1[i] == pass_value]) for pass_value in unique_pass_at_1]
    
    # Apply simple moving average
    window_size = 5  # You can adjust the window size for smoothing
    if len(mean_entropies_per_pass) >= window_size:
        smooth_y = moving_average(mean_entropies_per_pass, window_size)
        smooth_x = unique_pass_at_1[(window_size - 1):]  # Adjust x to match the length of smooth_y
        plt.plot(smooth_x, smooth_y, color='red', linestyle='--', label='SMA of Entropy per Pass@1')
   
    plt.title('Pass@1 vs Entropy')
    plt.xlabel('Pass@1')
    plt.ylabel('Entropy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pass_at_1_vs_entropy.png')
    plt.close()


def plot_stddentropy_vs_pass_at_1(results: List[Dict[str, Any]], output_dir: Path):
    # Calculate standard deviation and mean of entropy for each result    
    stddentropies = [np.std([item for sublist in r['entropies'] for item in sublist]) for r in results]
    mean_entropies = [np.mean([item for sublist in r['entropies'] for item in sublist]) for r in results]
    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]
    
    # Normalize mean entropies for color mapping
    norm = plt.Normalize(min(mean_entropies), max(mean_entropies))
    cmap = plt.cm.get_cmap('coolwarm')  # Blue (low) to Red (high)
    
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(pass_at_1, stddentropies, alpha=0.7, c=mean_entropies, cmap=cmap, norm=norm)
    
    plt.title('Pass@1 vs Stddentropy (color indicates mean entropy)')
    plt.xlabel('Pass@1')
    plt.ylabel('Stddentropy')
    plt.colorbar(scatter, label='Mean Entropy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pass_at_1_vs_stddentropy.png')
    plt.close()



def plot_varentropy_vs_pass_at_1(results: List[Dict[str, Any]], output_dir: Path):
    # Calculate variance of entropy for each result    
    varentropies = [np.var([item for sublist in r['entropies'] for item in sublist]) for r in results]
    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(pass_at_1, varentropies, alpha=0.7, color='teal')
    
    plt.title('Pass@1 vs Varentropy')
    plt.xlabel('Pass@1')
    plt.ylabel('Varentropy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pass_at_1_vs_varentropy.png')
    plt.close()

def plot_varentropy_vs_mean_entropy(results: List[Dict[str, Any]], output_dir: Path):
    # Calculate variance and mean of entropy for each result
    varentropies = [np.var([item for sublist in r['entropies'] for item in sublist]) for r in results]
    mean_entropies = [np.mean([item for sublist in r['entropies'] for item in sublist]) for r in results]
    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]

    # Normalize pass_at_1 for color mapping
    norm = plt.Normalize(min(pass_at_1), max(pass_at_1))
    cmap = plt.cm.get_cmap('coolwarm')

    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(mean_entropies, varentropies, c=pass_at_1, cmap=cmap, norm=norm, alpha=0.7)

    plt.title('Mean Entropy vs Varentropy with Pass@1 Indication')
    plt.xlabel('Mean Entropy')
    plt.ylabel('Varentropy')
    plt.colorbar(scatter, label='Pass@1 Ratio')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'mean_entropy_vs_varentropy.png')
    plt.close()

def plot_logprobs_vs_pass_at_1(results: List[Dict[str, Any]], output_dir: Path):
    avg_logprobs = [-r['avg_by_rank'][0] for r in results]
    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(pass_at_1, avg_logprobs, alpha=0.7, color='grey')

    # Calculate mean of avg_logprobs for each unique pass_at_1
    unique_pass_at_1 = np.unique(pass_at_1)
    mean_logprobs_per_pass = [np.mean([avg_logprobs[i] for i in range(len(pass_at_1)) if pass_at_1[i] == pass_value]) for pass_value in unique_pass_at_1]
    
    # Apply simple moving average
    window_size = 5  # You can adjust the window size for smoothing
    if len(mean_logprobs_per_pass) >= window_size:
        smooth_y = moving_average(mean_logprobs_per_pass, window_size)
        smooth_x = unique_pass_at_1[(window_size - 1):]  # Adjust x to match the length of smooth_y
        plt.plot(smooth_x, smooth_y, color='red', linestyle='--', label='SMA of Mean Logprob per Pass@1')
    
    plt.title('Logprobs vs Pass@1')
    plt.xlabel('Pass@1')
    plt.ylabel('Average Logprob per Token')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'logprobs_vs_pass_at_1.png')
    plt.close()

def plot_pass_at_1_vs_list_len_score(results: List[Dict[str, Any]], output_dir: Path):
    pass_at_1 = [r['pass@1'] for r in results]
    list_length = [r['list_length'] for r in results]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(list_length, pass_at_1, alpha=0.7, color='orange')
    plt.xscale('log', base=2)
        
    plt.title('List Length vs Pass@1')
    plt.xlabel('List Length (Log Scale Base 2)')
    plt.ylabel('Pass@1')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'list_length_vs_pass_at_1.png')
    plt.close()

def plot_agg_scores_mean_vs_pass_at_1(results: List[Dict[str, Any]], output_dir: Path, strategy: str = "last"):
    agg_scores_mean = []
    agg_scores_std = []
    for problem in results:
        scores = problem["scores"]
        agg_score = [aggregate_scores(s, strategy) for s in scores]
        agg_score_mean = float(np.mean(agg_score))
        agg_score_std = float(np.std(agg_score))
        agg_scores_mean.append(agg_score_mean)
        agg_scores_std.append(agg_score_std)


    agg_scores_prod_mean = [r[f'agg_scores_{strategy}_mean'] for r in results if r[f'agg_scores_{strategy}_mean'] is not None]
    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]
    
    if not agg_scores_prod_mean or not pass_at_1:
        print("No data available for plotting.")
        return None

    plt.figure(figsize=(12, 6))
    plt.scatter(pass_at_1, agg_scores_prod_mean, alpha=0.7, color='blue')
    
    # Perform linear regression on the data
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    # Reshape data for sklearn
    X = np.array(agg_scores_prod_mean).reshape(-1, 1)
    y = np.array(pass_at_1)

    if y.size == 0:
        print(f"No valid agg_scores_{strategy}_mean available for regression.")
        return None

    # Normalize the target data
    scaler_y = StandardScaler()
    y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y_normalized)

    # Predict using the model
    y_pred = model.predict(X)

    # Plot the linear regression line
    # plt.plot(scaler_y.inverse_transform(y_pred.reshape(-1, 1)), agg_scores_prod_mean, color='red', linestyle='-', label='Linear Regression')

    # Calculate and print the mean squared error (L2 loss) and mean absolute error (L1 loss)
    l2_loss = mean_squared_error(y_normalized, y_pred)
    l1_loss = mean_absolute_error(y_normalized, y_pred)
    print(f"Linear Regression L2 Loss: {l2_loss}, L1 Loss: {l1_loss}")

    # Update the plot title to include the losses
    plt.title(f'Pass@1 vs Agg Scores {strategy} Mean. L2 loss: {l2_loss:.4f}, L1 loss: {l1_loss:.4f}')
    plt.xlabel('Pass@1')
    plt.ylabel(f'Agg Scores {strategy} Mean')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f'pass_at_1_vs_agg_scores_{strategy}_mean.png')
    plt.close()

##########################################################################################################################




#                 #     #  ###  ####  #### 
#                 ##   ## #   # #   # #  
#                 # # # # #   # ####  ### 
#                 #  #  # #   # #  #  #  
#                 #     #  ###  #   # #### 




######### THESE ARE MISC PLOTTING FUNCTIONS THAT MIGHT BE HELPFUL IN THE FUTURE ##########################################

def plot_entropy_vs_mean_score(results: List[Dict[str, Any]], output_dir: Path):
    entropies = [r['avg_entropy'] for r in results]
    mean_scores = [r['mean_score'] for r in results if r['mean_score'] is not None]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(entropies, mean_scores, alpha=0.7, color='orange')
        
    # Calculate mean of entropies for each unique mean_score
    unique_scores = np.unique(mean_scores)
    mean_entropies_per_score = [np.mean([entropies[i] for i in range(len(mean_scores)) if mean_scores[i] == score]) for score in unique_scores]
    
    # Apply simple moving average
    window_size = 5  # You can adjust the window size for smoothing
    if len(mean_entropies_per_score) >= window_size:
        smooth_y = moving_average(mean_entropies_per_score, window_size)
        smooth_x = unique_scores[(window_size - 1):]  # Adjust x to match the length of smooth_y
        plt.plot(smooth_x, smooth_y, color='red', linestyle='--', label='SMA of Entropy per Mean Score')
     
    plt.title('Mean Score vs Entropy')
    plt.xlabel('Entropy')
    plt.ylabel('Mean Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mean_score_vs_entropy.png')
    plt.close()

def plot_logprobs_vs_mean_score(results: List[Dict[str, Any]], output_dir: Path):
    avg_logprobs = [-r['avg_by_rank'][0] for r in results]
    mean_scores = [r['mean_score'] for r in results if r['mean_score'] is not None]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(avg_logprobs, mean_scores, alpha=0.7, color='pink')
    
    # Calculate mean of avg_logprobs for each unique mean_score
    unique_scores = np.unique(mean_scores)
    mean_logprobs_per_score = [np.mean([avg_logprobs[i] for i in range(len(mean_scores)) if mean_scores[i] == score]) for score in unique_scores]
    
    # Apply simple moving average
    window_size = 5  # You can adjust the window size for smoothing
    if len(mean_logprobs_per_score) >= window_size:
        smooth_y = moving_average(mean_logprobs_per_score, window_size)
        smooth_x = unique_scores[(window_size - 1):]  # Adjust x to match the length of smooth_y
        plt.plot(smooth_x, smooth_y, color='red', linestyle='--', label='SMA of Mean Logprob per Mean Score')
    
    plt.title('Mean Score vs Logprobs')
    plt.xlabel('Average Logprob per Token')
    plt.ylabel('Mean Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mean_score_vs_logprobs.png')
    plt.close()

def plot_difficulty_scalar(results: List[Dict[str, Any]], output_dir: Path, percentile: float = 0.01):
    # Calculate variance and mean of entropy for each result    
    varentropies = [np.var([item for sublist in r['entropies'] for item in sublist]) for r in results]
    mean_entropies = [np.mean([item for sublist in r['entropies'] for item in sublist]) for r in results]
    varentropies = np.array(varentropies)
    mean_entropies = np.array(mean_entropies)

    difficulties = np.sqrt(varentropies**2 + mean_entropies**2)

    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(pass_at_1, difficulties, alpha=0.7, color='black')

    plt.title('Pass@1 vs Difficulty')
    plt.xlabel('Pass@1')
    plt.ylabel('Difficulty')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pass_at_1_vs_difficulty.png')
    plt.close()

def plot_lowest_quartile_logprobs_vs_pass_at_1(results: List[Dict[str, Any]], output_dir: Path, percentile: float = 0.25):
    # Calculate the average of the lowest 25% of rank 0 tokens
    log_probs = [r['original_logprobs'] for r in results]
    avg_logprobs = []
    std_logprobs = []  # New list to store standard deviations
    for problem in range(len(log_probs)):
        rank_probs = []
        for gen in range(len(log_probs[problem])):
            for token in range(len(log_probs[problem][gen])):
                rank_probs.append(log_probs[problem][gen][token][0])
        rank_probs.sort()
        if int(len(rank_probs) * percentile) > 0.0:
            lowest_quartile = rank_probs[:int(len(rank_probs) * percentile)]
        else:
            lowest_quartile = [rank_probs[0]] # take the min
        avg = float(np.mean(lowest_quartile))
        std = float(np.std(lowest_quartile))
        avg_logprobs.append(avg)
        std_logprobs.append(std)  # Append standard deviation

    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]
    
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()  # Get the current axis
    ax2 = ax1.twinx()  # Create a twin axis sharing the x-axis

    ax1.scatter(pass_at_1, avg_logprobs, alpha=0.7, color='blue', label='Avg Logprob')
    # ax2.scatter(pass_at_1, std_logprobs, alpha=0.7, color='grey', label='Std Logprob')  # Plot std deviations

    # Perform linear regression on the data
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    # Reshape data for sklearn
    X = np.array(avg_logprobs).reshape(-1, 1)
    y = np.array(pass_at_1)

    if y.size == 0:
        print("No valid logprobs available for regression.")
        return None

    # Normalize the target data
    scaler_y = StandardScaler()
    y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y_normalized)

    # Predict using the model
    y_pred = model.predict(X)

    # Plot the linear regression line
    # ax1.plot(scaler_y.inverse_transform(y_pred.reshape(-1, 1)), avg_logprobs, color='black', linestyle='-', label='Linear Regression')

    # Calculate and print the mean squared error (L2 loss) and mean absolute error (L1 loss)
    l2_loss = mean_squared_error(y_normalized, y_pred)
    l1_loss = mean_absolute_error(y_normalized, y_pred)
    print(f"Linear Regression L2 Loss: {l2_loss}, L1 Loss: {l1_loss}")

    # Calculate mean of avg_logprobs and std_logprobs for each unique pass_at_1
    unique_pass_at_1 = np.unique(pass_at_1)
    mean_logprobs_per_pass = [np.mean([avg_logprobs[i] for i in range(len(pass_at_1)) if pass_at_1[i] == pass_value]) for pass_value in unique_pass_at_1]
    mean_std_per_pass = [np.mean([std_logprobs[i] for i in range(len(pass_at_1)) if pass_at_1[i] == pass_value]) for pass_value in unique_pass_at_1]  # Mean std

    # Apply simple moving average
    window_size = 5  # You can adjust the window size for smoothing
    # if len(mean_logprobs_per_pass) >= window_size:
        # smooth_y_avg = moving_average(mean_logprobs_per_pass, window_size)
        # smooth_y_std = moving_average(mean_std_per_pass, window_size)  # Smooth std
        # smooth_x = unique_pass_at_1[(window_size - 1):]  # Adjust x to match the length of smooth_y
        # ax1.plot(smooth_x, smooth_y_avg, color='red', linestyle='--', label='SMA of Mean Logprob per Pass@1')
        # ax2.plot(smooth_x, smooth_y_std, color='green', linestyle='--', label='SMA of Std Logprob per Pass@1')  # Plot smoothed std

    ax1.set_title(f'Lowest Quartile Logprobs vs Pass@1. L2 loss: {l2_loss:.4f}, L1 loss: {l1_loss:.4f}')
    ax1.set_xlabel('Pass@1')
    ax1.set_ylabel(f'Logprob per Token (Lowest {percentile*100.0}%)')
    ax2.set_ylabel('Standard Deviation of Logprob')
    ax1.grid(True)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / f'lowest_{percentile*100.}_logprobs_vs_pass_at_1.png')
    plt.close()

def _plot_lowest_quartile_logprobs_vs_pass_at_1(results: List[Dict[str, Any]], output_dir: Path, percentile: float = 0.25):
    # Calculate the average of the lowest 25% of rank 0 tokens
    log_probs = [r['original_logprobs'] for r in results]
    avg_logprobs = []
    std_logprobs = []  # New list to store standard deviations
    for problem in range(len(log_probs)):
        rank_probs = []
        for gen in range(len(log_probs[problem])):
            for token in range(len(log_probs[problem][gen])):
                rank_probs.append(log_probs[problem][gen][token][0])
        rank_probs.sort()
        if int(len(rank_probs) * percentile) > 0.0:
            lowest_quartile = rank_probs[:int(len(rank_probs) * percentile)]
        else:
            lowest_quartile = [rank_probs[0]] # take the min
        avg = float(np.mean(lowest_quartile))
        std = float(np.std(lowest_quartile))
        avg_logprobs.append(avg)
        std_logprobs.append(std)  # Append standard deviation

    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]
    
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()  # Get the current axis
    ax2 = ax1.twinx()  # Create a twin axis sharing the x-axis

    ax1.scatter(avg_logprobs, pass_at_1, alpha=0.7, color='blue', label='Pass@1')
    # ax2.scatter(avg_logprobs, std_logprobs, alpha=0.7, color='grey', label='Std Logprob')  # Plot std deviations

    # Perform linear regression on the data
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    # Reshape data for sklearn
    X = np.array(avg_logprobs).reshape(-1, 1)
    y = np.array(pass_at_1)

    if y.size == 0:
        print("No valid logprobs available for regression.")
        return None

    # Normalize the target data
    scaler_y = StandardScaler()
    y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y_normalized)

    # Predict using the model
    y_pred = model.predict(X)

    # Plot the linear regression line
    # ax1.plot(avg_logprobs, scaler_y.inverse_transform(y_pred.reshape(-1, 1)), color='black', linestyle='-', label='Linear Regression')

    # Calculate and print the mean squared error (L2 loss) and mean absolute error (L1 loss)
    l2_loss = mean_squared_error(y_normalized, y_pred)
    l1_loss = mean_absolute_error(y_normalized, y_pred)
    print(f"Linear Regression L2 Loss: {l2_loss}, L1 Loss: {l1_loss}")

    # Calculate mean of pass_at_1 and std_logprobs for each unique avg_logprob
    unique_avg_logprobs = np.unique(avg_logprobs)
    mean_pass_per_logprob = [np.mean([pass_at_1[i] for i in range(len(avg_logprobs)) if avg_logprobs[i] == logprob_value]) for logprob_value in unique_avg_logprobs]
    mean_std_per_logprob = [np.mean([std_logprobs[i] for i in range(len(avg_logprobs)) if avg_logprobs[i] == logprob_value]) for logprob_value in unique_avg_logprobs]  # Mean std

    # Apply simple moving average
    window_size = 5  # You can adjust the window size for smoothing
    if len(mean_pass_per_logprob) >= window_size:
        smooth_y_avg = moving_average(mean_pass_per_logprob, window_size)
        smooth_y_std = moving_average(mean_std_per_logprob, window_size)  # Smooth std
        smooth_x = unique_avg_logprobs[(window_size - 1):]  # Adjust x to match the length of smooth_y
        ax1.plot(smooth_x, smooth_y_avg, color='red', linestyle='--', label='SMA of Pass@1 per Logprob')
        ax2.plot(smooth_x, smooth_y_std, color='green', linestyle='--', label='SMA of Std Logprob per Logprob')  # Plot smoothed std

    ax1.set_title(f'Pass@1 vs Lowest Quartile Logprobs. L2 loss: {l2_loss:.4f}, L1 loss: {l1_loss:.4f}')
    ax1.set_xlabel(f'Logprob per Token (Lowest {percentile*100.0}%)')
    ax1.set_ylabel('Pass@1')
    ax2.set_ylabel('Standard Deviation of Logprob')
    ax1.grid(True)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / f'pass_at_1_vs_lowest_{percentile*100.}_logprobs.png')
    plt.close()

def plot_lowest_quartile_entropy_vs_pass_at_1(results: List[Dict[str, Any]], output_dir: Path, percentile: float = 0.25):
    # Calculate the average of the lowest 25% of entropy values
    entropies = [r['entropies'] for r in results]
    avg_entropies = []
    std_entropies = []  # New list to store standard deviations
    for problem in range(len(entropies)):
        entropies_lst = []
        for gen in range(len(entropies[problem])):
            for token in range(len(entropies[problem][gen])):
                entropies_lst.append(entropies[problem][gen][token])
        entropies_lst.sort()
        if int(len(entropies_lst) * percentile) > 0.0:
            lowest_quartile = entropies_lst[len(entropies_lst) - int(len(entropies_lst) * percentile):]
        else:
            lowest_quartile = [entropies_lst[0]] # take the min
        avg = float(np.mean(lowest_quartile))
        std = float(np.std(lowest_quartile))
        avg_entropies.append(avg)
        std_entropies.append(std)  # Append standard deviation

    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]
    
    if not avg_entropies or not pass_at_1:
        print("No data available for plotting.")
        return None

    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()  # Get the current axis
    ax2 = ax1.twinx()  # Create a twin axis sharing the x-axis

    ax1.scatter(pass_at_1, avg_entropies, alpha=0.7, color='orange', label='Avg Entropy')
    ax2.scatter(pass_at_1, std_entropies, alpha=0.7, color='brown', label='Std Entropy')  # Plot std deviations

    # Perform linear regression on the data
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    # Reshape data for sklearn
    X = np.array(pass_at_1).reshape(-1, 1)
    y = np.array(avg_entropies)

    if y.size == 0:
        print("No valid entropy values available for regression.")
        return None

    # Normalize the target data
    scaler_y = StandardScaler()
    y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y_normalized)

    # Predict using the model
    y_pred = model.predict(X)

    # Plot the linear regression line
    # ax1.plot(pass_at_1, scaler_y.inverse_transform(y_pred.reshape(-1, 1)), color='black', linestyle='-', label='Linear Regression')

    # Calculate and print the mean squared error (L2 loss) and mean absolute error (L1 loss)
    l2_loss = mean_squared_error(y_normalized, y_pred)
    l1_loss = mean_absolute_error(y_normalized, y_pred)
    print(f"Linear Regression L2 Loss: {l2_loss}, L1 Loss: {l1_loss}")

    # Calculate mean of avg_entropies and std_entropies for each unique pass_at_1
    unique_pass_at_1 = np.unique(pass_at_1)
    mean_entropy_per_pass = [np.mean([avg_entropies[i] for i in range(len(pass_at_1)) if pass_at_1[i] == pass_value]) for pass_value in unique_pass_at_1]
    mean_std_per_pass = [np.mean([std_entropies[i] for i in range(len(pass_at_1)) if pass_at_1[i] == pass_value]) for pass_value in unique_pass_at_1]  # Mean std

    # Apply simple moving average
    window_size = 5  # You can adjust the window size for smoothing
    if len(mean_entropy_per_pass) >= window_size:
        smooth_y_avg = moving_average(mean_entropy_per_pass, window_size)
        smooth_y_std = moving_average(mean_std_per_pass, window_size)  # Smooth std
        smooth_x = unique_pass_at_1[(window_size - 1):]  # Adjust x to match the length of smooth_y
        ax1.plot(smooth_x, smooth_y_avg, color='red', linestyle='--', label='SMA of Avg Entropy per Pass@1')
        ax2.plot(smooth_x, smooth_y_std, color='green', linestyle='--', label='SMA of Std Entropy per Pass@1')  # Plot smoothed std

    ax1.set_title(f'Pass@1 vs Lowest Quartile Entropy. L2 loss: {l2_loss:.4f}, L1 loss: {l1_loss:.4f}')
    ax1.set_xlabel('Pass@1')
    ax1.set_ylabel(f'Entropy (Lowest {int(percentile*100)}%)')
    ax2.set_ylabel('Standard Deviation of Entropy')
    ax1.grid(True)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / f'pass_at_1_vs_lowest_{percentile*100}_entropy.png')
    plt.close()

def plot_lowest_quartile_entropy_vs_seq_len(results: List[Dict[str, Any]], output_dir: Path, percentile: float = 0.25):
    # Calculate the average of the lowest 25% of entropy values
    entropies = [r['entropies'] for r in results]
    avg_entropies = []
    for problem in range(len(entropies)):
        entropies_lst = []
        for gen in range(len(entropies[problem])):
            for token in range(len(entropies[problem][gen])):
                entropies_lst.append(entropies[problem][gen][token])
        entropies_lst.sort()
        if int(len(entropies_lst) * percentile) > 0.0:
            lowest_quartile = entropies_lst[len(entropies_lst) - int(len(entropies_lst) * percentile):]
        else:
            lowest_quartile = [entropies_lst[0]] # take the min
        avg = float(np.mean(lowest_quartile))
        avg_entropies.append(avg)

    seq_len = [r['list_length'] for r in results if r['list_length'] is not None]
    
    if not avg_entropies or not seq_len:
        print("No data available for plotting.")
        return None

    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()  # Get the current axis

    ax1.scatter(seq_len, avg_entropies, alpha=0.7, color='orange', label='Avg Entropy')

    # Perform linear regression on the data
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    # Reshape data for sklearn
    X = np.array(seq_len).reshape(-1, 1)
    y = np.array(avg_entropies)

    if y.size == 0:
        print("No valid entropy values available for regression.")
        return None

    # Normalize the target data
    scaler_y = StandardScaler()
    y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y_normalized)

    # Predict using the model
    y_pred = model.predict(X)

    # Plot the linear regression line
    # ax1.plot(seq_len, scaler_y.inverse_transform(y_pred.reshape(-1, 1)), color='black', linestyle='-', label='Linear Regression')

    # Calculate and print the mean squared error (L2 loss) and mean absolute error (L1 loss)
    l2_loss = mean_squared_error(y_normalized, y_pred)
    l1_loss = mean_absolute_error(y_normalized, y_pred)
    print(f"Linear Regression L2 Loss: {l2_loss}, L1 Loss: {l1_loss}")

    # Calculate mean of avg_entropies for each unique seq_len
    unique_seq_len = np.unique(seq_len)
    mean_entropy_per_seq = [np.mean([avg_entropies[i] for i in range(len(seq_len)) if seq_len[i] == seq_value]) for seq_value in unique_seq_len]

    # Apply simple moving average
    window_size = 5  # You can adjust the window size for smoothing
    if len(mean_entropy_per_seq) >= window_size:
        smooth_y_avg = moving_average(mean_entropy_per_seq, window_size)
        smooth_x = unique_seq_len[(window_size - 1):]  # Adjust x to match the length of smooth_y
        ax1.plot(smooth_x, smooth_y_avg, color='red', linestyle='--', label='SMA of Avg Entropy per Seq Len')

    ax1.set_xscale('log', base=2)  # Set x-axis to log scale
    ax1.set_title(f'Seq Len vs Lowest Quartile Entropy. L2 loss: {l2_loss:.4f}, L1 loss: {l1_loss:.4f}')
    ax1.set_xlabel('Seq Len')
    ax1.set_ylabel(f'Entropy (Lowest {int(percentile*100)}%)')
    ax1.grid(True)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / f'seq_len_vs_lowest_{percentile*100}_entropy.png')
    plt.close()

def plot_lowest_quartile_surprise_vs_pass_at_1(results: List[Dict[str, Any]], output_dir: Path, percentile: float = 0.25):
    # Calculate the average of the lowest 25% of reverse entropy values
    surprises = [r['surprises'] for r in results]
    avg_surprises = []
    std_surprises = []  # New list to store standard deviations
    for problem in range(len(surprises)):
        surprises_lst = []
        for gen in range(len(surprises[problem])):
            for token in range(len(surprises[problem][gen])):
                surprises_lst.append(surprises[problem][gen][token])
        surprises_lst.sort()
        if int(len(surprises_lst) * percentile) > 0.0:
            lowest_quartile = surprises_lst[len(surprises_lst) - int(len(surprises_lst) * percentile):]
        else:
            lowest_quartile = [surprises_lst[0]] # take the min
        avg = float(np.mean(lowest_quartile))
        std = float(np.std(lowest_quartile))
        avg_surprises.append(avg)
        std_surprises.append(std)  # Append standard deviation

    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]
    
    if not avg_surprises or not pass_at_1:
        print("No data available for plotting.")
        return None

    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()  # Get the current axis
    ax2 = ax1.twinx()  # Create a twin axis sharing the x-axis

    ax1.scatter(pass_at_1, avg_surprises, alpha=0.7, color='orange', label='Avg Surprise')
    # ax2.scatter(pass_at_1, std_surprises, alpha=0.7, color='brown', label='Std Rev Entropy')  # Plot std deviations

    # Perform linear regression on the data
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    # Reshape data for sklearn
    X = np.array(pass_at_1).reshape(-1, 1)
    y = np.array(avg_surprises)

    if y.size == 0:
        print("No valid reverse entropy values available for regression.")
        return None

    # Normalize the target data
    scaler_y = StandardScaler()
    y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y_normalized)

    # Predict using the model
    y_pred = model.predict(X)

    # Plot the linear regression line
    # ax1.plot(pass_at_1, scaler_y.inverse_transform(y_pred.reshape(-1, 1)), color='black', linestyle='-', label='Linear Regression')

    # Calculate and print the mean squared error (L2 loss) and mean absolute error (L1 loss)
    l2_loss = mean_squared_error(y_normalized, y_pred)
    l1_loss = mean_absolute_error(y_normalized, y_pred)
    print(f"Linear Regression L2 Loss: {l2_loss}, L1 Loss: {l1_loss}")

    # Calculate mean of pass_at_1 and std_surprises for each unique avg_surprise
    unique_pass_at_1 = np.unique(pass_at_1)
    mean_surprise_per_pass = [np.mean([avg_surprises[i] for i in range(len(pass_at_1)) if pass_at_1[i] == p]) for p in unique_pass_at_1]
    mean_std_per_pass = [np.mean([std_surprises[i] for i in range(len(pass_at_1)) if pass_at_1[i] == p]) for p in unique_pass_at_1]  # Mean std

    # Apply simple moving average
    window_size = 5  # You can adjust the window size for smoothing
    if len(mean_surprise_per_pass) >= window_size:
        smooth_y_avg = moving_average(mean_surprise_per_pass, window_size)
        smooth_y_std = moving_average(mean_std_per_pass, window_size)  # Smooth std
        smooth_x = unique_pass_at_1[(window_size - 1):]  # Adjust x to match the length of smooth_y
        ax1.plot(smooth_x, smooth_y_avg, color='red', linestyle='--', label='SMA of Avg Surprise per Pass@1')
        ax2.plot(smooth_x, smooth_y_std, color='green', linestyle='--', label='SMA of Std Surprise per Pass@1')  # Plot smoothed std

    ax1.set_title(f'Pass@1 vs Lowest Quartile Rev Entropy. L2 loss: {l2_loss:.4f}, L1 loss: {l1_loss:.4f}')
    ax1.set_xlabel('Pass@1')
    ax1.set_ylabel(f'Rev Entropy (Lowest {int(percentile*100)}%)')
    ax2.set_ylabel('Standard Deviation of Rev Entropy')
    ax1.grid(True)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / f'pass_at_1_vs_lowest_{percentile*100}_surprise.png')
    plt.close()

def plot_lowest_quartile_surprise_vs_seq_len(results: List[Dict[str, Any]], output_dir: Path, percentile: float = 0.25):
    # Calculate the average of the lowest 25% of reverse entropy values
    surprises = [r['surprises'] for r in results]
    avg_surprises = []
    std_surprises = []  # New list to store standard deviations
    for problem in range(len(surprises)):
        surprises_lst = []
        for gen in range(len(surprises[problem])):
            for token in range(len(surprises[problem][gen])):
                surprises_lst.append(surprises[problem][gen][token])
        surprises_lst.sort()
        if int(len(surprises_lst) * percentile) > 0.0:
            lowest_quartile = surprises_lst[len(surprises_lst) - int(len(surprises_lst) * percentile):]
        else:
            lowest_quartile = [surprises_lst[0]] # take the min
        avg = float(np.mean(lowest_quartile))
        std = float(np.std(lowest_quartile))
        avg_surprises.append(avg)
        std_surprises.append(std)  # Append standard deviation

    seq_len = [r['list_length'] for r in results if r['list_length'] is not None]
    
    if not avg_surprises or not seq_len:
        print("No data available for plotting.")
        return None

    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()  # Get the current axis
    ax2 = ax1.twinx()  # Create a twin axis sharing the x-axis

    ax1.scatter(seq_len, avg_surprises, alpha=0.7, color='orange', label='Avg Surprise')
    # ax2.scatter(seq_len, std_surprises, alpha=0.7, color='brown', label='Std Rev Entropy')  # Plot std deviations

    # Perform linear regression on the data
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    # Reshape data for sklearn
    X = np.array(seq_len).reshape(-1, 1)
    y = np.array(avg_surprises)

    if y.size == 0:
        print("No valid reverse entropy values available for regression.")
        return None

    # Normalize the target data
    scaler_y = StandardScaler()
    y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y_normalized)

    # Predict using the model
    y_pred = model.predict(X)

    # Plot the linear regression line
    # ax1.plot(seq_len, scaler_y.inverse_transform(y_pred.reshape(-1, 1)), color='black', linestyle='-', label='Linear Regression')

    # Calculate and print the mean squared error (L2 loss) and mean absolute error (L1 loss)
    l2_loss = mean_squared_error(y_normalized, y_pred)
    l1_loss = mean_absolute_error(y_normalized, y_pred)
    print(f"Linear Regression L2 Loss: {l2_loss}, L1 Loss: {l1_loss}")

    # Calculate mean of seq_len and std_surprises for each unique avg_surprise
    unique_seq_len = np.unique(seq_len)
    mean_surprise_per_seq = [np.mean([avg_surprises[i] for i in range(len(seq_len)) if seq_len[i] == s]) for s in unique_seq_len]
    mean_std_per_seq = [np.mean([std_surprises[i] for i in range(len(seq_len)) if seq_len[i] == s]) for s in unique_seq_len]  # Mean std

    # Apply simple moving average
    window_size = 5  # You can adjust the window size for smoothing
    if len(mean_surprise_per_seq) >= window_size:
        smooth_y_avg = moving_average(mean_surprise_per_seq, window_size)
        smooth_y_std = moving_average(mean_std_per_seq, window_size)  # Smooth std
        smooth_x = unique_seq_len[(window_size - 1):]  # Adjust x to match the length of smooth_y
        ax1.plot(smooth_x, smooth_y_avg, color='red', linestyle='--', label='SMA of Avg Surprise per Seq Len')
        ax2.plot(smooth_x, smooth_y_std, color='green', linestyle='--', label='SMA of Std Surprise per Seq Len')  # Plot smoothed std

    ax1.set_xscale('log', base=2)  # Set x-axis to log scale with base 2
    ax1.set_title(f'Seq Len vs Lowest Quartile Surprise. L2 loss: {l2_loss:.4f}, L1 loss: {l1_loss:.4f}')

    ax1.set_xlabel('Seq Len (Log Scale)')
    ax1.set_ylabel(f'Surprise (Lowest {int(percentile*100)}%)')
    ax2.set_ylabel('Standard Deviation of Surprise')
    ax1.grid(True)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / f'seq_len_vs_lowest_{percentile*100}_surprise.png')
    plt.close()

def plot_mean_score_vs_pass_at_1(results: List[Dict[str, Any]], output_dir: Path):
    mean_scores = [r['mean_score'] for r in results if r['mean_score'] is not None]
    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]
    
    if not mean_scores or not pass_at_1:
        print("No data available for plotting.")
        return None

    plt.figure(figsize=(12, 6))
    plt.scatter(pass_at_1, mean_scores, alpha=0.7, color='blue')
    
    # Perform linear regression on the data
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    # Reshape data for sklearn
    X = np.array(mean_scores).reshape(-1, 1)
    y = np.array(pass_at_1)

    if y.size == 0:
        print("No valid mean scores available for regression.")
        return None

    # Normalize the target data
    scaler_y = StandardScaler()
    y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y_normalized)

    # Predict using the model
    y_pred = model.predict(X)

    # Plot the linear regression line
    plt.plot(scaler_y.inverse_transform(y_pred.reshape(-1, 1)), mean_scores, color='red', linestyle='-', label='Linear Regression')

    # Calculate and print the mean squared error (L2 loss) and mean absolute error (L1 loss)
    l2_loss = mean_squared_error(y_normalized, y_pred)
    l1_loss = mean_absolute_error(y_normalized, y_pred)
    print(f"Linear Regression L2 Loss: {l2_loss}, L1 Loss: {l1_loss}")

    # Update the plot title to include the losses
    plt.title(f'Pass@1 vs Mean Score. L2 loss: {l2_loss:.4f}, L1 loss: {l1_loss:.4f}')
    plt.xlabel('Pass@1')
    plt.ylabel('Mean Score')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'pass_at_1_vs_mean_score.png')
    plt.close()

def plot_per_token_logprobs_for_ith_problem(results: List[Dict[str, Any]], output_dir: Path, problem_idx : int = 5):
    if len(results) < problem_idx + 1:
        print(f"Not enough problems to plot the {problem_idx + 1}th one.")
        return
    
    per_token_probs_dir = output_dir / "per_token_probs"
    per_token_probs_dir.mkdir(parents=True, exist_ok=True)

    # Extract log probabilities for the i'th problem
    problem = results[problem_idx]  # 0-based index
    pass_at_1 = problem["pass@1"]
    log_probs = problem['original_logprobs']
    
    plt.figure(figsize=(12, 6))
    
    # Plot each generation's log probabilities
    for gen_index, gen_log_probs in enumerate(log_probs):
        avg_log_probs_per_token = [token_probs[0] for token_probs in gen_log_probs]
        plt.plot(avg_log_probs_per_token, label=f'Generation {gen_index + 1}')
    
    plt.title(f"Per Token Logprobs for {problem_idx}'th Problem (pass@1: {pass_at_1})")
    plt.xlabel('Token Index')
    plt.ylabel('Average Log Probability')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(per_token_probs_dir / f'per_token_logprobs_{problem_idx}th_problem_{pass_at_1:.3f}.png')
    plt.close()

def plot_lowest_cutoff_logprobs_vs_pass_at_1(results: List[Dict[str, Any]], output_dir: Path, cutoff_value: float = -6.0):
    # Calculate the average of logprobs below the cutoff value
    log_probs = [r['original_logprobs'] for r in results]
    avg_logprobs = []
    count_below_cutoff = []  # New list to store counts of values below cutoff
    for problem in range(len(log_probs)):
        rank_probs = []
        for gen in range(len(log_probs[problem])):
            for token in range(len(log_probs[problem][gen])):
                rank_probs.append(log_probs[problem][gen][token][0])
        
        # Filter values below the cutoff
        below_cutoff = [prob for prob in rank_probs if prob < cutoff_value]
        count_below_cutoff.append(len(below_cutoff))
        
        if below_cutoff:
            avg = float(np.mean(below_cutoff))
        else:
            avg = float(0.)  # Handle case where no values are below cutoff
        
        avg_logprobs.append(avg)

    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]
    
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()  # Get the current axis
    ax2 = ax1.twinx()  # Create a twin axis sharing the x-axis

    ax1.scatter(pass_at_1, avg_logprobs, alpha=0.7, color='blue', label='Avg Logprob')
    ax2.scatter(pass_at_1, count_below_cutoff, alpha=0.7, color='grey', label='Count Below Cutoff')  # Plot counts

    # Perform linear regression on the data
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler

    # Reshape data for sklearn
    X = np.array(pass_at_1).reshape(-1, 1)  # Corrected: X should be pass_at_1
    y = np.array(avg_logprobs)

    if y.size == 0:
        print("No valid logprobs available for regression.")
        return None

    # Normalize the target data
    scaler_y = StandardScaler()
    y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y_normalized)

    # Predict using the model
    y_pred = model.predict(X)

    # Plot the linear regression line
    ax1.plot(pass_at_1, scaler_y.inverse_transform(y_pred.reshape(-1, 1)), color='black', linestyle='-', label='Linear Regression')

    # Calculate and print the mean squared error (L2 loss) and mean absolute error (L1 loss)
    l2_loss = mean_squared_error(y_normalized, y_pred)
    l1_loss = mean_absolute_error(y_normalized, y_pred)
    print(f"Linear Regression L2 Loss: {l2_loss}, L1 Loss: {l1_loss}")

    # Calculate mean of pass_at_1 and count_below_cutoff for each unique avg_logprob
    unique_avg_logprobs = np.unique(avg_logprobs)
    mean_pass_per_logprob = [np.mean([pass_at_1[i] for i in range(len(avg_logprobs)) if avg_logprobs[i] == logprob_value]) for logprob_value in unique_avg_logprobs]
    mean_count_per_logprob = [np.mean([count_below_cutoff[i] for i in range(len(avg_logprobs)) if avg_logprobs[i] == logprob_value]) for logprob_value in unique_avg_logprobs]  # Mean count

    # Apply simple moving average
    window_size = 5  # You can adjust the window size for smoothing
    if len(mean_pass_per_logprob) >= window_size:
        smooth_y_avg = moving_average(mean_pass_per_logprob, window_size)
        smooth_y_count = moving_average(mean_count_per_logprob, window_size)  # Smooth count
        smooth_x = unique_avg_logprobs[(window_size - 1):]  # Adjust x to match the length of smooth_y
        ax1.plot(smooth_x, smooth_y_avg, color='red', linestyle='--', label='SMA of Pass@1 per Logprob')
        ax2.plot(smooth_x, smooth_y_count, color='green', linestyle='--', label='SMA of Count Below Cutoff per Logprob')  # Plot smoothed count

    ax1.set_title(f'Pass@1 vs Logprobs Below Cutoff. L2 loss: {l2_loss:.4f}, L1 loss: {l1_loss:.4f}')
    ax1.set_xlabel('Pass@1')
    ax1.set_ylabel(f'Logprob per Token (Below {cutoff_value})')
    ax2.set_ylabel('Count Below Cutoff')
    ax1.grid(True)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / f'pass_at_1_vs_below_cutoff_{cutoff_value}.png')
    plt.close()

#############################################################################################################################



######### THESE FUNCTIONS ARE ONLY RELEVANT FOR CALCULATING ENTROPIES OVER THE ATTENTIONAL COEFFICIENTS ##################

def plot_per_head_per_layer_entropies_scatter(results: List[Dict[str, Any]], output_dir: Path):
    """
    Plot the per head per layer entropies as scatter plots from the results dictionary.

    :param results: A list of dictionaries containing analysis data, including attention entropies.
    :param output_dir: Directory to save the plots.
    """
    # Assuming all results have the same structure for attention entropies
    attention_entropies = [r["attention_entropies"] for r in results]
    list_lengths = [r["list_length"] for r in results]

    num_layers = len(attention_entropies[0])
    num_heads = len(attention_entropies[0][0]) if num_layers > 0 else 0

    for layer_idx in range(num_layers):
        plt.figure(figsize=(10, 6))
        for head_idx in range(num_heads):
            entropies = [attn_entr[layer_idx][head_idx] for attn_entr in attention_entropies]
            print(entropies)
            plt.scatter(list_lengths, entropies, label=f'Head {head_idx + 1}', alpha=0.6)
        
        plt.title(f'Layer {layer_idx + 1} Entropies (Scatter Plot)')
        plt.xlabel('Difficulty (Sequence length)')
        plt.ylabel('Entropy')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx + 1}_entropies_scatter.png')
        plt.close()

def plot_per_head_per_layer_entropies_scatter_pass_at_1(results: List[Dict[str, Any]], output_dir: Path):
    """
    Plot the per head per layer entropies as scatter plots from the results dictionary against pass@1.

    :param results: A list of dictionaries containing analysis data, including attention entropies.
    :param output_dir: Directory to save the plots.
    """
    # Assuming all results have the same structure for attention entropies
    attention_entropies = [r["attention_entropies"] for r in results]
    pass_at_1 = [r["pass_at_1"] for r in results]

    num_layers = len(attention_entropies[0])
    num_heads = len(attention_entropies[0][0]) if num_layers > 0 else 0

    for layer_idx in range(num_layers):
        plt.figure(figsize=(10, 6))
        for head_idx in range(num_heads):
            entropies = [attn_entr[layer_idx][head_idx] for attn_entr in attention_entropies]
            plt.scatter(pass_at_1, entropies, label=f'Head {head_idx + 1}', alpha=0.6)
        
        plt.title(f'Layer {layer_idx + 1} Entropies vs Pass@1 (Scatter Plot)')
        plt.xlabel('Pass@1')
        plt.ylabel('Entropy')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx + 1}_entropies_vs_pass_at_1_scatter.png')
        plt.close()

def plot_per_head_per_layer_min_entropies_scatter(results: List[Dict[str, Any]], output_dir: Path):
    """
    Plot the per head per layer min entropies as scatter plots from the results dictionary.

    :param results: A list of dictionaries containing analysis data, including min entropies.
    :param output_dir: Directory to save the plots.
    """
    # Assuming all results have the same structure for min entropies
    min_entropies = [r["min_entropies"] for r in results]
    list_lengths = [r["list_length"] for r in results]

    num_layers = len(min_entropies[0])
    num_heads = len(min_entropies[0][0]) if num_layers > 0 else 0

    for layer_idx in range(num_layers):
        plt.figure(figsize=(10, 6))
        for head_idx in range(num_heads):
            entropies = [min_entr[layer_idx][head_idx] for min_entr in min_entropies]
            plt.scatter(list_lengths, entropies, label=f'Head {head_idx + 1}', alpha=0.6)
        
        plt.title(f'Layer {layer_idx + 1} Min Entropies (Scatter Plot)')
        plt.xlabel('Difficulty (Sequence length)')
        plt.ylabel('Min Entropy')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx + 1}_min_entropies_scatter.png')
        plt.close()

def plot_per_head_per_layer_min_entropies_scatter_pass_at_1(results: List[Dict[str, Any]], output_dir: Path):
    """
    Plot the per head per layer min entropies as scatter plots from the results dictionary against pass@1.

    :param results: A list of dictionaries containing analysis data, including min entropies.
    :param output_dir: Directory to save the plots.
    """
    # Assuming all results have the same structure for min entropies
    min_entropies = [r["min_entropies"] for r in results]
    pass_at_1 = [r["pass_at_1"] for r in results]

    num_layers = len(min_entropies[0])
    num_heads = len(min_entropies[0][0]) if num_layers > 0 else 0

    for layer_idx in range(num_layers):
        plt.figure(figsize=(10, 6))
        for head_idx in range(num_heads):
            entropies = [min_entr[layer_idx][head_idx] for min_entr in min_entropies]
            plt.scatter(pass_at_1, entropies, label=f'Head {head_idx + 1}', alpha=0.6)
        
        plt.title(f'Layer {layer_idx + 1} Min Entropies vs Pass@1 (Scatter Plot)')
        plt.xlabel('Pass@1')
        plt.ylabel('Min Entropy')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / f'layer_{layer_idx + 1}_min_entropies_vs_pass_at_1_scatter.png')
        plt.close()

def _compute_attention_entropy(attention_weights, head_index = 0):
    # attention_weights: (batch_size, num_heads, seq_len, seq_len)
    # Select a specific head, e.g., head_index = 0

    num_layers = len(attention_weights[0])
    num_heads = len(attention_weights[0][0])

    attention_entropies = [[0.0 for _ in range(num_heads)] for _ in range(num_layers)]

    for layer in range(num_layers):
        for head in range(num_heads):
            entropies = []
            for run in range(len(attention_weights)):
                attn_coefficients = attention_weights[run][layer][head]
                attention_probs = np.array([np.exp(token) / np.sum(np.exp(attn_coefficients)) for token in attn_coefficients])  # (batch_size, seq_len, seq_len)
                # Compute entropy for each token's attention distribution
                entropies.append(-np.sum(attention_probs * np.log(attention_probs + 1e-9)))  # (batch_size, seq_len)
            attention_entropies[layer][head] = np.mean(entropies)




    return attention_entropies

def _compute_attention_min_entropy(attention_weights, head_index = 0):
    # attention_weights: (batch_size, num_heads, seq_len, seq_len)
    # Select a specific head, e.g., head_index = 0

    num_layers = len(attention_weights[0])
    num_heads = len(attention_weights[0][0])

    attention_entropies = [[0.0 for _ in range(num_heads)] for _ in range(num_layers)]

    for layer in range(num_layers):
        for head in range(num_heads):
            entropies = []
            for run in range(len(attention_weights)):
                attn_coefficients = attention_weights[run][layer][head]
                attention_probs = np.array([np.exp(token) / np.sum(np.exp(attn_coefficients)) for token in attn_coefficients])  # (batch_size, seq_len, seq_len)
                # Compute entropy for each token's attention distribution
                entropies.append(-np.log(np.max(attention_probs)))  # (batch_size, seq_len)
            attention_entropies[layer][head] = np.mean(entropies)

    return attention_entropies

def analyse_attentional_coeff(file_path: str, num_tokens_to_analyse: int) -> List[Dict[str, Any]]:
    results = []

    with open(file_path, 'r') as file:
        for line in tqdm(file):
            data = json.loads(line)
            log_probs = data.get('log_probs', [])
            attentions = data.get('attentions', [])
            attn_entropies = _compute_attention_entropy(attentions)
            min_entropies = _compute_attention_min_entropy(attentions)
            pass_at_1 = 0.
            if "completions" in data.keys():
                num_correct = sum([1 if str(data["answer"]) in compl else 0 for compl in data["completions"]])
                pass_at_1 = num_correct/len(data["completions"])
            analysis = {
                'unique_id': data.get('unique_id', None),
                'problem': data.get('problem', None),
                'original_logprobs': log_probs,
                'list_length': data.get('list_length', None),
                'attention_entropies': attn_entropies,
                'min_entropies': min_entropies,
                'pass@1': pass_at_1,
            }

            results.append(analysis)

    return results




##########################################################################################################################

#                 #     #  ###  ####  #### 
#                 ##   ## #   # #   # #  
#                 # # # # #   # ####  ### 
#                 #  #  # #   # #  #  #  
#                 #     #  ###  #   # #### 

##########################################################################################################################

def plot_agg_scores_mean_vs_pass_at_1_l1_l2_loss(results: List[Dict[str, Any]], output_dir: Path, strategy: str = "last"):
    agg_scores_mean = []
    agg_scores_std = []
    for problem in results:
        scores = problem["scores"]
        agg_score = [aggregate_scores(s, strategy) for s in scores]
        agg_score_mean = float(np.mean(agg_score))
        agg_score_std = float(np.std(agg_score))
        agg_scores_mean.append(agg_score_mean)
        agg_scores_std.append(agg_score_std)

    agg_scores_prod_mean = [r[f'agg_scores_{strategy}_mean'] for r in results if r[f'agg_scores_{strategy}_mean'] is not None]
    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]
    
    if not agg_scores_prod_mean or not pass_at_1:
        print("No data available for plotting.")
        return None

    plt.figure(figsize=(12, 6))
    plt.scatter(pass_at_1, agg_scores_prod_mean, alpha=0.7, color='blue')
    
    # Add a new line from the bottom left corner to the top right
    plt.plot([0, 1], [0, 1], 'g--', label='Diagonal Line')

    # Calculate L1 and L2 loss
    ground_truth_y = np.array(pass_at_1)  # Adjusted to match the diagonal line
    l2_loss = mean_squared_error(ground_truth_y, agg_scores_prod_mean)
    l1_loss = mean_absolute_error(ground_truth_y, agg_scores_prod_mean)
    
    # Update the plot title to include the losses
    plt.title(f'Pass@1 vs Agg Scores {strategy} Mean. L2 loss: {l2_loss:.4f}, L1 loss: {l1_loss:.4f}')
    plt.xlabel('Pass@1')
    plt.ylabel(f'Agg Scores {strategy} Mean')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f'pass_at_1_vs_agg_scores_{strategy}_mean.png')
    plt.close()



def plot_varentropy_vs_pass_at_1_regression_l1_l2_loss(results: List[Dict[str, Any]], output_dir: Path):
    # Calculate variance of entropy for each result    
    varentropies = [np.var([item for sublist in r['entropies'] for item in sublist]) for r in results]
    mean_entropies = [np.mean([item for sublist in r['entropies'] for item in sublist]) for r in results]
    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]
    
    # Normalize mean entropies for color mapping
    norm = plt.Normalize(min(mean_entropies), max(mean_entropies))
    cmap = plt.cm.get_cmap('coolwarm')  # Blue (low) to Red (high)
    
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(pass_at_1, varentropies, alpha=0.7, c=mean_entropies, cmap=cmap, norm=norm)
    
    # Add ground truth line from (0,1) to (1,0)
    plt.plot([0, 1], [1, 0], 'g--', label='Ground Truth Line')
    
    # Calculate L1 and L2 loss
    ground_truth_y = 1 - np.array(pass_at_1)
    l2_loss = mean_squared_error(ground_truth_y, varentropies)
    l1_loss = mean_absolute_error(ground_truth_y, varentropies)
    
    # Update the plot title to include the losses
    plt.title(f'Pass@1 vs Varentropy (color indicates mean entropy). L2 loss: {l2_loss:.4f}, L1 loss: {l1_loss:.4f}')
    plt.xlabel('Pass@1')
    plt.ylabel('Varentropy')
    plt.colorbar(scatter, label='Mean Entropy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pass_at_1_vs_varentropy.png')
    plt.close()

    return l1_loss, l2_loss


def plot_entropy_vs_pass_at_1_regression_l1_l2_loss(results: List[Dict[str, Any]], output_dir: Path):
    entropies = [np.mean([item for sublist in r['entropies'] for item in sublist]) for r in results]
    min_entropy = min(entropies)
    max_entropy = max(entropies)
    
    # Normalize entropies for color mapping
    normalized_entropies = [(e - min_entropy) / (max_entropy - min_entropy) for e in entropies]
    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]
    
    # Normalize mean entropies for color mapping
    norm = plt.Normalize(min(entropies), max(entropies))
    cmap = plt.cm.get_cmap('coolwarm')  # Blue (low) to Red (high)
    
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(pass_at_1, normalized_entropies, alpha=0.7, c=entropies, cmap=cmap, norm=norm)

    # Add ground truth line from (0,1) to (1,0)
    plt.plot([0, 1], [1, 0], 'g--', label='Ground Truth Line')
    
    # Calculate L1 and L2 loss
    ground_truth_y = 1 - np.array(pass_at_1)
    l2_loss = mean_squared_error(ground_truth_y, normalized_entropies)
    l1_loss = mean_absolute_error(ground_truth_y, normalized_entropies)
    
    # Update the plot title to include the losses
    plt.title(f'Pass@1 vs Entropy (color indicates mean entropy). L2 loss: {l2_loss:.4f}, L1 loss: {l1_loss:.4f}')
    plt.xlabel('Pass@1')
    plt.ylabel('Entropy')
    plt.colorbar(scatter, label='Mean Entropy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pass_at_1_vs_entropy.png')
    plt.close()

    return l1_loss, l2_loss

def plot_stddentropy_vs_pass_at_1_regression_l1_l2_loss(results: List[Dict[str, Any]], output_dir: Path):
    # Calculate standard deviation and mean of entropy for each result    
    stddentropies = [np.std([item for sublist in r['entropies'] for item in sublist]) for r in results]
    mean_entropies = [np.mean([item for sublist in r['entropies'] for item in sublist]) for r in results]
    pass_at_1 = [r['pass@1'] for r in results] 
    
    # Normalize mean entropies for color mapping
    norm = plt.Normalize(min(mean_entropies), max(mean_entropies))
    cmap = plt.cm.get_cmap('coolwarm')  # Blue (low) to Red (high)
    
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(pass_at_1, stddentropies, alpha=0.7, c=mean_entropies, cmap=cmap, norm=norm)
    
    # Add ground truth line from (0,1) to (1,0)
    plt.plot([0, 1], [1, 0], 'g--', label='Ground Truth Line')
    
    # Calculate L1 and L2 loss
    ground_truth_y = 1 - np.array(pass_at_1)
    l2_loss = mean_squared_error(ground_truth_y, stddentropies)
    l1_loss = mean_absolute_error(ground_truth_y, stddentropies)
    
    # # Update the plot title to include the losses
    plt.title(f'Pass@1 vs Stddentropy (color indicates mean entropy). L2 loss: {l2_loss:.4f}, L1 loss: {l1_loss:.4f}')
    plt.xlabel('Pass@1')
    plt.ylabel('Stddentropy')
    plt.colorbar(scatter, label='Mean Entropy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pass_at_1_vs_stddentropy.png')
    plt.close()

    return l1_loss, l2_loss




def get_loss_varentropy_vs_pass_at_1_regression_l1_l2_loss(results: List[Dict[str, Any]]):
    # Calculate variance of entropy for each result    
    varentropies = [np.var([item for sublist in r['entropies'] for item in sublist]) for r in results]
    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]
    
    # Calculate L1 and L2 loss
    ground_truth_y = 1 - np.array(pass_at_1)
    l2_loss = mean_squared_error(ground_truth_y, varentropies)
    l1_loss = mean_absolute_error(ground_truth_y, varentropies)

    return {'l1_loss': l1_loss, 'l2_loss': l2_loss}


def get_loss_entropy_vs_pass_at_1_regression_l1_l2_loss(results: List[Dict[str, Any]]):
    entropies = [np.mean([item for sublist in r['entropies'] for item in sublist]) for r in results]
    min_entropy = min(entropies)
    max_entropy = max(entropies)
    
    # Normalize entropies for color mapping
    normalized_entropies = [(e - min_entropy) / (max_entropy - min_entropy) for e in entropies]
    pass_at_1 = [r['pass@1'] for r in results if r['pass@1'] is not None]
   
    # Calculate L1 and L2 loss
    ground_truth_y = 1 - np.array(pass_at_1)
    l2_loss = mean_squared_error(ground_truth_y, normalized_entropies)
    l1_loss = mean_absolute_error(ground_truth_y, normalized_entropies)
    
    return {'l1_loss': l1_loss, 'l2_loss': l2_loss}


def get_loss_stddentropy_vs_pass_at_1_regression_l1_l2_loss(results: List[Dict[str, Any]]):
    # Calculate standard deviation and mean of entropy for each result    
    stddentropies = [np.std([item for sublist in r['entropies'] for item in sublist]) for r in results]
    pass_at_1 = [r['pass@1'] for r in results] 
    # Calculate L1 and L2 loss
    ground_truth_y = 1 - np.array(pass_at_1)
    l2_loss = mean_squared_error(ground_truth_y, stddentropies)
    l1_loss = mean_absolute_error(ground_truth_y, stddentropies)

    return {'l1_loss': l1_loss, 'l2_loss': l2_loss}


def plot_losses(entropy_losses, varentropy_losses, stddentropy_losses, output_dir: Path):
    sample_sizes = sorted(entropy_losses.keys())

    # Extract L1 and L2 losses for each sample size
    entropy_l1 = [entropy_losses[sample]['l1_loss'] for sample in sample_sizes]
    entropy_l2 = [entropy_losses[sample]['l2_loss'] for sample in sample_sizes]
    
    varentropy_l1 = [varentropy_losses[sample]['l1_loss'] for sample in sample_sizes]
    varentropy_l2 = [varentropy_losses[sample]['l2_loss'] for sample in sample_sizes]
    
    stddentropy_l1 = [stddentropy_losses[sample]['l1_loss'] for sample in sample_sizes]
    stddentropy_l2 = [stddentropy_losses[sample]['l2_loss'] for sample in sample_sizes]

    # save the losses to a json file
    print('saving losses to json file to directory and file name ', output_dir / f'entropy_losses.json')
    with open(output_dir / f'entropy_losses.json', 'w') as f:
        json.dump({'entropy_l1': entropy_l1, 'entropy_l2': entropy_l2, 'varentropy_l1': varentropy_l1, 'varentropy_l2': varentropy_l2, 'stddentropy_l1': stddentropy_l1, 'stddentropy_l2': stddentropy_l2}, f)  

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot and save for entropy losses
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(sample_sizes, entropy_l1, 'g-', label='Entropy L1 Loss')
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Number of Samples (log scale)')
    ax1.set_ylabel('L1 Loss', color='g')
    ax1.set_title('Entropy Losses')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(sample_sizes, entropy_l2, 'm-', label='Entropy L2 Loss')
    ax2.set_ylabel('L2 Loss', color='m')

    fig.tight_layout()
    plt.savefig(output_dir / f'entropy_losses_plot.png')
    plt.close()

    # Plot and save for varentropy losses
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(sample_sizes, varentropy_l1, 'g-', label='Varentropy L1 Loss')
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Number of Samples (log scale)')
    ax1.set_ylabel('L1 Loss', color='g')
    ax1.set_title('Varentropy Losses')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(sample_sizes, varentropy_l2, 'm-', label='Varentropy L2 Loss')
    ax2.set_ylabel('L2 Loss', color='m')

    fig.tight_layout()
    plt.savefig(output_dir / f'varentropy_losses_plot.png')
    plt.close()

    # Plot and save for stddentropy losses
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(sample_sizes, stddentropy_l1, 'g-', label='Stddentropy L1 Loss')
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Number of Samples (log scale)')
    ax1.set_ylabel('L1 Loss', color='g')
    ax1.set_title('Stddentropy Losses')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(sample_sizes, stddentropy_l2, 'm-', label='Stddentropy L2 Loss')
    ax2.set_ylabel('L2 Loss', color='m')

    fig.tight_layout()
    plt.savefig(output_dir / f'stddentropy_losses_plot.png')
    plt.close()

# Example usage
# plot_losses(entropy_losses, varentropy_losses, stddentropy_losses, Path('data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions_analysis'))


def loop_through_n_samples(results: List[Dict[str, Any]], num_tokens_to_analyse: int = 15, ) -> Dict[str, Dict[str, float]]:

    entropy_losses = {} 
    varentropy_losses = {} 
    stddentropy_losses = {} 
    
    # sample_sizes = [2**n for n in range(8)]
    sample_sizes = [16, 32]
    for n_samples_to_analyse in sample_sizes:
        print(n_samples_to_analyse)
        for i in tqdm(range(len(results))):
            n_samples_to_analyse = min(n_samples_to_analyse, len(results[i]["original_logprobs"]))
            results[i]["original_logprobs"] = results[i]["original_logprobs"][:n_samples_to_analyse]
            results[i]["surprises"] = results[i]["surprises"][:n_samples_to_analyse]
            results[i]["entropies"] = results[i]["entropies"][:n_samples_to_analyse]
            for idx in range(len(results[i]["original_logprobs"])):
                results[i]["original_logprobs"][idx] = results[i]["original_logprobs"][idx][:num_tokens_to_analyse]
                results[i]["surprises"][idx] = results[i]["surprises"][idx][:num_tokens_to_analyse]
                results[i]["entropies"][idx] = results[i]["entropies"][idx][:num_tokens_to_analyse]
        # create and save plots
        stddentropy_losses[n_samples_to_analyse] = get_loss_stddentropy_vs_pass_at_1_regression_l1_l2_loss(results)
        entropy_losses[n_samples_to_analyse] = get_loss_entropy_vs_pass_at_1_regression_l1_l2_loss(results)
        varentropy_losses[n_samples_to_analyse] = get_loss_varentropy_vs_pass_at_1_regression_l1_l2_loss(results)

    print(stddentropy_losses)
    print(entropy_losses)
    print(varentropy_losses)



    plot_losses(entropy_losses, varentropy_losses, stddentropy_losses, Path('data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions_analysis'))

   
    



####### THESE ARE OUR ENTRYPOINTS INTO ANALYSIS AND PLOTTING ###########################################################

def create_plots(results: List[Dict[str, Any]], output_dir: Path):
    """Create all plots and save them to the output directory."""
    # List of plotting functions with their names
    plot_functions = [
        # ("plot_entropy_vs_mean_score", plot_entropy_vs_mean_score),
        # ("plot_entropy_vs_pass_at_1", plot_entropy_vs_pass_at_1),
        # ("plot_logprobs_vs_mean_score", plot_logprobs_vs_mean_score),
        # ("plot_logprobs_vs_pass_at_1", plot_logprobs_vs_pass_at_1),
        # ("plot_varentropy_vs_pass_at_1", plot_varentropy_vs_pass_at_1),
        # ("plot_mean_score_vs_pass_at_1", plot_mean_score_vs_pass_at_1),
        # ("plot_per_token_logprobs_for_ith_problem", plot_per_token_logprobs_for_ith_problem),
        # ("plot_lowest_quartile_logprobs_vs_pass_at_1", plot_lowest_quartile_logprobs_vs_pass_at_1),
        # ("plot_lowest_quartile_entropy_vs_pass_at_1", plot_lowest_quartile_entropy_vs_pass_at_1),
        # ("plot_difficulty_scalar", plot_difficulty_scalar),
        # ("plot_varentropy_vs_mean_entropy", plot_varentropy_vs_mean_entropy),
        # ("plot_per_head_per_layer_entropies_scatter", plot_per_head_per_layer_entropies_scatter),
        # ("plot_pass_at_1_vs_list_len_score", plot_pass_at_1_vs_list_len_score),
        # ("plot_per_head_per_layer_entropies_vs_pass_at_1", plot_per_head_per_layer_entropies_scatter_pass_at_1),
        # ("plot_per_head_per_layer_min_entropies_scatter", plot_per_head_per_layer_min_entropies_scatter),
        # ("plot_per_head_per_layer_min_entropies_scatter_pass_at_1", plot_per_head_per_layer_min_entropies_scatter_pass_at_1),
        ("plot_stddentropy_vs_pass_at_1", plot_stddentropy_vs_pass_at_1),

    ]


    # Iterate over the functions and call them with tqdm
    for name, func in tqdm(plot_functions, desc="Generating plots"):
        # print(f"Calling {name}...")
        func(results, output_dir)
    print("Plotting lowest quartile logprobs vs pass@1 with threshold 0.2")
    plot_lowest_quartile_logprobs_vs_pass_at_1(results, output_dir, 0.2)
    print("Plotting lowest quartile logprobs vs pass@1 with threshold 0.1")
    plot_lowest_quartile_logprobs_vs_pass_at_1(results, output_dir, 0.1)
    print("Plotting lowest quartile logprobs vs pass@1 with threshold 0.05")
    plot_lowest_quartile_logprobs_vs_pass_at_1(results, output_dir, 0.05)
    print("Plotting lowest quartile logprobs vs pass@1 with threshold 0.01")
    plot_lowest_quartile_logprobs_vs_pass_at_1(results, output_dir, 0.01)
    print("Plotting lowest quartile logprobs vs pass@1 with threshold 0.005")
    plot_lowest_quartile_logprobs_vs_pass_at_1(results, output_dir, 0.005)
    print("Plotting lowest quartile logprobs vs pass@1 with threshold 0.001")
    plot_lowest_quartile_logprobs_vs_pass_at_1(results, output_dir, 0.001)
    print("Plotting lowest quartile logprobs vs pass@1 with threshold 0.0005")
    plot_lowest_quartile_logprobs_vs_pass_at_1(results, output_dir, 0.0005)
    print("Plotting lowest quartile logprobs vs pass@1 with threshold 0.0")
    plot_lowest_quartile_logprobs_vs_pass_at_1(results, output_dir, 0.0)
    print("Plotting lowest quartile logprobs vs pass@1 with threshold 1.0")
    plot_lowest_quartile_logprobs_vs_pass_at_1(results, output_dir, 1.0)
    print("Plotting lowest quartile reverse entropy vs pass@1 with threshold 0.2")
    plot_lowest_quartile_surprise_vs_pass_at_1(results, output_dir, 0.2)
    print("Plotting lowest quartile reverse entropy vs pass@1 with threshold 0.1")
    plot_lowest_quartile_surprise_vs_pass_at_1(results, output_dir, 0.1)
    print("Plotting lowest quartile reverse entropy vs pass@1 with threshold 0.05")
    plot_lowest_quartile_surprise_vs_pass_at_1(results, output_dir, 0.05)
    print("Plotting lowest quartile reverse entropy vs pass@1 with threshold 0.01")
    plot_lowest_quartile_surprise_vs_pass_at_1(results, output_dir, 0.01)
    print("Plotting lowest quartile reverse entropy vs pass@1 with threshold 0.005")
    plot_lowest_quartile_surprise_vs_pass_at_1(results, output_dir, 0.005)
    print("Plotting lowest quartile reverse entropy vs pass@1 with threshold 0.001")
    plot_lowest_quartile_surprise_vs_pass_at_1(results, output_dir, 0.001)
    print("Plotting lowest quartile reverse entropy vs pass@1 with threshold 0.0")
    plot_lowest_quartile_surprise_vs_pass_at_1(results, output_dir, 1.)
    print("Plotting lowest quartile reverse entropy vs pass@1 with threshold 0.2")
    plot_lowest_quartile_entropy_vs_pass_at_1(results, output_dir, 0.2)
    print("Plotting lowest quartile reverse entropy vs pass@1 with threshold 0.1")
    plot_lowest_quartile_entropy_vs_pass_at_1(results, output_dir, 0.1)
    print("Plotting lowest quartile reverse entropy vs pass@1 with threshold 0.05")
    plot_lowest_quartile_entropy_vs_pass_at_1(results, output_dir, 0.05)
    print("Plotting lowest quartile reverse entropy vs pass@1 with threshold 0.01")
    plot_lowest_quartile_entropy_vs_pass_at_1(results, output_dir, 0.01)
    print("Plotting lowest quartile reverse entropy vs pass@1 with threshold 0.005")
    plot_lowest_quartile_entropy_vs_pass_at_1(results, output_dir, 0.005)
    print("Plotting lowest quartile reverse entropy vs pass@1 with threshold 0.001")
    plot_lowest_quartile_entropy_vs_pass_at_1(results, output_dir, 0.001)
    print("Plotting lowest quartile reverse entropy vs pass@1 with threshold 0.0")
    plot_lowest_quartile_entropy_vs_pass_at_1(results, output_dir, 1.)

    print("Plotting entropy vs sequence length")
    plot_lowest_quartile_entropy_vs_seq_len(results, output_dir, 1.)
    print("Plotting surprise vs sequence length")
    plot_lowest_quartile_surprise_vs_seq_len(results, output_dir, 1.)
    print("Plotting entropy vs sequence length")
    plot_lowest_quartile_entropy_vs_seq_len(results, output_dir, 0.)
    print("Plotting surprise vs sequence length")
    plot_lowest_quartile_surprise_vs_seq_len(results, output_dir, 0.)

    print("Plotting entropy vs sequence length")
    plot_lowest_quartile_entropy_vs_seq_len(results, output_dir, 0.5)
    print("Plotting surprise vs sequence length")
    plot_lowest_quartile_surprise_vs_seq_len(results, output_dir, 0.5)
    print("Plotting entropy vs sequence length")
    plot_lowest_quartile_entropy_vs_seq_len(results, output_dir, 0.33)
    print("Plotting surprise vs sequence length")
    plot_lowest_quartile_surprise_vs_seq_len(results, output_dir, 0.33)

    plot_agg_scores_mean_vs_pass_at_1(results, output_dir, "last")
    plot_agg_scores_mean_vs_pass_at_1(results, output_dir, "min")
    plot_agg_scores_mean_vs_pass_at_1(results, output_dir, "prod")

    # for i in tqdm(range(100), desc="Plotting per token logprobs"):
        # plot_per_token_logprobs_for_ith_problem(results, output_dir, i)

    # plot_lowest_cutoff_logprobs_vs_pass_at_1(results, output_dir, -4)
    # plot_lowest_cutoff_logprobs_vs_pass_at_1(results, output_dir, -5)
    # plot_lowest_cutoff_logprobs_vs_pass_at_1(results, output_dir, -6)
    # plot_lowest_cutoff_logprobs_vs_pass_at_1(results, output_dir, -7)
    # plot_lowest_cutoff_logprobs_vs_pass_at_1(results, output_dir, -8)
    # plot_lowest_cutoff_logprobs_vs_pass_at_1(results, output_dir, -9)

def analyze_logprobs(file_path: str, num_tokens_to_analyse: int) -> List[Dict[str, Any]]:
    results = []
    
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Processing lines"):
            data = json.loads(line)
            # Skip if no log_probs
            if 'log_probs' not in data:
                continue

            scores = data["scores"]
            agg_scores_last = [aggregate_scores(s, "last") for s in scores]
            agg_scores_min = [aggregate_scores(s, "min") for s in scores]
            agg_scores_prod = [aggregate_scores(s, "prod") for s in scores]

            log_probs = data['log_probs']
            if num_tokens_to_analyse:
                if len(data['log_probs'][0]) < num_tokens_to_analyse:
                    raise ValueError(f"log_probs list ({len(data['log_probs'])}) is shorter than num_tokens_to_analyse ({num_tokens_to_analyse})")
                for idx, generation in enumerate(log_probs):
                    log_probs[idx] = generation[:num_tokens_to_analyse]


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
            entropies = [[] for _ in range(num_generations)]
            surprises = [[] for _ in range(num_generations)]
            for gen in range(num_generations):
                for token_logprobs in log_probs[gen]:
                    # Convert log probs to probabilities
                    probs = np.exp(token_logprobs)
                    # Calculate entropy
                    entropy = float(-np.sum(probs * token_logprobs))
                    surprise = float(-(1. - probs[0]) * token_logprobs[0])
                    entropies[gen].append(entropy)
                    surprises[gen].append(surprise)

            pass_at_1 = data.get("pass@1", None)
            if "list_length" in data.keys():
                num_correct = sum([1 if str(data["answer"]) in compl else 0 for compl in data["completions"]])
                pass_at_1 = num_correct/len(data["completions"])

            # Calculate difficulty as the standard deviation of the entropies
            difficulty =  np.std([item for sublist in entropies for item in sublist])

            # set all difficulties greater than 0.8 to zero
            difficulty = 0.0 if difficulty > 0.8 else difficulty


            analysis = {
                'unique_id': data.get('unique_id', None),
                'problem': data.get('problem', None),
                'level': data.get('level', None),
                'list_length': data.get('seq_len', None),
                'original_logprobs': log_probs,
                'max_logprobs_per_token': [max(token_probs) for gen in log_probs for token_probs in gen],
                'avg_by_rank': avg_by_rank,
                'stds_by_rank': stds_by_rank,
                'avg_logprob_per_token': float(np.mean([np.mean(probs) for gen in log_probs for probs in gen])),
                'entropies': entropies,
                'surprises': surprises,
                'avg_entropy': float(np.mean([item for sublist in entropies for item in sublist])),
                'difficulty': difficulty,
                'pass@1': pass_at_1,
                'mean_score': data.get('mean_score', None),
                'level_mean': data.get('level_mean', None),
                'level_pass': data.get('level_pass', None),
                'scores': data.get('scores', None),
                'agg_scores_min_mean': float(np.mean(agg_scores_min)),
                'agg_scores_min_std': float(np.std(agg_scores_min)),
                'agg_scores_prod_mean': float(np.mean(agg_scores_prod)),
                'agg_scores_prod_std': float(np.std(agg_scores_prod)),
                'agg_scores_last_mean': float(np.mean(agg_scores_last)),
                'agg_scores_last_std': float(np.std(agg_scores_last))
            }
            
            results.append(analysis)
    
    
    mean_difficulty = np.mean([result['difficulty'] for result in results])
    for i, result in enumerate(results):
        results[i]['difficulty'] = mean_difficulty
        # print(results[i]['difficulty'])

    return results


# def combined_plot(best_of_n_completions_dir: Path):
    

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import msgpack
    import json
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Analyze log probabilities from a JSONL file.")
    parser.add_argument("--file_path", type=str, help="Path to the JSONL file.")
    parser.add_argument("--num_tokens", type=int, default=None, help="Number of tokens to analyze.")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to analyze.")
    parser.add_argument("--loop_samples", action='store_true', help="Run loop through n_samples to calculate losses.")
    parser.add_argument("--temperature", type=str, default='default', help="Temperature to use for the analysis.")


    args = parser.parse_args()

    file_path = args.file_path
    num_tokens_to_analyse = args.num_tokens
    n_samples_to_analyse = args.n_samples

    print('Starting analysis...')

    if num_tokens_to_analyse is not None:
        print(f"Using {num_tokens_to_analyse} tokens to analyze")
       
    if n_samples_to_analyse is not None:
        print(f"Using {n_samples_to_analyse} samples to analyze")

    if file_path.endswith('.jsonl'):
        if Path(file_path).stem.startswith('best_of_n'):
            results = analyze_logprobs(file_path, num_tokens_to_analyse)
        else:
            results = analyse_attentional_coeff(file_path, num_tokens_to_analyse)
        # Create output directory based on input file name
        output_dir = Path(file_path).parent / (Path(file_path).stem + '_analysis')
        # Save results to JSON file
        output_dir.mkdir(exist_ok=True)
        output_msgpack = output_dir / f'analysis.msgpack'
        with open(output_msgpack, 'wb') as f:
            packed = msgpack.packb(results)
            f.write(packed)
        print(f"Results saved to: {output_msgpack}")
    else:
        print(f"Loading {file_path}, this might take a while...")
        file_size_in_bytes = Path(file_path).stat().st_size
        if file_size_in_bytes >= 1e9:
            file_size = file_size_in_bytes / 1e9
            size_unit = "GB"
        else:
            file_size = file_size_in_bytes / 1e3
            size_unit = "KB"
        print(f"The size of the file is: {file_size:.2f} {size_unit}")
        output_dir = Path(file_path).parent
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                results = json.load(f)
            # Save results to msgpack file
            output_msgpack = output_dir / f'analysis.msgpack'
            with open(output_msgpack, 'wb') as f:
                packed = msgpack.packb(results)
                f.write(packed)
            print(f"Results saved to: {output_msgpack}")
        elif file_path.endswith('.msgpack'):
            with open(file_path, 'rb') as f:
                results = msgpack.unpackb(f.read())

        
        if args.loop_samples:
            print("Looping through n_samples to analyse")
            entropy_losses = {} 
            varentropy_losses = {} 
            stddentropy_losses = {} 

            for n_samples_to_analyse in reversed([2**n for n in range(0, 8)]):
            # for n_samples_to_analyse in reversed(range(1, 128, 10)):
                print('Analysing n_samples:', n_samples_to_analyse)

                for i in tqdm(range(len(results))):
                    n_samples_to_analyse = min(n_samples_to_analyse, len(results[i]["original_logprobs"]))
                    results[i]["original_logprobs"] = results[i]["original_logprobs"][:n_samples_to_analyse]
                    results[i]["surprises"] = results[i]["surprises"][:n_samples_to_analyse]
                    results[i]["entropies"] = results[i]["entropies"][:n_samples_to_analyse]

                    for idx in range(len(results[i]["original_logprobs"])):
                        results[i]["original_logprobs"][idx] = results[i]["original_logprobs"][idx][:num_tokens_to_analyse]
                        results[i]["surprises"][idx] = results[i]["surprises"][idx][:num_tokens_to_analyse]
                        results[i]["entropies"][idx] = results[i]["entropies"][idx][:num_tokens_to_analyse]

                stddentropy_losses[n_samples_to_analyse] = get_loss_stddentropy_vs_pass_at_1_regression_l1_l2_loss(results)
                entropy_losses[n_samples_to_analyse] = get_loss_entropy_vs_pass_at_1_regression_l1_l2_loss(results)
                varentropy_losses[n_samples_to_analyse] = get_loss_varentropy_vs_pass_at_1_regression_l1_l2_loss(results)   
            
            file_path = Path(file_path).parent 
            plot_losses(entropy_losses, varentropy_losses, stddentropy_losses, file_path)

            # end the progrma herer
            exit()

        elif n_samples_to_analyse or num_tokens_to_analyse:
            for i in tqdm(range(len(results))):
                n_samples_to_analyse = min(n_samples_to_analyse, len(results[i]["original_logprobs"]))
                results[i]["original_logprobs"] = results[i]["original_logprobs"][:n_samples_to_analyse]
                results[i]["surprises"] = results[i]["surprises"][:n_samples_to_analyse]
                results[i]["entropies"] = results[i]["entropies"][:n_samples_to_analyse]
                for idx in range(len(results[i]["original_logprobs"])):
                    results[i]["original_logprobs"][idx] = results[i]["original_logprobs"][idx][:num_tokens_to_analyse]
                    results[i]["surprises"][idx] = results[i]["surprises"][idx][:num_tokens_to_analyse]
                    results[i]["entropies"][idx] = results[i]["entropies"][idx][:num_tokens_to_analyse]

        # Create and save plots
        create_plots(results, output_dir)
        print(f"Plots saved in: {output_dir}")
            
        # Print summary of first result as example
        if results:
            print("Analysis of first entry:")
            for key, value in results[0].items():
                if key in ['original_logprobs', 'max_logprobs_per_token', 
                           'entropies', 'surprises', 'scores', 'attention_entropies']:
                    print(f"{key}: [...]")  # Skip printing
                else:
                    print(f"{key}: {value}")

