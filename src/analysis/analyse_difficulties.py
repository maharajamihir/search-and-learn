# TODO @mihir
import argparse
from datasets import load_dataset
from pathlib import Path
import json
import numpy as np
from sal.utils.qwen_math_parser import extract_answer, strip_string
from sal.utils.score import aggregate_scores
from sal.estimate_difficulty.varentropy_based_difficulty import calculate_per_token_entropies

def _get_dataset_difficulty(problem, dataset):
    """
    Calculate the difficulty of a problem within a specified dataset.

    This function determines the difficulty level of a given problem based on the dataset it belongs to.
    Currently, it supports the 'MATH-500' dataset, where the difficulty is calculated as 6 minus the problem's level.
    If the dataset is not recognized, a ValueError is raised.

    Args:
        problem (dict): A dictionary representing the problem, expected to contain a 'level' key.
        dataset (str): The name of the dataset to which the problem belongs.

    Returns:
        int: The calculated difficulty level of the problem.

    Raises:
        ValueError: If the dataset is not recognized or not implemented.
    """
    if 'MATH-500' in dataset:
        return 6 - problem['level']
    else:
        raise ValueError(f"Dataset '{dataset}' is not known or not implemented yet.")

def _get_empirical_difficulty(problem, dataset):
    predicted_answers = [extract_answer(compl, dataset) for compl in problem['completions']]
    gt_answer = strip_string(problem['answer'])
    pass_at_1 = predicted_answers.count(gt_answer) / len(problem['completions'])
    empirical_difficulty = 1-pass_at_1
    return empirical_difficulty

def _get_prm_based_difficulty(problem, dataset):
    scores = problem["scores"]
    agg_scores_last = np.mean([aggregate_scores(s, "last") for s in scores])
    agg_scores_min = np.mean([aggregate_scores(s, "min") for s in scores])
    agg_scores_prod = np.mean([aggregate_scores(s, "prod") for s in scores])
    return {"last": 1.-agg_scores_last, "min": 1.-agg_scores_min, "prod": 1.-agg_scores_prod}

def _get_varentropy_difficulty(problem, dataset):
    if not 'log_probs' in problem.keys():
        raise ValueError(f"Generation does not contrain logprobs. Please check if these are generations from the correct run, or rerun with logprobs turned on.")
    if not 'entropies' in problem.keys():
        entropies = [calculate_per_token_entropies(logprobs) for logprobs in problem["log_probs"]]
        problem['entropies'] = [np.mean(e) for e in entropies]
        problem['varentropies'] = [np.var(e) for e in entropies]
    rank_1_log_probs = [lp[0] for log_probs in problem['log_probs'] for lp in log_probs]
    return {"log_prob_difficulty": -np.mean(rank_1_log_probs), 
            "entropy_difficulty": np.mean(problem['entropies']), 
            "varentropy_difficulty": np.mean(problem['varentropies'])}

def _get_verbal_difficulty(problem, dataset):
    if "verbal_difficulties" not in problem.keys():
        raise ValueError("Verbal difficulties not found")
    return problem["verbal_difficulties"]

def _get_probability_of_difficult(problem, dataset):
    pass

def _get_difficulty_probe_difficulty(problem, dataset):
    pass

def analyse_difficulties(dataset, model):
    file_path = Path("data") / dataset / model / "completions" / "best_of_n_completions.jsonl"
    
    with open(file_path, 'r') as f:
        results = [json.loads(line) for line in f]
    print(f"Loaded dataset from {file_path}")

    difficulty_types = [
        "varentropy",
        "empirical",
        "prm_based",
        "verbal",
        "probability_of_difficult",
        "difficulty_probe"
    ]

    difficulty_results = {}

    for difficulty in difficulty_types:
        difficulty_path = Path("data") / dataset / model / "difficulties" / f"{difficulty}_difficulty.jsonl"
        difficulty_results[difficulty] = None

        if difficulty_path.exists():
            with open(difficulty_path, 'r') as df:
                difficulty_results[difficulty] = [json.loads(line) for line in df]
            print(f"Loaded {difficulty} difficulties from {difficulty_path}")
        else:
            print(f"{difficulty.capitalize()} difficulty file not found at {difficulty_path}")

    for idx, problem in enumerate(results):
        results[idx]['dataset_difficulty'] = _get_dataset_difficulty(problem, dataset)
        results[idx]['empirical_difficulty'] = _get_empirical_difficulty(problem, dataset)
        results[idx]['pass@1'] = 1-results[idx]['empirical_difficulty']
        results[idx]['prm_based_difficulty'] = _get_prm_based_difficulty(problem, dataset)
        if difficulty_results["varentropy"]:
            assert results[idx]['problem'] == difficulty_results["varentropy"][idx]['problem']
            results[idx]['varentropy_difficulty'] = _get_varentropy_difficulty(difficulty_results["varentropy"][idx], dataset)
        if difficulty_results["verbal"]:
            assert results[idx]['problem'] == difficulty_results["verbal"][idx]['problem']
            results[idx]['verbal_difficulty'] = _get_verbal_difficulty(difficulty_results["verbal"][idx], dataset)
        if difficulty_results["probability_of_difficult"]:
            assert results[idx]['problem'] == difficulty_results["probability_of_difficult"][idx]['problem']
            results[idx]['probability_of_difficult'] = _get_probability_of_difficult(difficulty_results["probability_of_difficult"][idx], dataset)
        if difficulty_results["difficulty_probe"]:
            assert results[idx]['problem'] == difficulty_results["difficulty_probe"][idx]['problem']
            results[idx]['difficulty_probe_difficulty'] = _get_difficulty_probe_difficulty(difficulty_results["difficulty_probe"][idx], dataset)

    # Save the accumulated results to a JSONL file
    output_path = Path("data") / dataset / model / "difficulties" / "difficulties_accumulated.jsonl"
    with open(output_path, 'w') as outfile:
        for result in results:
            json.dump(result, outfile)
            outfile.write('\n')
    print(f"Saved accumulated difficulties to {output_path}")

    



def main():
    parser = argparse.ArgumentParser(description="Analyse difficulties in a dataset using a specified model.")
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset used')
    parser.add_argument('--model', type=str, required=True, help='Model to used for run')
        
    args = parser.parse_args()
        
    analyse_difficulties(args.dataset_name, args.model)

if __name__ == "__main__":
    main()
