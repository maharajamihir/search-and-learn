# TODO @mihir: TEST

import numpy as np
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM

def calculate_per_token_entropies(per_token_logprobs):
    """
    Calculate the entropy for each token given its log probabilities.

    Args:
        per_token_logprobs (list of list of float): A list where each element is a list of log probabilities
                                                    for a token across different possible tokens.

    Returns:
        list of float: A list of entropy values for each token.
    """
    entropies = []
    for logprobs in per_token_logprobs:
        # Convert log probabilities to probabilities
        probs = np.exp(logprobs)
        # Calculate entropy for the current token
        entropy = -np.sum(probs * logprobs)
        entropies.append(entropy)
    return entropies


def varentropy_difficulty(x, config: Config, llm: LLM, prm: PRM):

    if "question" in x.keys():
        x["problem"] = x["question"]

    convs = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in x["problem"]
    ]
    tokenizer = llm.get_tokenizer()

    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template
    templated_convs = tokenizer.apply_chat_template(
        convs, tokenize=False, add_generation_prompt=True
    )

    # Duplicate convs to generate config.n completions per prompt so we can do continous batching
    # This makes [p1, p2, p3, p4] become [p1, p1, p2, p2, p3, p3, p4, p4] for e.g. config.n=2
    templated_convs = [c for conv in templated_convs for c in [conv] * config.n]

    # Initialize empty lists for completions and completion tokens
    completions = [[] for _ in range(len(x["problem"]))]
    completion_tokens = [[] for _ in range(len(x["problem"]))]
    log_probs = [[] for _ in range(len(x["problem"]))]
    entropies = [None for _ in range(len(x["problem"]))]
    varentropies = [None for _ in range(len(x["problem"]))]

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        logprobs=config.log_probs,
        n=1,  # Since we've already duplicated the prompt_token_ids, we only need to generate 1 completion per prompt
    )
    # Generate using vLLM
    responses = llm.generate(
        templated_convs,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    if len(responses) != len(x["problem"]) * config.n:
        raise ValueError(
            f"Generated {len(responses)} responses instead of {len(x['problem'] * config.n)}"
        )

    for i in range(len(completions)):
        completions[i] = [
            output.text
            for r in responses[i * config.n : (i + 1) * config.n]
            for output in r.outputs
        ]
        completion_tokens[i] = [
            len(output.token_ids)
            for r in responses[i * config.n : (i + 1) * config.n]
            for output in r.outputs
        ]
        for r in responses[i * config.n : (i + 1) * config.n]:
            for output in r.outputs:
                log_probs_per_token = []

                for token_distr in output.logprobs:
                    log_prob_objects = token_distr.values()
                    log_prob_values = [lp_obj.logprob for lp_obj in  log_prob_objects]
                    log_probs_per_token.append(log_prob_values) 

                # Calculate entropy and variance of entropy for the sequence
                entropies_per_token = calculate_per_token_entropies(log_probs_per_token)
                entropy_per_seq = np.mean(entropies_per_token)
                varentropy_per_seq = np.var(entropies_per_token)
                
                # Append log probabilities and store entropy metrics
                log_probs[i].append(log_probs_per_token)
                entropies[i] = entropy_per_seq
                varentropies[i] = varentropy_per_seq

    x["completions"] = completions
    x["completion_tokens"] = completion_tokens
    x["log_probs"] = log_probs
    x["entropies"] = entropies
    x["varentropies"] = varentropies

    return x