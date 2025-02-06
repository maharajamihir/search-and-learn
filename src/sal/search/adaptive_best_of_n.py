#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores
from datasets import Dataset
import msgpack

def load_adaptive_best_of_n_dataset(path, max_num_generations):
    with open(path, 'rb') as f:
        data = msgpack.unpack(f, raw=False)
    for idx, element in enumerate(data):
        difficulty = element.get('difficulty', 0.5)  # Default to 1 if difficulty is not present
        data[idx]['n'] = int(difficulty * max_num_generations + 0.5)
    dataset = Dataset.from_list(data)
    return dataset


def adaptive_best_of_n(x, config: Config, llm: LLM, prm: PRM):
    tokenizer = llm.get_tokenizer()

    convs = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in x["problem"]
    ]
    tokenizer = llm.get_tokenizer()
    # TODO: set the augmented template from a file
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template
    templated_convs = tokenizer.apply_chat_template(
        convs, tokenize=False, add_generation_prompt=True
    )

    n_generations = x["n"]
    if type(n_generations) == list and type(n_generations[0]) == int:
        n_generations = n_generations[0]
    
    if n_generations == 0:
        x["completions"] = []
        x["scores"] = []
        x["pred"] = []
        x["completion_tokens"] = []
        x["log_probs"] = []
        return x

    # Duplicate convs to generate n_generations completions per prompt so we can do continous batching
    # This makes [p1, p2, p3, p4] become [p1, p1, p2, p2, p3, p3, p4, p4] for e.g. n_generations=2
    templated_convs = [c for conv in templated_convs for c in [conv] * n_generations]

    # Initialize empty lists for completions and completion tokens
    completions = [[] for _ in range(len(x["problem"]))]
    completion_tokens = [[] for _ in range(len(x["problem"]))]
    log_probs = [[] for _ in range(len(x["problem"]))]
    # TODO @mihir check the hyperparameters
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

    if len(responses) != len(x["problem"]) * n_generations:
        raise ValueError(
            f"Generated {len(responses)} responses instead of {len(x['problem'] * n_generations)}"
        )

    for i in range(len(completions)):
        completions[i] = [
            output.text
            for r in responses[i * n_generations : (i + 1) * n_generations]
            for output in r.outputs
        ]
        completion_tokens[i] = [
            len(output.token_ids)
            for r in responses[i * n_generations : (i + 1) * n_generations]
            for output in r.outputs
        ]
        for r in responses[i * n_generations : (i + 1) * n_generations]:
            for output in r.outputs:
                log_probs_per_token = []
                for token_distr in output.logprobs:
                    log_prob_objects = token_distr.values()
                    log_prob_values = [lp_obj.logprob for lp_obj in  log_prob_objects]
                    log_probs_per_token.append(log_prob_values) 
                log_probs[i].append(log_probs_per_token)

    # Check we generated the correct number of completions for each prompt
    for c in completions:
        if len(c) != n_generations:
            raise ValueError(f"Generated {len(c)} completions instead of {n_generations}")
    scores = prm.score(x["problem"], completions)
    agg_scores = [
        [aggregate_scores(s, config.agg_strategy) for s in score] for score in scores
    ]

    # Select the completion with the highest score
    pred = [completion[np.argmax(s)] for completion, s in zip(completions, agg_scores)]

    x["completions"] = completions
    x["scores"] = scores
    x["pred"] = pred
    x["completion_tokens"] = completion_tokens
    x["log_probs"] = log_probs
    return x
