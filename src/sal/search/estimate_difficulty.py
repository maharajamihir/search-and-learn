
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

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores
import torch
import torch.nn.functional as F
from tqdm import tqdm

def compute_attention_entropy(attention_weights):
    # attention_weights: (batch_size, num_heads, seq_len, seq_len)
    # Select a specific head, e.g., head_index = 0
    head_index = 0
    attention_head = attention_weights[:, head_index, :, :]  # (batch_size, seq_len, seq_len)

    # Normalize the attention weights to form a probability distribution
    attention_probs = F.softmax(attention_head, dim=-1)  # (batch_size, seq_len, seq_len)

    # Compute entropy for each token's attention distribution
    entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-9), dim=-1)  # (batch_size, seq_len)

    return entropy


def estimate_difficulty(x, config: Config, llm: LLM, prm: PRM):

    convs = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in x["problem"]
    ]
    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, return_dict_in_generate=True, output_attentions=True, attn_implementation="eager")
    tokenizer.pad_token = tokenizer.eos_token
    # TODO: set the augmented template from a file
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template
    templated_convs = tokenizer.apply_chat_template(
        convs, tokenize=False, add_generation_prompt=True
    )

    # Duplicate convs to generate config.n completions per prompt so we can do continous batching
    # This makes [p1, p2, p3, p4] become [p1, p1, p2, p2, p3, p3, p4, p4] for e.g. config.n=2
    templated_convs = [c for conv in templated_convs for c in [conv] * config.n]

    # Initialize empty lists for completions and completion tokens
    log_probs = [[] for _ in range(len(x["problem"]))]
    attentions = [[] for _ in range(len(x["problem"]))]
    completions = [[] for _ in range(len(x["problem"]))]
    # Generate using Hugging Face
    attn_list = []
    log_probs_list = []
    completions_list = []
    batch_size = 8  # Define the batch size
    # Process in batches
    for batch_start in tqdm(range(0, len(templated_convs), batch_size)):
        batch_convs = templated_convs[batch_start:batch_start + batch_size]
        attentions_list_batch = [[] for _ in range(len(batch_convs))]
        log_probs_list_batch = [[] for _ in range(len(batch_convs))]
        completions_list_batch = [[] for _ in range(len(batch_convs))]


        # Tokenize and move to device
        inputs = tokenizer(batch_convs, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            for _ in tqdm(range(config.max_tokens)):
                outputs = model(**inputs, attn_implementation="eager")
                # Sample next token using temperature
                next_token_logits = outputs.logits[:, -1, :] / config.temperature
                next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                next_token_probs = next_token_probs[:, :config.log_probs]  # cutoff logprobs for memory efficiency

                for i in range(len(batch_convs)):
                    log_probs_list_batch[i].append(next_token_probs[i].cpu())
                    completions_list_batch[i].append(next_token[i].cpu())  # Add next token to completions

                # gather the attentions coefficients over the last non-padded token per head
                for i, attention in enumerate(outputs.attentions[0]):
                    selected_token_attentions = attention[:, -1, :]
                    attentions_list_batch[i].append(selected_token_attentions.cpu())
                # Update inputs for the next iteration
                inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token], dim=1)
                inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones((len(batch_convs), 1), dtype=torch.int).to(model.device)], dim=1)

        attn_list.extend(attentions_list_batch)
        log_probs_list.extend(log_probs_list_batch)
        completions_list.extend(completions_list_batch)  # Extend completions list

    completions_list = torch.tensor(completions_list)
    completions_list = [tokenizer.decode(compl) for compl in completions_list]
    # Correctly assign log_probs, attentions, and completions to each problem
    for i in range(len(x["problem"])):
        log_probs[i] = [
            log_prob
            for log_prob in log_probs_list[i * config.n : (i + 1) * config.n]
        ]
        attentions[i] = [
            attn 
            for attn in attn_list[i * config.n : (i + 1) * config.n]
        ]
        completions[i] = [
            completion
            for completion in completions_list[i * config.n : (i + 1) * config.n]
        ]

    x["completions"] = completions
    x["log_probs"] = log_probs
    x["attentions"] = attentions
    return x
