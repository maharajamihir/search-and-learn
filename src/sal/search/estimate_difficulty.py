
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
    # TODO @mihir check the hyperparameters
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        logprobs=config.log_probs,
        n=1,  # Since we've already duplicated the prompt_token_ids, we only need to generate 1 completion per prompt
    )
    # Generate using Hugging Face

    attn_list = []
    log_probs_list = []
    for conv in tqdm(templated_convs):
        attentions_list_conv = []
        log_probs_list_conv = []
        # Generate text with attention outputs
        inputs = tokenizer(conv, return_tensors="pt")
        inputs = inputs.to(model.device)  # Move inputs to the same device as the model
        with torch.no_grad():
            for _ in range(config.max_tokens):
                outputs = model(**inputs, attn_implementation="eager")
                # Sample next token using temperature
                next_token_logits = outputs.logits[:, -1, :] / config.temperature
                next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                next_token_probs = next_token_probs[:, :config.log_probs] # cutoff logprobs for memory efficiency
                log_probs_list_conv.append(next_token_probs.cpu()) 
            
                # gather the attentions coefficients over the selected token per head
                next_token_attentions = outputs.attentions
                selected_token_attentions = [
                    attention[:, :, -1, :].cpu() for attention in next_token_attentions
                ]
                attentions_list_conv.append(selected_token_attentions)

                inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token], dim=-1)
                inputs['attention_mask'] = torch.ones(inputs["input_ids"].shape, dtype=torch.int)
        attn_list.append(attentions_list_conv)
        log_probs_list.append(log_probs_list_conv)
    
    for i in range(len(x["problem"])):
        log_probs[i] = [
            log_prob
            for log_prob in log_probs_list[i * config.n : (i + 1) * config.n]
        ]
        attentions[i] = [
            attn 
            for attn in attn_list[i * config.n : (i + 1) * config.n]
        ]

    x["log_probs"] = log_probs
    x["attentions"] = attentions
    return x
