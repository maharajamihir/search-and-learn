# TODO @mihir

import numpy as np
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def probability_of_difficult(x, config: Config, llm: LLM, prm: PRM):
    # Initialize the tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(llm.model_name)
    model = AutoModelForCausalLM.from_pretrained(llm.model_name)

    if "question" in x.keys():
        x["problem"] = x["question"]

    # Prepare the prompts
    prompts = x["problem"]

    # Tokenize the prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    # Generate outputs
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=False, output_hidden_states=False)

    # Calculate the likelihood of the token "difficult"
    difficult_token_id = tokenizer.convert_tokens_to_ids("difficult")
    difficult_likelihoods = []

    for i, prompt in enumerate(prompts):
        # Get the logits for the last token in the sequence
        logits = outputs.logits[i, -1, :]
        # Calculate the probability of the "difficult" token
        probs = torch.softmax(logits, dim=-1)
        difficult_likelihood = probs[difficult_token_id].item()
        difficult_likelihoods.append(difficult_likelihood)

    x["difficult_likelihoods"] = difficult_likelihoods

    return x
