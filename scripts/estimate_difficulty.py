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

import logging

import torch
from vllm import LLM
from pathlib import Path

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.estimate_difficulty import difficulty_probe, empirical_difficulty, prm_based_difficulty, probability_of_difficult, varentropy_difficulty, verbal_difficulty
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "difficulty_probe": difficulty_probe,
    "empirical_difficulty": empirical_difficulty,
    "prm_based_difficulty": prm_based_difficulty,
    "probability_of_difficult": probability_of_difficult,
    "varentropy_difficulty": varentropy_difficulty,
    "verbal_difficulty": verbal_difficulty,
}

vllm_approaches = [
    "empirical_difficulty", 
    "prm_based_difficulty", 
    "varentropy_difficulty", 
    "verbal_difficulty"
]

hf_approaches = [
    "probability_of_difficult"
]


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse(True)

    approach_fn = APPROACHES[config.approach]

    num_gpus = torch.cuda.device_count()

    dataset = get_dataset(config)

    llm = LLM(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=config.seed,
        tensor_parallel_size=num_gpus,
    )
    prm = load_prm(config)
    
    dataset = dataset.map(
        approach_fn,
        batched=True,
        batch_size=config.search_batch_size,
        fn_kwargs={"config": config, "llm": llm, "prm": prm},
        desc=f"Estimating difficulty using {config.approach}",
        load_from_cache_file=False,
    )
    save_dataset(dataset, config, difficulties=True)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
