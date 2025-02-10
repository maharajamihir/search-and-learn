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

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n, dvts, estimate_difficulty, adaptive_best_of_n, load_adaptive_best_of_n_dataset
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
    "estimate_difficulty": estimate_difficulty,
    "adaptive_best_of_n": adaptive_best_of_n
}


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse(True)

    approach_fn = APPROACHES[config.approach]

    num_gpus = torch.cuda.device_count()

    if config.approach == "adaptive_best_of_n":
        dataset = load_adaptive_best_of_n_dataset(config.dataset_name, config.n)
    else:
        dataset = get_dataset(config)

    if config.approach != "estimate_difficulty":
        llm = LLM(
            model=config.model_path,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=True,
            seed=config.seed,
            tensor_parallel_size=num_gpus,
        )
        prm = load_prm(config)
    else: 
        llm = None
        prm = None

    dataset = dataset.map(
        approach_fn,
        batched=True,
        batch_size=config.search_batch_size,
        fn_kwargs={"config": config, "llm": llm, "prm": prm},
        desc="Running search",
        load_from_cache_file=False,
    )

    if config.approach != "estimate_difficulty":
        dataset = score(dataset, config)

    save_dataset(dataset, config)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
