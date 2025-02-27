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
from sal.search import beam_search, best_of_n, dvts, adaptive_best_of_n, load_adaptive_best_of_n_dataset
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
    "estimate_difficulty": adaptive_best_of_n,
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
    # dataset = get_dataset(config)

    llm = LLM(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=config.seed,
        tensor_parallel_size=num_gpus,
    )
    prm = load_prm(config)
    if config.approach == "estimate_difficulty":
        dataset = dataset.map(
            approach_fn,
            batched=True,
            batch_size=config.search_batch_size,
            fn_kwargs={"config": config, "llm": llm, "prm": prm},
            desc="Running search",
            load_from_cache_file=False,
        )
        save_dataset(dataset, config)
    else:
        batch_size = 10
        num_batches = len(dataset) // batch_size + (1 if len(dataset) % batch_size != 0 else 0)

        for index in range(num_batches):
            batch_start = index * batch_size
            batch_end = min((index + 1) * batch_size, len(dataset))
            batch = dataset.select(range(batch_start, batch_end))

            batch = batch.map(
                approach_fn,
                batched=True,
                batch_size=config.search_batch_size,
                fn_kwargs={"config": config, "llm": llm, "prm": prm},
                desc=f"Running search on batch {index + 1}/{num_batches}",
                load_from_cache_file=False,
            )

            if config.approach != "estimate_difficulty":
                batch = score(batch, config)

            file_name = Path(f"{config.dataset_name}_{config.dataset_split}_{config.n}_{config.num_samples}_{config.approach}_stash-{index}.jsonl")
        
            save_dataset(batch, config, output_file=file_name)
    
        save_dataset(batch, config)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
