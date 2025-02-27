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
import time
from pathlib import Path

from datasets import Dataset, load_dataset
from sal.config import Config

logger = logging.getLogger()


def get_dataset(config: Config) -> Dataset:
    if config.dataset_name == "openai/gsm8k":
        dataset = load_dataset(config.dataset_name, "main", split=config.dataset_split)
    else:
        dataset = load_dataset(config.dataset_name, split=config.dataset_split)

    if config.dataset_start is not None and config.dataset_end is not None:
        dataset = dataset.select(range(config.dataset_start, config.dataset_end))
    if config.num_samples is not None:
        dataset = dataset.select(range(min(len(dataset), config.num_samples)))

    return dataset


def save_dataset(dataset, config, output_file=None, difficulties=False):
    if config.output_dir is None:
        config.output_dir = f"data/{config.dataset_name}/{config.model_path}"
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    if not output_file: 
        subdir = "difficulties" if difficulties else "completions"
        output_file = Path(f"{config.output_dir}/{subdir}/{config.approach}_completions.jsonl")
        counter = 1
        while output_file.exists():
            output_file = Path(f"{config.output_dir}/{subdir}/{config.approach}_completions-{counter}.jsonl")
            counter += 1

    dataset.to_json(output_file, lines=True)
    logger.info(f"Saved completions to {output_file}")
