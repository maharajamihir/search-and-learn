# TODO @mihir

import numpy as np
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM


def difficulty_probe(x, config: Config, llm: LLM, prm: PRM):
    # TODO
    return x