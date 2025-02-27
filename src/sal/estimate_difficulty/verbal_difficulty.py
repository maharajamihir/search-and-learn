# TODO @mihir: TEST

from vllm import LLM, SamplingParams
import re

from sal.config import Config
from sal.models.reward_models import PRM


def _extract_difficulty_value(completion):
    '''
    Extract the difficulty value from the completion string
    '''
    match = re.search(r'(\d+)(?!.*\d)', completion)
    if match:
        return int(match.group(1))
    else:
        return None

def verbal_difficulty(x, config: Config, llm: LLM, prm: PRM):

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
    verbal_difficulties = [[] for _ in range(len(x["problem"]))]

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
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
        verbal_difficulties[i] = [
            _extract_difficulty_value(output.text)
            for r in responses[i * config.n : (i + 1) * config.n]
            for output in r.outputs
        ]

    x["completions"] = completions
    x["completion_tokens"] = completion_tokens
    x["verbal_difficulties"] = verbal_difficulties

    return x
