# Copyright 2024 the LlamaFactory team.
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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from src.alignment.extras.constants import IGNORE_INDEX
from src.alignment.extras.logging import get_logger
from src.alignment.data.processors.processor_utils import get_paligemma_token_type_ids, get_pixel_values, infer_seqlen


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from src.alignment.hparams import DataArguments
    from src.alignment.data.template import Template


logger = get_logger(__name__)


def _encode_cls_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Tuple[List[int], List[int], List[int], List[int]]:

    prompt_content = prompt[0]["content"]
    # _, prompt_ids = template.encode_oneturn(tokenizer, prompt, system, tools)

    prompt_ids = tokenizer.encode(prompt_content)

    prompt_ids = prompt_ids[:data_args.cutoff_len - 1]
    prompt_ids = prompt_ids + [tokenizer.eos_token_id]

    return prompt_ids


def preprocess_cls_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:

    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    for i in range(len(examples["prompt"])):

        input_ids = _encode_cls_example(
            prompt=examples["prompt"][i],
            response="",
            system="",
            tools="",
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(int(examples["labels"][i]))

    return model_inputs


def print_cls_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    # valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
    # valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
    # print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
    # print("chosen_inputs:\n{}".format(tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False)))
    # print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
    # print("chosen_labels:\n{}".format(tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)))
    # print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
    # print("rejected_inputs:\n{}".format(tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)))
    # print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
    # print("rejected_labels:\n{}".format(tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)))

    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label:\n{}".format(example["labels"]))

