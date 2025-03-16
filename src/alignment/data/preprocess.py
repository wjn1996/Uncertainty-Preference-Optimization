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

from functools import partial
from typing import TYPE_CHECKING, Callable, Literal, Optional, Tuple

from src.alignment.data.processors.feedback import preprocess_feedback_dataset
from src.alignment.data.processors.pairwise import preprocess_pairwise_dataset, print_pairwise_dataset_example
from src.alignment.data.processors.pretrain import preprocess_pretrain_dataset
from src.alignment.data.processors.classification import preprocess_cls_dataset, print_cls_dataset_example
from src.alignment.data.processors.supervised import (
    preprocess_packed_supervised_dataset,
    preprocess_supervised_dataset,
    print_supervised_dataset_example,
)
from src.alignment.data.processors.unsupervised import preprocess_unsupervised_dataset, print_unsupervised_dataset_example


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from src.alignment.hparams import DataArguments
    from src.alignment.data.template import Template


def get_preprocess_and_print_func(
    data_args: "DataArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto", "cls"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    do_generate: bool = False,
) -> Tuple[Callable, Callable]:
    if stage == "pt":
        # construct for pre-training
        preprocess_func = partial(
            preprocess_pretrain_dataset,
            tokenizer=tokenizer,
            data_args=data_args,
        )
        print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)
    elif stage == "sft" and not do_generate:
        # construct for supervised fine-tuning
        if data_args.packing:
            if data_args.neat_packing:
                from datasets.arrow_writer import OptimizedTypedSequence, TypedSequence

                def __init__(self, data, **kwargs):
                    return TypedSequence.__init__(
                        self,
                        data,
                        type=kwargs.pop("type", None),
                        try_type=kwargs.pop("try_type", None),
                        optimized_int_type=kwargs.pop("optimized_int_type", None),
                    )

                OptimizedTypedSequence.__init__ = __init__
            preprocess_func = partial(
                preprocess_packed_supervised_dataset,
                template=template,
                tokenizer=tokenizer,
                data_args=data_args,
            )
        else:
            preprocess_func = partial(
                preprocess_supervised_dataset,
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                data_args=data_args,
            )

        print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)
    elif stage == "rm":
        # construct for reward model
        preprocess_func = partial(
            preprocess_pairwise_dataset,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        print_function = partial(print_pairwise_dataset_example, tokenizer=tokenizer)
    elif stage == "kto":
        # construct for kto preference learning
        preprocess_func = partial(
            preprocess_feedback_dataset,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)
    elif stage == "cls":
        # construct for cls learning
        preprocess_func = partial(
            preprocess_cls_dataset,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        print_function = partial(print_cls_dataset_example, tokenizer=tokenizer)
    else:
        # default for unsupervised
        preprocess_func = partial(
            preprocess_unsupervised_dataset,
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            data_args=data_args,
        )
        print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)

    return preprocess_func, print_function
