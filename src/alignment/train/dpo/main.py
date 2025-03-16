#!/usr/bin/env python
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

from src.alignment.hparams import get_infer_args, get_train_args
from typing import TYPE_CHECKING, List, Optional

from src.alignment.data import PairwiseDataCollatorWithPadding, get_dataset
from src.alignment.extras.constants import IGNORE_INDEX
from src.alignment.extras.ploting import plot_loss
from src.alignment.hparams import ModelArguments
from src.alignment.model import load_model, load_tokenizer
from src.alignment.train.callbacks import fix_valuehead_checkpoint, LogCallback
from src.alignment.train.trainer_utils import create_modelcard_and_push, create_ref_model
from src.alignment.train.dpo.trainer import CustomDPOTrainer
from src.alignment.extras.read_yaml_args import read_yaml_file

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from src.alignment.hparams import DataArguments, FinetuningArguments


if __name__ == "__main__":
    args_file = sys.argv[1]
    args = read_yaml_file(args_file)
    print(args)
    callbacks = list()
    callbacks.append(LogCallback())
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    dataset_module = get_dataset(model_args, data_args, training_args, stage="rm", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    data_collator = PairwiseDataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Create reference model
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None and (not training_args.do_train):  # use the model itself
            ref_model = model
        else:
            ref_model = create_ref_model(model_args, finetuning_args)
    else:
        ref_model = None

    # Update arguments
    training_args.remove_unused_columns = False  # important for pairwise dataset

    # Initialize our Trainer
    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "rewards/accuracies"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        if id(model) == id(ref_model):  # unable to compute rewards if reference model is the model itself
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    # create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)

