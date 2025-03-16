#!/usr/bin/env python
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

from typing import TYPE_CHECKING, List, Optional

from src.alignment.data import PairwiseDataCollatorWithPadding, get_dataset
from src.alignment.extras.ploting import plot_loss
from src.alignment.model import load_model, load_tokenizer
from src.alignment.train.callbacks import fix_valuehead_checkpoint, LogCallback
from src.alignment.train.trainer_utils import create_modelcard_and_push
from src.alignment.train.rm.metric import ComputeAccuracy
from src.alignment.train.rm.trainer import PairwiseTrainer
from src.alignment.hparams import get_infer_args, get_train_args
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
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)
    data_collator = PairwiseDataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # Update arguments
    training_args.remove_unused_columns = False  # important for pairwise dataset

    print("\n====dataset example=====\n")
    print(dataset_module["train_dataset"][0])
    
    # Initialize our Trainer
    trainer = PairwiseTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeAccuracy(),
        **dataset_module,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if training_args.should_save:
            fix_valuehead_checkpoint(model, training_args.output_dir, training_args.save_safetensors)

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict")
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)

