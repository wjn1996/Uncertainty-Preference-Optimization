#!/usr/bin/env python
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
import torch
from typing import TYPE_CHECKING, List, Optional

from src.alignment.data import CLSDataCollatorWithPadding, get_dataset
from src.alignment.extras.ploting import plot_loss
from src.alignment.model import load_model, load_tokenizer, load_model_for_cls
from src.alignment.train.callbacks import fix_clshead_checkpoint, LogCallback
from src.alignment.train.trainer_utils import create_modelcard_and_push
from src.alignment.train.estimator.metric import ComputeAccuracy
from src.alignment.train.estimator.trainer import CLSTrainer
from src.alignment.hparams import get_infer_args, get_train_args
from src.alignment.extras.read_yaml_args import read_yaml_file

from transformers import AutoModelForSequenceClassification, AutoConfig
from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model

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
    dataset_module = get_dataset(model_args, data_args, training_args, stage="cls", **tokenizer_module)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    # config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    # setattr(config, "num_labels", num_labels)
    # setattr(config, "pad_token_id", config.eos_token_id)
    # model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path)


    model = load_model_for_cls(tokenizer, model_args, finetuning_args, training_args.do_train, num_labels=2)

    # 查看模型参数
    print("\n====parameters=====\n")
    for name, param in model.named_parameters():
        print(name)

        
    model = model.to(torch.bfloat16)

    data_collator = CLSDataCollatorWithPadding(tokenizer)

    # Update arguments
    training_args.remove_unused_columns = False  # important for pairwise dataset

    # Initialize our Trainer
    trainer = CLSTrainer(
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
            fix_clshead_checkpoint(model, training_args.output_dir, training_args.save_safetensors)

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

