# https://github.com/tatsu-lab/alpaca_eval

import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
import torch
import os
import json
import argparse
import datasets
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoTokenizer
from src.alignment.data.template import get_template_and_fix_tokenizer, TEMPLATES

def apply_chat_template(examples, tokenizer, system_prompt):
    all_tokenized_prompts = list()
    for example in tqdm(examples):
        prompt = example["instruction"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        all_tokenized_prompts.append(prompt)
    return all_tokenized_prompts


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #type是要传入的参数的数据类型  help是该参数的提示信息
    parser.add_argument("--model_name_or_path", type=str, help="The model name or path of LLM")
    parser.add_argument("--model_name", type=str, help="The model name")
    parser.add_argument("--method_version", type=str, help="The version of the evaluated model, for example: baseline-dpo, ours-dpo-iter2, etc.")
    parser.add_argument("--lora_model_path", type=str, default=None, help="The lora model path if the evaluated LLM has adapter")
    parser.add_argument("--save_dir", type=str, default="examples/evaluation/alpaca_eval/outputs/", help="The save path of prediction.")
    parser.add_argument("--cut_n", type=int, default=None, help="The number of examples cut off when debugging")
    parser.add_argument("--temperature", type=float, default=0.8, help="The temperature")
    parser.add_argument("--topp", type=float, default=0.9, help="The topp")

    # parser.add_argument("--save_step", type=int, default=10, help='保存间隔')
    args = parser.parse_args()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    print("len(tokenizer)={}".format(len(tokenizer)))
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    # load evaluated LLM (with lora)
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=torch.cuda.device_count(),
        enable_lora=True if args.lora_model_path is not None and args.lora_model_path != "None" else False,
    )
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.topp, repetition_penalty=1.2, max_tokens=1024)

    template = TEMPLATES.get(args.model_name, None)
    system_prompt = template.default_system

    # load alpaca_eval (you can change the path to huggingface directly)
    eval_set = datasets.load_dataset("data/alpaca_eval", "alpaca_eval")["eval"]

    if args.cut_n is not None:
        eval_set = [example for ei, example in enumerate(eval_set) if ei < args.cut_n]

    all_tokenized_prompts = tokenized_prompt = apply_chat_template(eval_set, tokenizer, system_prompt)
    assert len(all_tokenized_prompts) == len(eval_set)
    # for example in tqdm(eval_set):
        # generate here is a placeholder for your models generations
        # example["output"] = generate(example["instruction"])
        # assert 1> 2
        # example["generator"] = "my_model" # name of your model
    
    # inference based on vLLM
    if args.lora_model_path is not None and args.lora_model_path != "None":
        outputs = llm.generate(
            all_tokenized_prompts, 
            sampling_params, 
            use_tqdm=True,
            lora_request=LoRARequest("sql_adapter", 1, args.lora_model_path)
        )
    else:
        outputs = llm.generate(
            all_tokenized_prompts, 
            sampling_params, 
            use_tqdm=True
        )
    
    predictions = list()
    for example, output in zip(eval_set, outputs):
        generated_text = output.outputs[0].text
        predictions.append({
            "instruction": example["instruction"],
            "dataset": example["dataset"],
            "output": generated_text,
            "generator": "[{}]{}".format(args.model_name, args.method_version),
            "datasplit": "eval",
        })
    
    # save outputs
    # with open(os.path.join(args.save_dir, f"alpaca_{args.model_name}_{args.method_version}.json"), "w", encoding="utf-8") as fw:
    #     for example in tqdm(predictions):
    #         fw.write(json.dumps(example, ensure_ascii=False) + "\n")

    with open(os.path.join(args.save_dir, f"alpaca_{args.model_name}_{args.method_version}.json"), "w", encoding="utf-8") as fw:
        json.dump(predictions, fw, indent=4)
    