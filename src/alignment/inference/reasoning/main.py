#!/usr/bin/env python
# coding: utf-8

"""
This file aims to generate multiple responses (at least 4) for each prompt, and save the file as `prompt_responses.json`
"""
import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import torch
from tqdm.auto import tqdm
import argparse
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import pickle
import random
import glob
import json
from src.alignment.data.template import get_template_and_fix_tokenizer, TEMPLATES

from accelerate import Accelerator
from accelerate.utils import gather_object
from statistics import mean
accelerator = Accelerator()


def load_prompt(data_path, system_prompt: str=None, cut_num: int=None):
    """
    [
        {
            "messages": [
                {"role": "system", "content": "xxx"},
                {"role": "user", "content": "xxx"},
            ],
            "reference": {"role": "assistant", "content": "xxx"},
            "from": "xx",
        }
    ]
    """
    examples = list()
    with open(data_path, "r", encoding="utf-8") as fr:
        for line in tqdm(fr.readlines()):
            example = json.loads(line)
            assert "messages" in example.keys() and len(example["messages"]) > 0, "the data must has 'messages' key."
            if example["messages"][0]["role"] != "system":
                system_message = {
                    "role": "system",
                    "content": system_prompt
                }
                example["messages"] = [system_message] + example["messages"]
            assert example["messages"][0]["role"] == "system" and example["messages"][-1]["role"] == "user"
            assert example["reference"]["role"] == "assistant"
            examples.append(example)
    
    if cut_num is not None:
        examples = examples[:cut_num]
    return examples

def apply_chat_template(examples, tokenizer, use_chat_template=False, assistant_prefix_sequence=""):
    all_tokenized_examples = list()
    for example in tqdm(examples):
        messages = example["messages"]
        if use_chat_template:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # User: {instruction}.\n\nAssistant: Let's think step by step.\nStep 1: 
            prompt = "User: " + "\n".join([message["content"] for message in messages]) + "\n\nAssistant: "
        all_tokenized_examples.append(prompt + assistant_prefix_sequence)
    return all_tokenized_examples

def inference_with_vllm(llm, prompt_data, tokenized_prompt_data, N, lora_path=None, temperature=1.0, topp=1.0, assistant_prefix_sequence=""
):
    """
    return format:
    {
        "messages": [{"role": "xx", "content": "xx"}, xxx],
        "from": "xxx",
        "reference": {"role": "assistant", "content": "xxx"},
        "responses": [{"role": "assistant", "content": "xx"}, ...]
    }
    """
    print(f"reasoning {N} responses for each prompt.")
    print(f"the number of prompt data is {len(tokenized_prompt_data)}.")
    results = dict()
    N_w_low_temp = int(N / 2)
    for n in range(N):
        if n < N_w_low_temp:
            sampling_params = SamplingParams(temperature=temperature, top_p=topp, repetition_penalty=1.2, max_tokens=1024)
        else:
            sampling_params = SamplingParams(temperature=temperature - 0.1, top_p=topp, repetition_penalty=1.2, max_tokens=1024)
        results[n] = list()
        print(f"-- reasoning at the {n + 1} time(s) --")
        if lora_path is not None and lora_path != "None":
            outputs = llm.generate(
                tokenized_prompt_data, 
                sampling_params, 
                use_tqdm=True,
                lora_request=LoRARequest("sql_adapter", 1, lora_path)
            )
        else:
            outputs = llm.generate(
                tokenized_prompt_data, 
                sampling_params, 
                use_tqdm=True
            )
        
        for output in outputs:
            generated_text = output.outputs[0].text
            results[n].append(generated_text)
    
    for ei, example in enumerate(prompt_data):
        example["responses"] = list()
        for n in range(N):
            example["responses"].append({
                "role": "assistant",
                "content": assistant_prefix_sequence + results[n][ei],
            })
    return prompt_data

def inference_with_llm(llm, tokenizer, _prompt_data, _tokenized_prompt_data, N, batch_size=8):

    for ei, example in enumerate(_prompt_data):
        example["id"] = ei
        example["responses"] = list()

    _prompt_data_concate = list()
    for prompt_data_example, tokenized_prompt_data_example in zip(_prompt_data, _tokenized_prompt_data):
        _prompt_data_concate.append((
            prompt_data_example, tokenized_prompt_data_example
        ))

    accelerator.wait_for_everyone()    

    # results = dict()

    N_w_low_temp = int(N / 2)
    for n in tqdm(range(N)):
        if n < N_w_low_temp:
            temperature=0.9
            top_p=0.9
            repetition_penalty=1.2
        else:
            temperature=0.9
            top_p=0.9
            repetition_penalty=1.2

        with accelerator.split_between_processes(_prompt_data_concate) as prompt_data_concate:
            prompt_data, tokenized_prompt_data = list(), list()
            for (prompt_data_example, tokenized_prompt_data_example) in prompt_data_concate:
                prompt_data.append(prompt_data_example)
                tokenized_prompt_data.append(tokenized_prompt_data_example)
            assert len(prompt_data) == len(tokenized_prompt_data)

            all_results = list()
            tokenizer.padding_side = "left"
            batch_prompt_data = list()
            tokenized_batch_prompt_data = list()

            for i in range(0, len(tokenized_prompt_data), batch_size):
                cur_prompt_data = prompt_data[i: i + batch_size]
                cur_tokenized_prompt_data = tokenized_prompt_data[i: i + batch_size]
                batch_prompt_data.append(cur_prompt_data)

                tokenized_cur_prompt_data = tokenizer(
                    cur_tokenized_prompt_data,
                    padding="max_length",
                    return_tensors="pt",
                    max_length=max([len(i)for i in cur_tokenized_prompt_data]) + 10,
                    add_special_tokens=False,
                )
                tokenized_cur_prompt_data = tokenized_cur_prompt_data.to(llm.device)
                tokenized_batch_prompt_data.append(tokenized_cur_prompt_data)

            with torch.no_grad():
                for batch, batch_prompt_example in zip(tokenized_batch_prompt_data, batch_prompt_data):
                    outputs = llm.generate(
                        **batch,
                        max_new_tokens=512,
                        do_sample=True,
                        top_p=top_p,
                        temperature=temperature,
                        min_length=None,
                        use_cache=True,
                        # top_k=50,
                        repetition_penalty=repetition_penalty,
                        length_penalty=1
                    )
                    outputs = [output_ids[len(input_ids):] for output_ids, input_ids in zip(outputs, batch["input_ids"])]
                    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                    for prompt_example, output in zip(batch_prompt_example, responses):
                        prompt_example["responses"].append(output)
                        all_results.append({
                            "id": prompt_example["id"],
                            "response": output
                        })

        results_gathered=gather_object(all_results)
        print("results_gathered=", results_gathered)
        id2response = dict()
        for res in results_gathered:
            id2response[res["id"]] = res["response"]
        
        for example in _prompt_data:
            id = example["id"]
            example["responses"].append(id2response[id])
    
    for example in _prompt_data:
        del example["id"]
    
    return _prompt_data




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #type是要传入的参数的数据类型  help是该参数的提示信息
    parser.add_argument("--prompt_data_path", type=str, help="The path of prompt set")
    parser.add_argument("--iteration_stage", type=int, help="The stage idx of the iteration preference optimization, e.g., 1, 2, 3")
    parser.add_argument("--model_name_or_path", type=str, help="The model name or path of LLM")
    parser.add_argument("--model_name", type=str, help="The model name")
    parser.add_argument("--lora_model_path", type=str, default=None, help="The lora model path")
    parser.add_argument("--save_dir", type=str, default="outputs/llm_bench", help="The save path (must equal to the benchmark directory, e.g., llm_bench)")
    parser.add_argument("--generate_n", type=int, default="6", help="The number of generated responses of each prompt")
    parser.add_argument("--cut_n", type=int, default=None, help="The number of examples cut off when debugging")
    parser.add_argument("--block_num", type=int, default=1, help="The number of block")
    parser.add_argument("--temperature", type=float, default=0.8, help="The temperature")
    parser.add_argument("--topp", type=float, default=0.9, help="The topp")
    parser.add_argument("--assistant_prefix_sequence", type=str, default="", help="The prefix prompt before the assistant generation. e.g., <Assistant>: Let's think step by step. ")
    parser.add_argument('--use_vllm', action='store_true', help="Whether to use the vLLM")
    parser.add_argument('--use_chat_template', action='store_true', help="Whether to use the chat template from tokenizer")

    # parser.add_argument("--save_step", type=int, default=10, help='保存间隔')
    args = parser.parse_args()
    print(args.assistant_prefix_sequence)

    save_path = os.path.join(args.save_dir, f"iteration_stage_{args.iteration_stage}", "prompt_responses/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print("load tokenizer ...")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    print("len(tokenizer)={}".format(len(tokenizer)))
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    # load template
    template = TEMPLATES.get(args.model_name, None)

    # load data
    print("load prompt data ...")
    prompt_data = load_prompt(args.prompt_data_path, system_prompt=template.default_system, cut_num=args.cut_n)
    tokenized_prompt_data = apply_chat_template(prompt_data, tokenizer, use_chat_template=args.use_chat_template, assistant_prefix_sequence=args.assistant_prefix_sequence)

    print("=====example=====\n")
    print(tokenized_prompt_data[0])
    print("\n=================\n")


    print("load model and start inference ...")
    # load model and reason
    if args.use_vllm:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=torch.cuda.device_count(),
            enable_lora=True if args.lora_model_path is not None and args.lora_model_path != "None" else False,
        )
    else:
        llm = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True).cuda()
        if args.lora_model_path is not None and args.lora_model_path != "None":
            print("loading peft")
            llm = PeftModel.from_pretrained(llm, args.lora_model_path)
    
    # split data in difference block
    data_num_per_block = int((len(prompt_data) - 1) / args.block_num) + 1
    data_idx_span_per_block = [[block_idx * data_num_per_block, (block_idx + 1) * data_num_per_block] for block_idx in range(0, args.block_num)]

    print(f"total block num: {args.block_num}")
    print(f"data num per block: {data_num_per_block}")

    block_idx = 0
    for (block_start, block_end) in tqdm(data_idx_span_per_block):
        print(f"==== inference for block {block_idx} ====")
        block_prompt_data = prompt_data[block_start: block_end]
        block_tokenized_prompt_data = tokenized_prompt_data[block_start: block_end]

        if args.use_vllm:
            outputs = inference_with_vllm(llm, block_prompt_data, block_tokenized_prompt_data, args.generate_n, lora_path=args.lora_model_path, temperature=args.temperature, topp=args.topp, assistant_prefix_sequence=args.assistant_prefix_sequence)
        else:
            outputs = inference_with_llm(llm, tokenizer, block_prompt_data, block_tokenized_prompt_data, args.generate_n)

        print("save results ...")
        # save reasoning results
        with open(os.path.join(save_path, "prompt_responses.json"), "a", encoding="utf-8") as fw:
            for example in tqdm(outputs):
                fw.write(json.dumps(example, ensure_ascii=False) + "\n")
        
        block_idx += 1
    
    print("done.")
