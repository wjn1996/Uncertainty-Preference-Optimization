

import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
import torch
import os
import shortuuid
import json
import argparse
import datasets
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoTokenizer
from src.alignment.data.template import get_template_and_fix_tokenizer, TEMPLATES
from examples.evaluation.mt_bench.conversation import conv_templates, register_conv_template, Conversation, SeparatorStyle, get_conv_template
import random
import time
from enum import auto, IntEnum

temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}

def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #type是要传入的参数的数据类型  help是该参数的提示信息
    parser.add_argument("--model_name_or_path", type=str, help="The model name or path of LLM")
    parser.add_argument("--model_name", type=str, default="zephyr", help="The model name")
    parser.add_argument("--method_version", type=str, help="The version of the evaluated model, for example: baseline-dpo, ours-dpo-iter2, etc.")
    parser.add_argument("--lora_model_path", type=str, default=None, help="The lora model path if the evaluated LLM has adapter")
    parser.add_argument("--save_dir", type=str, default="examples/evaluation/mt_bench/outputs/", help="The save path of prediction.")
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

    template = TEMPLATES.get(args.model_name, None)
    system_prompt = template.default_system

    questions = load_questions(question_file="data/mt_bench/question.jsonl")
    random.shuffle(questions)

    register_conv_template(
        Conversation(
            name="zephyr",
            system_template=f"<|system|>\n{system_prompt}",
            roles=("<|user|>", "<|assistant|>"),
            sep_style=SeparatorStyle.CHATML,
            sep="</s>",
            stop_token_ids=[2],
            stop_str="</s>",
        )
    )

    num_choices = 1

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7
        

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conv_template(args.model_name)
            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                # print(f"prompt [{j}]={prompt}")

                if temperature < 1e-4:
                    sampling_params = SamplingParams(
                        temperature=temperature, 
                        top_p=1.0, 
                        max_tokens=2048
                    )

                else:
                    sampling_params = SamplingParams(
                        temperature=temperature, 
                        top_p=0.95, 
                        max_tokens=2048
                    )
                
                # some models may error out when generating long outputs
                if args.lora_model_path is not None and args.lora_model_path != "None":
                    outputs = llm.generate(
                        [prompt], 
                        sampling_params, 
                        use_tqdm=False,
                        lora_request=LoRARequest("sql_adapter", 1, args.lora_model_path)
                    )
                else:
                    outputs = llm.generate(
                        [prompt], 
                        sampling_params, 
                        use_tqdm=False,
                    )
                
                generated_text = outputs[0].outputs[0].text
                conv.update_last_message(generated_text)
                turns.append(generated_text)

            choices.append({"index": i, "turns": turns})

        # Dump answers
        answer_file = os.path.join(args.save_dir, f"mtbench_{args.model_name}_{args.method_version}.jsonl")

        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": args.model_name,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")