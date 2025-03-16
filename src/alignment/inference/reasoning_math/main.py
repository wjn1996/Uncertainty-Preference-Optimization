import argparse
import json
import pdb
import os
from tqdm import tqdm

# from evaluation.eval.eval_script import eval_math 
# from evaluation.data_processing.answer_extraction import extract_math_answer

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import torch
import sys
MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"

invalid_outputs = []

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def load_prompt(data_path, system_prompt: str="", cut_num: int=None):
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
            if example["messages"][0]["role"] == "system":
                example["messages"] = example["messages"][1:]
            assert example["reference"]["role"] == "assistant"
            examples.append(example)
    
    if cut_num is not None:
        examples = examples[:cut_num]
    return examples

def test_hendrycks_math(model, data_path, remainder=0, n_groups=MAX_INT, batch_size=1, tensor_parallel_size=1, args=None):
    
    save_dir = args.save_dir
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    attributes = []
    if args.prompt == 'alpaca':
        problem_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
        )
    elif args.prompt == 'alpaca-cot-step':
        problem_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\nLet's think step by step.\nStep 1: "
        )
    elif args.prompt == 'alpaca-cot-prefix':
        problem_prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\nLet's think step by step.\n{prefix}"
        )
    elif args.prompt == 'deepseek-math':
        problem_prompt = (
            "User: {instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant:"
        )
    elif args.prompt == 'deepseek-math-step':
        problem_prompt = (
            "User: {instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant: Let's think step by step.\nStep 1: "
        )
    elif args.prompt == 'qwen2-boxed':
        problem_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    elif args.prompt == 'qwen2-boxed-step':
        problem_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            "<|im_start|>assistant\nLet's think step by step.\nStep 1: "
        )
    elif args.prompt == 'qwen2-boxed-prefix':
        problem_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            "<|im_start|>assistant\nLet's think step by step.\n{prefix}"
        )

    print('prompt =====', problem_prompt)
    prompt_data = load_prompt(data_path, "", args.cut_n)
    for example in prompt_data:
        # if "prefix" in item:
        #     temp_instr = problem_prompt.format(instruction=item["instruction"], prefix=item['prefix'])
        # else:
        instruction = example["messages"][0]["content"] # the first message is the user instruction.
        temp_instr = problem_prompt.format(instruction=instruction)
        example["messages"][0]["content"] = temp_instr
        hendrycks_math_ins.append(temp_instr)

    print("args.seed: ", args.seed)
    print('length ===', len(hendrycks_math_ins))
    hendrycks_math_ins = hendrycks_math_ins[remainder::n_groups]

    print("processed length ===", len(hendrycks_math_ins))
    hendrycks_math_ins = hendrycks_math_ins * args.rep

    print('total length ===', len(hendrycks_math_ins))
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=batch_size)

    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.topp, max_tokens=2048)
    print('sampling =====', sampling_params)

    llm = LLM(
        model=model, 
        tensor_parallel_size=torch.cuda.device_count(), 
        dtype=torch.bfloat16, 
        enable_lora=True if args.lora_model_path is not None and args.lora_model_path != "None" else False,
        max_model_len=65528
        # max_num_batched_tokens=65528
    )

    print(f"reasoning {args.generate_n} responses for each prompt.")
    print(f"the number of prompt data is {len(prompt_data)}.")
    results = dict()
    N = args.generate_n
    for n in range(args.generate_n):
        results[n] = list()
        
        # res_completions = []
        for idx, prompt in enumerate(batch_hendrycks_math_ins):
            if isinstance(prompt, list):


                pass
            else:
                prompt = [prompt]
            
            if args.lora_model_path is not None and args.lora_model_path != "None":
                outputs = llm.generate(
                    prompt, 
                    sampling_params,
                    use_tqdm=True,
                    lora_request=LoRARequest("sql_adapter", 1, args.lora_model_path)
                )
            else:
                outputs = llm.generate(
                    prompt, 
                    sampling_params,
                    use_tqdm=True,
                )
            
            for output in outputs:
                generated_text = output.outputs[0].text
                results[n].append(generated_text)
            
    for ei, example in enumerate(prompt_data):
        example["responses"] = list()
        for n in range(args.generate_n):
            example["responses"].append({
                "role": "assistant",
                "content": "Let's think step by step.\nStep 1:" + results[n][ei],
            })
    
    return prompt_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_data_path", type=str, help="The path of prompt set")
    parser.add_argument("--iteration_stage", type=int, help="The stage idx of the iteration preference optimization, e.g., 1, 2, 3")
    parser.add_argument("--model_name_or_path", type=str, help="The model name or path of LLM")
    parser.add_argument("--model_name", type=str, help="The model name")
    parser.add_argument("--lora_model_path", type=str, default=None, help="The lora model path")
    parser.add_argument("--save_dir", type=str, default="outputs/math_bench", help="The save path (must equal to the benchmark directory, e.g., llm_bench)")
    parser.add_argument("--generate_n", type=int, default="6", help="The number of generated responses of each prompt")
    parser.add_argument("--cut_n", type=int, default=None, help="The number of examples cut off when debugging")
    parser.add_argument("--remainder", type=int, default=0) # index
    parser.add_argument("--n_groups", type=int, default=1)  # group number
    parser.add_argument("--batch_size", type=int, default=400)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    parser.add_argument("--prompt", type=str, default='qwen2-boxed-step')
    parser.add_argument("--temperature", type=float, default=0.9, help="The temperature")
    parser.add_argument("--topp", type=float, default=0.95, help="The topp")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rep", type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    """
    need input format:
    {
        "messages": [{"role": "xx", "content": "xx"}, xxx],
        "from": "xxx",
        "reference": {"role": "assistant", "content": "xxx"},
    }

    need save format
    return format:
    {
        "messages": [{"role": "xx", "content": "xx"}, xxx],
        "from": "xxx",
        "reference": {"role": "assistant", "content": "xxx"},
        "responses": [{"role": "assistant", "content": "xx"}, ...]
    }
    """
    args = parse_args()
    outputs = test_hendrycks_math(
        model=args.model_name_or_path,
        data_path=args.prompt_data_path, 
        remainder=args.remainder, 
        n_groups=args.n_groups, 
        batch_size=args.batch_size, 
        tensor_parallel_size=args.tensor_parallel_size, 
        args=args
    )

    save_path = os.path.join(args.save_dir, f"iteration_stage_{args.iteration_stage}", "prompt_responses/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("save results ...")
    # save reasoning results
    with open(os.path.join(save_path, "prompt_responses.json"), "a", encoding="utf-8") as fw:
        for example in tqdm(outputs):
            fw.write(json.dumps(example, ensure_ascii=False) + "\n")
