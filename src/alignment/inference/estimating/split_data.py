import sys
import os
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import json
import argparse
from tqdm import tqdm
import random

def random_sampling(examples, sampling_rate: float = 1.0):
    selected_examples = random.sample(examples, int(len(examples) * sampling_rate))
    return selected_examples

def load_prompt_responses_with_rewards(data_path, cut_num: int=None):
    """
    return format:
    {
        "messages": [{"role": "xx", "content": "xx"}, xxx],
        "from": "xxx",
        "reference": {"role": "assistant", "content": "xxx"},
        "responses": [{"role": "assistant", "content": "xx"}, ...],
        "reward_scores_of_reference": xx,
        "reward_scores_of_responses": [xx, xx, ..]
    }
    """
    examples = list()
    with open(data_path, "r", encoding="utf-8") as fr:
        for line in tqdm(fr.readlines()):
            examples.append(json.loads(line))
    if cut_num is not None:
        examples = examples[:cut_num]
    return examples

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #type是要传入的参数的数据类型  help是该参数的提示信息
    parser.add_argument("--prompt_responses_with_rewards_path", type=str, help="The path of prompt set")
    parser.add_argument("--sampling_rate", type=float, default=1.0, help="Sampling from all prompt responses with rewards")
    parser.add_argument("--device_num", type=int, default=1, help="The number of device")

    args = parser.parse_args()

    save_path = "/".join(args.prompt_responses_with_rewards_path.split("/")[:-1])
    save_file_name = ".".join(args.prompt_responses_with_rewards_path.split("/")[-1].split(".")[:-1])
    save_file_type = args.prompt_responses_with_rewards_path.split("/")[-1].split(".")[-1]
    
    prompt_responses_with_rewards = load_prompt_responses_with_rewards(args.prompt_responses_with_rewards_path)

    if args.sampling_rate < 1.0:
        prompt_responses_with_rewards = random_sampling(prompt_responses_with_rewards, args.sampling_rate)
        print("random sampling num {}".format(len(prompt_responses_with_rewards)))

    # split the data based on device number
    example_number_per_device = int((len(prompt_responses_with_rewards) - 1) / args.device_num) + 1
    example_idx_span_per_device = [[device_id * example_number_per_device, (device_id + 1) * example_number_per_device] for device_id in range(0, args.device_num)]

    for ei, (s, e) in enumerate(example_idx_span_per_device):
        with open(os.path.join(save_path, f"{save_file_name}_device_{ei}.{save_file_type}"), "w", encoding="utf-8") as fw:
            for example in tqdm(prompt_responses_with_rewards[s: e]):
                fw.write(json.dumps(example, ensure_ascii=False) + "\n")