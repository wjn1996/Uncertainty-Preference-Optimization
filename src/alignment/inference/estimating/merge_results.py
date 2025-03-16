
"""
Read all estimation results from multiple gpu, and save a new preference pair data and update the dataset_info.json for the next iteration preference optimzation.
"""
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
from random import shuffle


def change_json_key_name(example):
    return {
        "conversations": example["conversations"] if "from" in example["conversations"][0].keys() else [{"from": chat["role"].replace("user", "human"), "value": chat["content"]} for chat in example["conversations"]],
        "chosen": example["chosen"] if "from" in example["chosen"].keys() else {"from": example["chosen"]["role"].replace("user", "human"), "value": example["chosen"]["content"]},
        "rejected": example["rejected"] if "from" in example["rejected"].keys() else {"from": example["rejected"]["role"].replace("user", "human"), "value": example["rejected"]["content"]},
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #type是要传入的参数的数据类型  help是该参数的提示信息
    parser.add_argument("--save_path", type=str, help="The save dir")
    parser.add_argument("--new_pair_data_path_to_save", type=str, help="The final preference data need to save (must share the same dir with dataset_info.json)")
    parser.add_argument("--original_preference_data_path", type=str, help="The path of the labeled preference data")
    parser.add_argument("--new_pair_data_name_to_datasetinfo", type=str, help="The name want to update in dataset_info.json")
    parser.add_argument("--device_num", type=int, default=1, help="The number of device")

    args = parser.parse_args()

    save_path = args.save_path

    condidate_preference_pairs, preference_pairs_with_highest_chosen, preference_pairs_with_largest_reward_margin, mc_preference_pairs, easy_condidate_preference_pairs, hard_condidate_preference_pairs = list(), list(), list(), list(), list(), list()
    

    if not os.path.exists(os.path.join(save_path, "all_mc_prompt_responses_pairs_device.json")):
        # load estimation results
        for device_id in range(args.device_num):
            try:
                with open(os.path.join(save_path, f"all_candidate_prompt_responses_pairs_device_{device_id}.json"), "r", encoding="utf-8") as fr:
                    condidate_preference_pairs.extend([change_json_key_name(json.loads(line)) for line in fr.readlines()])
                
                with open(os.path.join(save_path, f"prompt_responses_pairs_with_highest_chosen_device_{device_id}.json"), "r", encoding="utf-8") as fr:
                    preference_pairs_with_highest_chosen.extend([change_json_key_name(json.loads(line)) for line in fr.readlines()])
                
                with open(os.path.join(save_path, f"prompt_responses_pairs_with_largest_reward_margin_device_{device_id}.json"), "r", encoding="utf-8") as fr:
                    preference_pairs_with_largest_reward_margin.extend([change_json_key_name(json.loads(line)) for line in fr.readlines()])
                
                with open(os.path.join(save_path, f"all_mc_prompt_responses_pairs_device_{device_id}.json"), "r", encoding="utf-8") as fr:
                    mc_preference_pairs.extend([change_json_key_name(json.loads(line)) for line in fr.readlines()])

                with open(os.path.join(save_path, f"easy_prompt_responses_pairs_device_{device_id}.json"), "r", encoding="utf-8") as fr:
                    easy_condidate_preference_pairs.extend([change_json_key_name(json.loads(line)) for line in fr.readlines()])

                with open(os.path.join(save_path, f"hard_prompt_responses_pairs_device_{device_id}.json"), "r", encoding="utf-8") as fr:
                    hard_condidate_preference_pairs.extend([change_json_key_name(json.loads(line)) for line in fr.readlines()])
            except:
                continue
    else:
        with open(os.path.join(save_path, f"all_candidate_prompt_responses_pairs_device.json"), "r", encoding="utf-8") as fr:
            condidate_preference_pairs.extend([change_json_key_name(json.loads(line)) for line in fr.readlines()])
        
        with open(os.path.join(save_path, f"prompt_responses_pairs_with_highest_chosen_device.json"), "r", encoding="utf-8") as fr:
            preference_pairs_with_highest_chosen.extend([change_json_key_name(json.loads(line)) for line in fr.readlines()])
        
        with open(os.path.join(save_path, f"prompt_responses_pairs_with_largest_reward_margin_device.json"), "r", encoding="utf-8") as fr:
            preference_pairs_with_largest_reward_margin.extend([change_json_key_name(json.loads(line)) for line in fr.readlines()])
        
        with open(os.path.join(save_path, f"all_mc_prompt_responses_pairs_device.json"), "r", encoding="utf-8") as fr:
            mc_preference_pairs.extend([change_json_key_name(json.loads(line)) for line in fr.readlines()])

        with open(os.path.join(save_path, f"easy_prompt_responses_pairs_device.json"), "r", encoding="utf-8") as fr:
            easy_condidate_preference_pairs.extend([change_json_key_name(json.loads(line)) for line in fr.readlines()])

        with open(os.path.join(save_path, f"hard_prompt_responses_pairs_device.json"), "r", encoding="utf-8") as fr:
            hard_condidate_preference_pairs.extend([change_json_key_name(json.loads(line)) for line in fr.readlines()])

    # load from original preference pairs
    with open(os.path.join(args.original_preference_data_path), "r", encoding="utf-8") as fr:
        original_preference_labeled_data = [json.loads(line) for line in fr.readlines()]
    
    shuffle(original_preference_labeled_data)
    original_preference_labeled_data = original_preference_labeled_data[:int(len(original_preference_labeled_data) * 0.4)]

    # random selecting same other kinds examples
    shuffle(preference_pairs_with_largest_reward_margin)
    sampled_preference_pairs_with_largest_reward_margin = preference_pairs_with_largest_reward_margin[:int(len(preference_pairs_with_largest_reward_margin) * 0.3)]
    shuffle(hard_condidate_preference_pairs)
    sampled_hard_condidate_preference_pairs = hard_condidate_preference_pairs[:int(len(hard_condidate_preference_pairs) * 0.2)]



    # save
    print("saving all sampled candidate preference pairs")
    with open(os.path.join(save_path, "all_candidate_prompt_responses_pairs_device.json"), "w", encoding="utf-8") as fw:
        for ei, example in enumerate(tqdm(condidate_preference_pairs)):
            fw.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print("saving all preference pairs where each pair has the highest chosen score")
    with open(os.path.join(save_path, "prompt_responses_pairs_with_highest_chosen_device.json"), "w", encoding="utf-8") as fw:
        for ei, example in enumerate(tqdm(preference_pairs_with_highest_chosen)):
            fw.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print("saving all preference pairs where each pair has the largest reward margin")
    with open(os.path.join(save_path, "prompt_responses_pairs_with_largest_reward_margin_device.json"), "w", encoding="utf-8") as fw:
        for ei, example in enumerate(tqdm(preference_pairs_with_largest_reward_margin)):
            fw.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print("saving the all preference pairs after mc evaluation")
    with open(os.path.join(save_path, "all_mc_prompt_responses_pairs_device.json"), "w", encoding="utf-8") as fw:
        for ei, example in enumerate(tqdm(mc_preference_pairs)):
            fw.write(json.dumps(example, ensure_ascii=False) + "\n")

    print("saving the easy preference pairs")
    with open(os.path.join(save_path, "easy_prompt_responses_pairs_device.json"), "w", encoding="utf-8") as fw:
        for ei, example in enumerate(tqdm(easy_condidate_preference_pairs)):
            fw.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print("saving the hard preference pairs")
    with open(os.path.join(save_path, "hard_prompt_responses_pairs_device.json"), "w", encoding="utf-8") as fw:
        for ei, example in enumerate(tqdm(hard_condidate_preference_pairs)):
            fw.write(json.dumps(example, ensure_ascii=False) + "\n")

    # merge same data 
    preference_data_for_next_iteration = original_preference_labeled_data + easy_condidate_preference_pairs + sampled_preference_pairs_with_largest_reward_margin + sampled_hard_condidate_preference_pairs
    shuffle(preference_data_for_next_iteration)

    # deducate
    prompt_list = list()

    print("saving the easy preference pairs with sampled original preference data")
    if not os.path.exists(os.path.join("data", args.new_pair_data_path_to_save)):
        os.makedirs(os.path.join("data", args.new_pair_data_path_to_save))


    num = 0
    with open(os.path.join("data", args.new_pair_data_path_to_save, "pair_train_data.json"), "w", encoding="utf-8") as fw:
        for ei, example in enumerate(tqdm(preference_data_for_next_iteration)):
            # print(example["conversations"])
            user_prompt = example["conversations"][-1]["value"]
            if user_prompt not in prompt_list:
                prompt_list.append(user_prompt)
                num += 1
            else:
                continue
            example_to_save = {
                "conversations": example["conversations"],
                "chosen": example["chosen"],
                "rejected": example["rejected"],
            }
            fw.write(json.dumps(example_to_save, ensure_ascii=False) + "\n")
    
    print(f"the number of final preference data need to train: {num}")
    
    print("update the dataset_info.json to enable the next iteration training. add name: {}".format(args.new_pair_data_name_to_datasetinfo))
    dataset_info = json.load(open("data/dataset_info.json", "r", encoding="utf-8"))
    dataset_info[args.new_pair_data_name_to_datasetinfo] = {
      "file_name": os.path.join(args.new_pair_data_path_to_save, "pair_train_data.json"),
      "ranking": True,
      "formatting": "sharegpt",
      "columns": {
        "messages": "conversations",
        "chosen": "chosen",
        "rejected": "rejected"
      }
    }
    with open("data/dataset_info.json", "w", encoding="utf-8") as fw:
        json.dump(dataset_info, fw, indent=4)
    

    