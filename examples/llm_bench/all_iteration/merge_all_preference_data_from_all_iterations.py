import os
import json
import argparse
from tqdm import tqdm
from random import shuffle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #type是要传入的参数的数据类型  help是该参数的提示信息
    parser.add_argument("--data_path", type=str, help="The path saved all iterations preference data")
    parser.add_argument("--data_name", type=str, help="The name of the preference data")
    parser.add_argument("--iteration_total", type=int, help="The number of total iteration")
    parser.add_argument("--new_pair_data_name_to_datasetinfo", type=str, help="The name want to add in the dataset_info.json")
    args = parser.parse_args()

    # read all data and filter deducated prompt
    all_preference_data = list()
    prompt_response_list = list()
    total_num_before_filtering, total_num_final = 0, 0

    all_pair_files = [os.path.join(args.data_path, "pair_train_data.json")] # the original preference_data
    # add the new preference data devried from iterations after the uncertainty estimation
    all_pair_files += [os.path.join(args.data_path, f"iteration_{iter_idx}", "pair_train_data.json") for iter_idx in range(1, args.iteration_total + 1)]
    for file in all_pair_files:
        with open(file, "r", encoding="utf-8") as fr:
            for line in tqdm(fr.readlines()):
                total_num_before_filtering += 1
                example = json.loads(line)
                prompt = example["conversations"][0]["value"] if example["conversations"][0]["from"] == "human" else example["conversations"][1]["value"]
                response = example["chosen"]["value"] + example["rejected"]["value"]
                prompt_response = prompt + response
                if prompt_response not in prompt_response_list:
                    total_num_final += 1
                    all_preference_data.append(example)
                    prompt_response_list.append(prompt_response)
    
    print("total_num_before_filtering=", total_num_before_filtering)
    print("total_num_final=", total_num_final)

    shuffle(all_preference_data)

    # save data
    save_dir = os.path.join(args.data_path, "all_iteration")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "pair_train_data.json"), "w", encoding="utf-8") as fw:
        for example in tqdm(all_preference_data):
            fw.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print("update the dataset_info.json to enable the next iteration training. add name: {}".format(args.new_pair_data_name_to_datasetinfo))
    dataset_info = json.load(open("data/dataset_info.json", "r", encoding="utf-8"))
    assert args.new_pair_data_name_to_datasetinfo is not None and args.new_pair_data_name_to_datasetinfo != ""
    dataset_info[args.new_pair_data_name_to_datasetinfo] = {
      "file_name": os.path.join(args.data_name, "pair_train_data.json"),
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


"""
run script:
python3 examples/llm_bench/all_iteration/merge_all_preference_data_from_all_iterations.py \
--data_path=data/UltraFeedback \
--data_name=all_iteration/UltraFeedback \
--iteration_total=3 \
--new_pair_data_name_to_datasetinfo=ultrafeedback_all_iter_pair
"""
