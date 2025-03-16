
"""
This file aims to construct candidate pairs based on the generated responses and assigned rewards.
    1. For each prompt, we sort the response based on the reward score.
    2. For the pair construction, we provide the following paradigms:
    - directly construct all candiadte pairs: used for estimation;
    - following ultrafeedback, the response with the best reward score is chosen (accepted), the rejected is random selected from the rest;

This file will save the following results:
    1. all candidate pairs: direct construct all candidate pairs with the estimated certainty score. this file can be used to sampled with or without the consideration of the certainty score.
    2. the paradigm of ultrafeedback, only consider rewards, do not consider certainty: for each prompt, choose only one pair, the response with the best reward score is chosen (accepted), the rejected is random selected from the rest;
    3. the paradigm of ultrafeedback, consider both certainty and reward scores: for each prompt, choose only one pair which is the best pair with highest certainty score and highest chosen rewards.
    4. the paradigm of ultrafeedback, only consider the reward margin, choose only on pair, where the chosen has the highest reward score and the rejected has the lowest reward score.
    
"""
import sys
import os
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
import argparse
import numpy as np
import json
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel 
from torch.utils.data import DataLoader, Dataset, DistributedSampler 
from random import shuffle, choices
from tqdm import tqdm
from typing import TYPE_CHECKING, List, Optional

from src.alignment.data import CLSDataCollatorWithPadding, get_dataset
from src.alignment.extras.ploting import plot_loss
from src.alignment.model import load_model, load_tokenizer, load_model_for_cls
from src.alignment.train.callbacks import fix_valuehead_checkpoint, LogCallback
from src.alignment.train.trainer_utils import create_modelcard_and_push
from src.alignment.train.estimator.metric import ComputeAccuracy
from src.alignment.train.estimator.trainer import CLSTrainer
from src.alignment.hparams import get_infer_args, get_train_args
from src.alignment.extras.read_yaml_args import read_yaml_file
from src.alignment.extras.constants import CLS_HEAD_SAFE_WEIGHTS_NAME
from src.alignment.inference.estimating.mc_evaluation import mc_estimation
from src.alignment.inference.estimating.sampler import sample_by_bald_difficulty, sample_by_bald_easiness

from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from peft import LoraConfig, TaskType, PeftModel, PeftConfig, get_peft_model
from safetensors import safe_open

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from src.alignment.hparams import DataArguments, FinetuningArguments


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

def get_estimation_prompt(query, accepted, rejected):
    return f"You are a good preference labeler. Give you one query from human and two candidate responses answered by the same assistant, your job is to find the better response which is more helpful, harmless for the human query.\n\nQ: {query}\n\n\nResponse #1: {accepted} \n\n\nReponse #2: {rejected}"

def construct_preference_pair_to_pair_and_cls_format(examples, tokenizer, device):
    """
    input format:
    {
        "messages": [{"role": "xx", "content": "xx"}, xxx],
        "from": "xxx",
        "reference": {"role": "assistant", "content": "xxx"},
        "responses": [{"role": "assistant", "content": "xx"}, ...],
        "reward_scores_of_reference": xx,
        "reward_scores_of_responses": [xx, xx, ..]
    }
    output format of pair for iteration preference training (sharegpt like):
    {
        "conversations": [{"from": "user", "value": "xxx"}],
        "chosen": {"from": "assistant", "value": "xxx"},
        "rejected": {"from": "assistant", "value": "xxx"},
        "chosen_reward_score": xxx,
        "rejected_reward_score": xxx,
    }
    output format of cls for estimation
    {
        "instruction": "xxx",
        "label": xxx,
        "chosen_reward_score": xxx,
        "rejected_reward_score": xxx,
    }
    return shape: List[dict], List[dict], List[dict] List[int]
    """
    condidate_preference_pairs, candidate_estimation_examples, candidate_tokenized_estimation_examples = list(), list(), list()
    preference_pairs_with_highest_chosen = list() # for each prompt, only choose one pair where chosen score is the highest
    preference_pairs_with_largest_reward_margin = list()
    prompt_responses_number = list()

    for example in tqdm(examples):
        reference, responses = example["reference"], example["responses"]
        reward_scores_of_reference = example["reward_scores_of_reference"]
        reward_scores_of_responses = example["reward_scores_of_responses"]
        all_candidate_responses = [reference] + responses
        all_candidate_responses_rewards = [reward_scores_of_reference] + reward_scores_of_responses
        all_candidate_responses_rewards = np.array(all_candidate_responses_rewards)

        # sort the rewards (high to low) and obtain the index
        sorted_indices = np.argsort(-all_candidate_responses_rewards)
        # only main four responses
        if len(sorted_indices) > 4:
            sorted_indices = np.concatenate((sorted_indices[:2], sorted_indices[-2:]))
        
        # for each prompt, only choose one pair where chosen score is the highest, these pair may not for mc evaluation
        selected_rejected_idx = choices(sorted_indices[1:])[0]
        preference_pairs_with_highest_chosen.append({
            "conversations": example["messages"],
            "chosen": {
                "from": all_candidate_responses[sorted_indices[0]]["role"],
                "value": all_candidate_responses[sorted_indices[0]]["content"],
            },
            "rejected": {
                "from": all_candidate_responses[selected_rejected_idx]["role"],
                "value": all_candidate_responses[selected_rejected_idx]["content"],
            },
            "chosen_reward_score": all_candidate_responses_rewards[sorted_indices[0]],
            "rejected_reward_score": all_candidate_responses_rewards[selected_rejected_idx],
        })
        #for each prompt, only chose one pair where the chosen score is the highest and the rejected score is the lowest, i.e, has a largest reward margin.
        preference_pairs_with_largest_reward_margin.append({
            "conversations": example["messages"],
            "chosen": {
                "from": all_candidate_responses[sorted_indices[0]]["role"],
                "value": all_candidate_responses[sorted_indices[0]]["content"],
            },
            "rejected": {
                "from": all_candidate_responses[sorted_indices[-1]]["role"],
                "value": all_candidate_responses[sorted_indices[-1]]["content"],
            },
            "chosen_reward_score": all_candidate_responses_rewards[sorted_indices[0]],
            "rejected_reward_score": all_candidate_responses_rewards[sorted_indices[-1]],
        })
        # obtain all candidate pairs
        preference_pairs, estimation_examples, tokenized_estimation_examples = list(), list(), list()
        for ii in range(0, len(sorted_indices)):
            i = sorted_indices[ii]
            for jj in range(ii + 1, len(sorted_indices)):
                j = sorted_indices[jj]
                if all_candidate_responses_rewards[i] > all_candidate_responses_rewards[j]:
                    preference_pairs.append({
                        "conversations": example["messages"],
                        "chosen": {
                            "from": all_candidate_responses[i]["role"],
                            "value": all_candidate_responses[i]["content"],
                        },
                        "rejected": {
                            "from": all_candidate_responses[j]["role"],
                            "value": all_candidate_responses[j]["content"],
                        },
                        "chosen_reward_score": all_candidate_responses_rewards[i],
                        "rejected_reward_score": all_candidate_responses_rewards[j],
                    })

                    instruction = get_estimation_prompt(example["messages"][0]["content"], all_candidate_responses[i]["content"], all_candidate_responses[j]["content"])
                    estimation_examples.append({
                        "instruction": instruction,
                        "label": 0, # because the chosen is always at the first position in the instruciton, so the label is fixed as 0
                    })
                    input_ids = tokenizer.encode(instruction)
                    input_ids = input_ids + [tokenizer.eos_token_id]
                    attention_mask = [1] * len(input_ids)
                    tokenized_estimation_examples.append({
                        "input_ids": torch.Tensor([input_ids]).long().to(device),
                        "attention_mask": torch.Tensor([attention_mask]).long().to(device),
                    })
                    
        condidate_preference_pairs.extend(preference_pairs)
        candidate_estimation_examples.extend(estimation_examples)
        candidate_tokenized_estimation_examples.extend(tokenized_estimation_examples)
        assert len(preference_pairs) == len(estimation_examples)
        prompt_responses_number.append(len(preference_pairs))

    print("total preference pairs num={}".format(len(condidate_preference_pairs)))
    
    return (
        condidate_preference_pairs, 
        candidate_estimation_examples,
        candidate_tokenized_estimation_examples, 
        prompt_responses_number, 
        preference_pairs_with_highest_chosen, 
        preference_pairs_with_largest_reward_margin
    )


def random_sampling(condidate_preference_pairs, candidate_estimation_examples, candidate_tokenized_estimation_examples, sampling_rate):
    """
    random sample from all candidate preference pairs
    """
    index_list = list(range(len(condidate_preference_pairs)))
    shuffle(index_list)
    index_list = index_list[:int(len(index_list) * args.sampling_rate)]
    sampled_condidate_preference_pairs, sampled_candidate_estimation_examples, sampled_candidate_tokenized_estimation_examples = list(), list(), list()
    for index in index_list:
        sampled_condidate_preference_pairs.append(condidate_preference_pairs[index])
        sampled_candidate_estimation_examples.append(candidate_estimation_examples[index])
        sampled_candidate_tokenized_estimation_examples.append(candidate_tokenized_estimation_examples[index])

    return sampled_condidate_preference_pairs, sampled_candidate_estimation_examples, sampled_candidate_tokenized_estimation_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #type是要传入的参数的数据类型  help是该参数的提示信息
    parser.add_argument("--prompt_responses_with_rewards_path", type=str, help="The path of prompt set")
    parser.add_argument("--iteration_stage", type=int, help="The stage idx of the iteration preference optimization, e.g., 1, 2, 3")
    parser.add_argument("--model_name_or_path", type=str, help="The model name or path of LLM")
    parser.add_argument("--model_name", type=str, help="The model name")
    parser.add_argument("--estimator_model", type=str, default=None, help="The lora model path of estimator model")
    parser.add_argument("--save_dir", type=str, default="outputs/llm_bench", help="The save path (must equal to the benchmark directory, e.g., llm_bench)")
    parser.add_argument("--sampling_rate", type=float, default=1.0, help="Sampling from all candidate_pairs")
    parser.add_argument("--bald_rate", type=float, default=0.5, help="Sampling BALD from all pairs after the mc evaluation")
    parser.add_argument("--cut_n", type=int, default=None, help="The number of examples cut off when debugging")
    parser.add_argument("--device_id", type=int, default=0, help="the device id of the curent inference session")
    parser.add_argument('--do_train', action='store_true', help="Whether to train the model")

    args = parser.parse_args()
    assert args.bald_rate <= 0.5 and args.bald_rate > 0, "the bald_rate must <=0.5 and > 0.0"

    args_json = {
        "model_name_or_path": args.model_name_or_path, # base backbone model path
        # "adapter_name_or_path": args.estimator_model, # value head model path
        "output_dir": args.save_dir,
        "template": args.model_name,
        "finetuning_type": "lora",
        "lora_target": "all",
        "stage": "es",
        "lora_dropout": 0.1, # used for MC Dropout
        # "do_train": True,
        # "reward_model_type": "inference",
    }

    save_path = os.path.join(args.save_dir, f"iteration_stage_{args.iteration_stage}", "prompt_responses_pair_with_uncertainty/")
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            pass

    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args_json)

    device_num = torch.cuda.device_count() # the number of available device

    # load tokenizer
    print("load tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    # load estimator model
    print("load estimator model ...")
    base_config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    peft_config = PeftConfig.from_pretrained(args.estimator_model)
    
    # load base backbone
    # 先通过训练模式加载一个带有lora的分类模型
    estimator_model = load_model_for_cls(tokenizer, model_args, finetuning_args, True, num_labels=2)

    # estimator_model的CSL的初始化的，所以需要将训练好的CLS head的值覆盖到estimator_model上
    # patch trained score head
    with safe_open(os.path.join(args.estimator_model, CLS_HEAD_SAFE_WEIGHTS_NAME), framework="pt", device="cpu") as f:
        clshead_state_dict = {key: f.get_tensor(key) for key in f.keys()}
    
    print("clshead_state_dict=", clshead_state_dict)
    estimator_model.load_state_dict(clshead_state_dict, strict=False)
    print("load tuned score cls head from {}".format(os.path.join(args.estimator_model, CLS_HEAD_SAFE_WEIGHTS_NAME)))
    
    # patch trained lora model
    # 由于还有lora参数，lora在backbone和CLS head都存在，所以最后在使用训练的lora覆盖整个estimator_model
    with safe_open(os.path.join(args.estimator_model, "adapter_model.safetensors"), framework="pt", device="cpu") as f:
        lora_state_dict = {key: f.get_tensor(key) for key in f.keys()}
    estimator_model.load_state_dict(lora_state_dict, strict=False)
    print("load tuned score lora from {}".format(os.path.join(args.estimator_model, "adapter_model.safetensors")))
    estimator_model.to(torch.float)    
    estimator_model.train() # open train mode because we should use MC drouput.
    # estimator_model = nn.DataParallel(estimator_model)
    # estimator_model = estimator_model.cuda()

    for name, param in estimator_model.named_parameters():
        print(f"{name}")

    # load prompt reponses generated by the policy at the last iteration stage
    print("load preference pairs and estimation cls examples ...")
    prompt_responses_with_rewards = load_prompt_responses_with_rewards(args.prompt_responses_with_rewards_path, cut_num=args.cut_n)

    (
        condidate_preference_pairs, 
        candidate_estimation_examples,
        candidate_tokenized_estimation_examples, 
        prompt_responses_number, 
        preference_pairs_with_highest_chosen, 
        preference_pairs_with_largest_reward_margin
    ) = construct_preference_pair_to_pair_and_cls_format(
        prompt_responses_with_rewards,
        tokenizer,
        device=estimator_model.device
    )

    

    
    if args.sampling_rate < 1.0:
        # sample from multiple pairs
        print("before sampling num: {}".format(len(condidate_preference_pairs)))
        condidate_preference_pairs, candidate_estimation_examples, candidate_tokenized_estimation_examples = random_sampling(
            condidate_preference_pairs, candidate_estimation_examples, candidate_tokenized_estimation_examples, args.sampling_rate
        )
        print("after sampling num: {}".format(len(condidate_preference_pairs)))

    # perform mc estimation
    y_mean, y_var, y_pred, y_T = mc_estimation(estimator_model, candidate_tokenized_estimation_examples, T=10, num_classes=2)

    # split the examples where the prediction label of estimator is not 0
    # Note: the chosen response generated by reward model of each pair are fixed at the first position, so that the correct label is fixed as 0.
    # therefore, if the prediction of the estimator is not 0, it means the predictions of the estimator and reward model are not consistent, the prediction of the current example must be a noise.
    # for the rest examples, we use bald sampling to select reliable pairs.
    print("filtering noise where the prediction is not correct ...")
    wait_for_bald_sampling_index_list = list()
    noise_index_list = list()
    clean_condidate_preference_pairs = list()
    for ei, (preference_pair, y_p) in enumerate(zip(condidate_preference_pairs, y_pred)):
        if y_p == 0:
            wait_for_bald_sampling_index_list.append(ei)
            clean_condidate_preference_pairs.append(preference_pair)
        else:
            noise_index_list.append(ei)
    print("wait_for_bald_sampling_index_list=", wait_for_bald_sampling_index_list)
    
    y_mean = y_mean[wait_for_bald_sampling_index_list]
    y_var = y_var[wait_for_bald_sampling_index_list]
    y_pred = y_pred[wait_for_bald_sampling_index_list]
    y_T = y_T[:, wait_for_bald_sampling_index_list]

    assert len(y_mean) == len(y_pred) and len(y_pred) == len(clean_condidate_preference_pairs) and len(y_T[0]) == len(y_mean)

    # print("y_mean=", y_mean)
    # print("y_var=", y_var)
    # print("y_pred=", y_pred)
    # print("y_T=", y_T)
    
    print("total pair num: {}".format(len(condidate_preference_pairs)))
    print("wait for bald sampling pair num: {}".format(len(wait_for_bald_sampling_index_list)))
    print("noise pair num: {}".format(len(noise_index_list)))
    print("pass rate: {}".format(1 - len(noise_index_list) / len(condidate_preference_pairs)))

    print("sampling easy and hard preference pairs based on bald ...")

    post_sample_num = int(len(wait_for_bald_sampling_index_list) * args.bald_rate) # sampling num

    easy_condidate_preference_pairs, easy_y_mean, easy_y_var, easy_y_pred, easy_BALD_acq, easy_p_norm = sample_by_bald_easiness(
        X=clean_condidate_preference_pairs, 
        y_mean=y_mean, 
        y_var=y_var, 
        y=y_pred, 
        num_samples=post_sample_num, 
        num_classes=2, # DO NOT CHANGE
        y_T=y_T)
    
    hard_condidate_preference_pairs, hard_y_mean, hard_y_var, hard_y_pred, hard_BALD_acq, hard_p_norm = sample_by_bald_difficulty(
        X=clean_condidate_preference_pairs, 
        y_mean=y_mean, 
        y_var=y_var, 
        y=y_pred, 
        num_samples=post_sample_num, 
        num_classes=2, # DO NOT CHANGE
        y_T=y_T)
    
    assert len(easy_condidate_preference_pairs) == len(easy_y_pred)
    assert len(hard_condidate_preference_pairs) == len(hard_y_pred)

    # save
    print("saving all sampled candidate preference pairs")
    with open(os.path.join(save_path, f"all_candidate_prompt_responses_pairs_device_{args.device_id}.json"), "w", encoding="utf-8") as fw:
        for ei, example in enumerate(tqdm(condidate_preference_pairs)):
            fw.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print("saving all preference pairs where each pair has the highest chosen score")
    with open(os.path.join(save_path, f"prompt_responses_pairs_with_highest_chosen_device_{args.device_id}.json"), "w", encoding="utf-8") as fw:
        for ei, example in enumerate(tqdm(preference_pairs_with_highest_chosen)):
            fw.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print("saving all preference pairs where each pair has the largest reward margin")
    with open(os.path.join(save_path, f"prompt_responses_pairs_with_largest_reward_margin_device_{args.device_id}.json"), "w", encoding="utf-8") as fw:
        for ei, example in enumerate(tqdm(preference_pairs_with_largest_reward_margin)):
            fw.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print("saving the all preference pairs after mc evaluation")
    with open(os.path.join(save_path, f"all_mc_prompt_responses_pairs_device_{args.device_id}.json"), "w", encoding="utf-8") as fw:
        for ei, example in enumerate(tqdm(clean_condidate_preference_pairs)):
            example["uncertainty_evaluation"] = {
                "y_mean": y_mean[ei].tolist(),
                "y_var": y_var[ei].tolist(),
                "y_pred": y_pred[ei].tolist(),
            }
            fw.write(json.dumps(example, ensure_ascii=False) + "\n")

    print("saving the easy preference pairs")
    with open(os.path.join(save_path, f"easy_prompt_responses_pairs_device_{args.device_id}.json"), "w", encoding="utf-8") as fw:
        for ei, example in enumerate(tqdm(easy_condidate_preference_pairs)):
            example["uncertainty_evaluation"] = {
                "y_mean": easy_y_mean[ei].tolist(),
                "y_var": easy_y_var[ei].tolist(),
                "y_pred": easy_y_pred[ei].tolist(),
                "bald_score": easy_BALD_acq[ei].tolist(),
                "p_norm": easy_p_norm[ei].tolist(),
            }
            fw.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print("saving the hard preference pairs")
    with open(os.path.join(save_path, f"hard_prompt_responses_pairs_device_{args.device_id}.json"), "w", encoding="utf-8") as fw:
        for ei, example in enumerate(tqdm(hard_condidate_preference_pairs)):
            example["uncertainty_evaluation"] = {
                "y_mean": hard_y_mean[ei].tolist(),
                "y_var": hard_y_var[ei].tolist(),
                "y_pred": hard_y_pred[ei].tolist(),
                "bald_score": hard_BALD_acq[ei].tolist(),
                "p_norm": hard_p_norm[ei].tolist(),
            }
            fw.write(json.dumps(example, ensure_ascii=False) + "\n")

    

