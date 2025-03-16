#!/bin/bash

MAIN_DIR="UncertaintyDPO/"
cd $MAIN_DIR

source activate xxx


MASTER_PORT=$(shuf -n 1 -i 10000-65535)
echo ${MASTER_PORT}

MODEL_NAME=zephyr
BASE_MODEL_PATH=alignment-handbookm/zephyr-7b-sft-full
SAVE_PATH=outputs/llm_bench
STAGE_IDX=1 # DO NOT CHANGE

DO_GENERATING=true # true or false
DO_REWARDING=true # true or false
DO_ESTIMATING=true # true or false

## First: sampling 35k prompt from SFT and pair data, and load the policy model at init_stage to inference.

if [ "$DO_GENERATING" = "true" ]; then

    export CUDA_VISIBLE_DEVICES="4,5"
    GPU_NUM=2

    POLICY_MODEL_PATH=outputs/llm_bench/init_stage/policy_dpo/zephyr-7b-sft-full/lora/checkpoint-4000
    PROMPT_DATA_PATH=data/prompt_set/sampled_prompt_data_for_iter$STAGE_IDX.json
    GENERATE_N=6


    python3 src/alignment/inference/reasoning/main.py \
    --prompt_data_path=$PROMPT_DATA_PATH \
    --model_name_or_path=$BASE_MODEL_PATH \
    --model_name=$MODEL_NAME \
    --lora_model_path=$POLICY_MODEL_PATH \
    --save_dir=$SAVE_PATH \
    --generate_n=$GENERATE_N \
    --iteration_stage=$STAGE_IDX \
    --use_vllm \
    --use_chat_template \
    --block_num=10

fi

## Second: assigning reward to the generated responses of each prompt.


if [ "$DO_REWARDING" = "true" ]; then

    export CUDA_VISIBLE_DEVICES="7"
    GPU_NUM=1

    REWARD_MODEL_PATH=outputs/llm_bench/init_stage/rm/zephyr-7b-sft-full/lora/checkpoint-3000/
    PROMPT_DATA_PATH=outputs/llm_bench/iteration_stage_$STAGE_IDX/prompt_responses/prompt_responses.json

    python3 src/alignment/inference/rewarding/main.py \
    --prompt_responses_path=$PROMPT_DATA_PATH \
    --model_name_or_path=$BASE_MODEL_PATH \
    --model_name=$MODEL_NAME \
    --reward_model=$REWARD_MODEL_PATH \
    --save_dir=$SAVE_PATH \
    --iteration_stage=$STAGE_IDX

fi

## Third: construct candidate pair based on the assigned reward, and estimating uncertainty of each pair to devide two set, i.e., easy certainty set and hard uncertainty set.

if [ "$DO_ESTIMATING" = "true" ]; then

    GPU_NUM=8
    PROMPT_DATA_RESPONSES_WITH_REWARDS_PATH=outputs/llm_bench/iteration_stage_$STAGE_IDX/prompt_responses_with_rewards
    ESTIMATOR_MODEL=outputs/llm_bench/init_stage/estimator/zephyr-7b-sft-full/lora/checkpoint-2500
    SAMPLE_RATE=0.25
    # SAMPLE_RATE=0.27

    python3 src/alignment/inference/estimating/split_data.py \
    --prompt_responses_with_rewards_path=$PROMPT_DATA_RESPONSES_WITH_REWARDS_PATH/prompt_responses_with_rewards.json \
    --device_num=$GPU_NUM

    # Please choose GPU_NUM gpus
    CUDA_VISIBLE_DEVICES=0 nohup bash examples/llm_bench/iteration_stage_$STAGE_IDX/run_mc_evaluation.sh $PROMPT_DATA_RESPONSES_WITH_REWARDS_PATH/prompt_responses_with_rewards_device_0.json $BASE_MODEL_PATH $MODEL_NAME $ESTIMATOR_MODEL $STAGE_IDX $SAMPLE_RATE 0 > logs/estimator/zephyr-7b-sft-full-estimator_iter1_device0.log &
    CUDA_VISIBLE_DEVICES=1 nohup bash examples/llm_bench/iteration_stage_$STAGE_IDX/run_mc_evaluation.sh $PROMPT_DATA_RESPONSES_WITH_REWARDS_PATH/prompt_responses_with_rewards_device_1.json $BASE_MODEL_PATH $MODEL_NAME $ESTIMATOR_MODEL $STAGE_IDX $SAMPLE_RATE 1 > logs/estimator/zephyr-7b-sft-full-estimator_iter1_device1.log &
    CUDA_VISIBLE_DEVICES=2 nohup bash examples/llm_bench/iteration_stage_$STAGE_IDX/run_mc_evaluation.sh $PROMPT_DATA_RESPONSES_WITH_REWARDS_PATH/prompt_responses_with_rewards_device_2.json $BASE_MODEL_PATH $MODEL_NAME $ESTIMATOR_MODEL $STAGE_IDX $SAMPLE_RATE 2 > logs/estimator/zephyr-7b-sft-full-estimator_iter1_device2.log &
    CUDA_VISIBLE_DEVICES=3 nohup bash examples/llm_bench/iteration_stage_$STAGE_IDX/run_mc_evaluation.sh $PROMPT_DATA_RESPONSES_WITH_REWARDS_PATH/prompt_responses_with_rewards_device_3.json $BASE_MODEL_PATH $MODEL_NAME $ESTIMATOR_MODEL $STAGE_IDX $SAMPLE_RATE 3 > logs/estimator/zephyr-7b-sft-full-estimator_iter1_device3.log &
    CUDA_VISIBLE_DEVICES=4 nohup bash examples/llm_bench/iteration_stage_$STAGE_IDX/run_mc_evaluation.sh $PROMPT_DATA_RESPONSES_WITH_REWARDS_PATH/prompt_responses_with_rewards_device_4.json $BASE_MODEL_PATH $MODEL_NAME $ESTIMATOR_MODEL $STAGE_IDX $SAMPLE_RATE 4 > logs/estimator/zephyr-7b-sft-full-estimator_iter1_device4.log &
    CUDA_VISIBLE_DEVICES=5 nohup bash examples/llm_bench/iteration_stage_$STAGE_IDX/run_mc_evaluation.sh $PROMPT_DATA_RESPONSES_WITH_REWARDS_PATH/prompt_responses_with_rewards_device_5.json $BASE_MODEL_PATH $MODEL_NAME $ESTIMATOR_MODEL $STAGE_IDX $SAMPLE_RATE 5 > logs/estimator/zephyr-7b-sft-full-estimator_iter1_device5.log &
    CUDA_VISIBLE_DEVICES=6 nohup bash examples/llm_bench/iteration_stage_$STAGE_IDX/run_mc_evaluation.sh $PROMPT_DATA_RESPONSES_WITH_REWARDS_PATH/prompt_responses_with_rewards_device_6.json $BASE_MODEL_PATH $MODEL_NAME $ESTIMATOR_MODEL $STAGE_IDX $SAMPLE_RATE 6 > logs/estimator/zephyr-7b-sft-full-estimator_iter1_device6.log &
    CUDA_VISIBLE_DEVICES=7 nohup bash examples/llm_bench/iteration_stage_$STAGE_IDX/run_mc_evaluation.sh $PROMPT_DATA_RESPONSES_WITH_REWARDS_PATH/prompt_responses_with_rewards_device_7.json $BASE_MODEL_PATH $MODEL_NAME $ESTIMATOR_MODEL $STAGE_IDX $SAMPLE_RATE 7 > logs/estimator/zephyr-7b-sft-full-estimator_iter1_device7.log &
    
    wait

    python3 src/alignment/inference/estimating/merge_results.py \
    --save_path=outputs/llm_bench/iteration_stage_$STAGE_IDX/prompt_responses_pair_with_uncertainty \
    --new_pair_data_path_to_save=UltraFeedback/iteration_$STAGE_IDX \
    --original_preference_data_path=data/UltraFeedback/pair_train_data.json \
    --new_pair_data_name_to_datasetinfo=ultrafeedback_iter$STAGE_IDX_pair \
    --device_num=$GPU_NUM

fi