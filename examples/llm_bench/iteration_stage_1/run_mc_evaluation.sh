#!/bin/bash

prompt_responses_with_rewards_path=$1
model_name_or_path=$2
model_name=$3
estimator_model=$4
iteration_stage=$5
sampling_rate=$6
device_id=$7

python3 src/alignment/inference/estimating/main.py \
--prompt_responses_with_rewards_path=$prompt_responses_with_rewards_path \
--model_name_or_path=$model_name_or_path \
--model_name=$model_name \
--estimator_model=$estimator_model \
--iteration_stage=$iteration_stage \
--sampling_rate=$sampling_rate \
--device_id=$device_id