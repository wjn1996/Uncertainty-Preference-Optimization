export CUDA_VISIBLE_DEVICES="2"

MODEL_NAME=zephyr

### [baseline] zephyr-7b-sft-full
# MODLE_VERSION=SFT
# BASE_MODEL_PATH=alignment-handbook/zephyr-7b-sft-full
# LORA_PATH=None

### [baseline]: zephyr-7b-sft-full + dpo
# MODLE_VERSION=DPO
# BASE_MODEL_PATH=alignment-handbook/zephyr-7b-sft-full
# LORA_PATH=outputs/llm_bench/init_stage/policy_dpo/zephyr-7b-sft-full/lora/checkpoint-4000

### [ours]: baseline + iter1-dpo
# MODLE_VERSION=UPO-iter1
# BASE_MODEL_PATH=alignment-handbook/zephyr-7b-sft-full
# LORA_PATH=outputs/llm_bench/iteration_stage_1/policy_dpo/zephyr-7b-sft-full/lora/checkpoint-2500

# MODLE_VERSION=UPO-iter1-2
# BASE_MODEL_PATH=alignment-handbook/zephyr-7b-sft-full
# LORA_PATH=outputs/llm_bench/iteration_stage_1/policy_dpo/zephyr-7b-sft-full/lora/checkpoint-2500

# MODLE_VERSION=UPO-iter1-v2
# BASE_MODEL_PATH=alignment-handbook/zephyr-7b-sft-full
# LORA_PATH=outputs/llm_bench/iteration_stage_1/policy_dpo/zephyr-7b-sft-full-v2/lora/checkpoint-4000


### [ours]: baseline + iter2-dpo
# MODLE_VERSION=UPO-iter2
# BASE_MODEL_PATH=alignment-handbook/zephyr-7b-sft-full
# LORA_PATH=outputs/llm_bench/iteration_stage_2/policy_dpo/zephyr-7b-sft-full/lora/checkpoint-1500


# MODLE_VERSION=UPO-iter2-v2
# BASE_MODEL_PATH=alignment-handbook/zephyr-7b-sft-full
# LORA_PATH=outputs/llm_bench/iteration_stage_2/policy_dpo/zephyr-7b-sft-full-v2/lora/checkpoint-1800

### [ours]: baseline + iter3-dpo
# MODLE_VERSION=UPO-iter3
# BASE_MODEL_PATH=alignment-handbook/zephyr-7b-sft-full
# LORA_PATH=outputs/llm_bench/iteration_stage_3/policy_dpo/zephyr-7b-sft-full/lora/checkpoint-2000

# MODLE_VERSION=UPO-iter3-v2
# BASE_MODEL_PATH=alignment-handbook/zephyr-7b-sft-full
# LORA_PATH=outputs/llm_bench/iteration_stage_3/policy_dpo/zephyr-7b-sft-full-v2/lora/checkpoint-1800

### [ours]: baseline + all_iter-dpo

# MODLE_VERSION=UPO-alliter-v2
# BASE_MODEL_PATH=alignment-handbook/zephyr-7b-sft-full
# LORA_PATH=outputs/llm_bench/all_iteration/policy_dpo/zephyr-7b-sft-full-v2/lora/checkpoint-4500

MODLE_VERSION=UPO-alliter-v3
BASE_MODEL_PATH=alignment-handbook/zephyr-7b-sft-full
LORA_PATH=outputs/llm_bench/all_iteration/policy_dpo/zephyr-7b-sft-full-v3/lora/checkpoint-4500



python3 examples/evaluation/mt_bench/prediction.py \
--model_name_or_path=$BASE_MODEL_PATH \
--method_version=$MODLE_VERSION \
--model_name=$MODEL_NAME \
--lora_model_path=$LORA_PATH \
--temperature=0.8 \
--topp=0.9