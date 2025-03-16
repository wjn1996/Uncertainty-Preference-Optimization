#!/bin/bash
MAIN_DIR="UncertaintyDPO/"
cd $MAIN_DIR

source activate xx

echo $(pwd)
echo $PYTHONPATH
echo $PATH
which python3.10
which python3

# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
echo ${MASTER_PORT}
torchrun --nnodes 1 --node_rank 0 --nproc_per_node 8 --master_addr 127.0.0.1 --master_port $MASTER_PORT src/alignment/train/estimator/main.py examples/llm_bench/init_stage/args/zephyr_es.yaml

