export CUDA_VISIBLE_DEVICES="4,5"

torchrun --nnodes 1 --node_rank 0 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 21013 src/alignment/train/estimator/main.py examples/llm_bench/init_stage/args/zephyr_es.yaml

