export CUDA_VISIBLE_DEVICES="0,1,2,3"

torchrun --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_addr 127.0.0.1 --master_port 21002 src/alignment/train/rm/main.py examples/llm_bench/init_stage/args/zephyr_rm.yaml

