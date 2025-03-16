执行run_pred.sh获得预测结果，在outputs下存储预测结果
前往https://github.com/tatsu-lab/alpaca_eval按照步骤完成评测。

dowload the alpaca_eval code
run python3 set.py install

run the script, e.g.:
```
alpaca_eval --model_outputs "examples/evaluation/alpaca_eval/outputs/alpaca_zephyr_SFT.json" --annotators_config "weighted_alpaca_eval_gpt4_turbo"
```