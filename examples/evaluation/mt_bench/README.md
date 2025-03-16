Following the instruciton as FactChat: https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge

You first download the question.jsonl at `data/mt_bench`
You can generate all responses by run:
```
bash examples/mt_bench/run_pred.sh
```
and the result will be saved at `examples/mt_bench_outputs`

The, you move this file into the repository of FastChat at the folder `fastchat/llm_judge/data/mt_bench/model_answer`.
Then run the judgment by:
```
export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call]
```
After that, you can submit your `gpt-4-single.jsonl` on huggingface, and run the code at: https://colab.research.google.com/drive/15O3Y8Rxq37PuMlArE291P4OC6ia37PQK#scrollTo=Vp5YVvaTpIiX to get the figure.

