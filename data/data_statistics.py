from transformers import AutoTokenizer
import json
from tqdm import tqdm


if __name__ == "__main__":
    ultra_feedback_dir = "UltraFeedback"
    tokenizer = AutoTokenizer.from_pretrained("alignment-handbook/zephyr-7b-sft-full")
    
    examples = list()
    final_examples = list()
    with open("./UltraFeedback/cls_train_data.json", "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            examples.append(json.loads(line))

    length_dict = {
        "<1000": 0,
        "1000~2000": 0,
        "2000~3000": 0,
        ">3000":0
    }
    for example in tqdm(examples):
        tokens = tokenizer.tokenize(example["instruction"])
        if len(tokens) <= 1000:
            length_dict["<1000"] += 1
        elif len(tokens) <= 2000:
            length_dict["1000~2000"] += 1
        elif len(tokens) <= 3000:
            length_dict["2000~3000"] += 1
        else:
            length_dict[">3000"] += 1
        if len(tokens) <= 2000:
            final_examples.append(example)

    
    print(length_dict)
    print("len(final_examples)=", len(final_examples))

    with open("./UltraFeedback/cls_train_data.json", "w", encoding="utf-8") as fw:
        for example in tqdm(final_examples):
            fw.write(json.dumps(example, ensure_ascii=False) + "\n")
