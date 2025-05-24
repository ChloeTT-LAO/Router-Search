from ..evaluate import evaluate
import os
import json
import numpy as np
import argparse


def dataset_evaluate(dataset, method, model_name):
    rec_list = []
    f1_list = []
    acc_list = []

    with open(os.path.join('/path/to/results/grpo', f"{dataset}_{method}_step_4_grpo.jsonl"), 'r') as f, open(
            os.path.join('/path/to/results/grpo', f"{dataset}_{method}.jsonl"), 'w') as o:
        for line in f:
            data = json.loads(line.strip())
            output = data['answer_3']

            if dataset == 'infoseek':
                question = data['question']
                answer = data['answer']
            elif dataset == "hotpotqa":
                question = data['question']
                answer = [data['answer']]
            elif dataset == 'dyn_vqa':
                question = data['question']
                answer = data['answer']
            elif dataset in ['openwikitqa', "2wikimultihopqa"]:
                answer = [data['answer']]

            if output == "":
                rec_list.append(0)
                f1_list.append(0)
                acc_list.append(0)
                continue

            _, rec, f1 = evaluate(output, answer)
            acc = 1 if rec == 1 else 0
            # acc = 1 if llm_acc(question, output, answer) else 0

            rec_list.append(rec)
            f1_list.append(f1)
            acc_list.append(acc)

            if acc == 1:
                json.dump(data, o, ensure_ascii=False)
                o.write('\n')

    print(f"{dataset} eval info:")
    print('Token Recall: ', round(100 * float(sum(rec_list)) / len(rec_list), 2))
    print('Token F1: ', round(100 * float(sum(f1_list)) / len(f1_list), 2))
    print('Token Acc: ', round(100 * float(sum(acc_list)) / len(acc_list), 2))


def parse_arguments():
    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--dataset_name", type=str, default="hotpotqa")
    parser.add_argument("--method", type=str, default="cot")
    parser.add_argument("--model_name", type=str, default="qwen")
    return parser.parse_args()


def main():
    args = parse_arguments()
    method = args.method
    dataset_name = args.dataset_name
    model_name = args.model_name

    if dataset_name == "all":
        dataset_name = ['2wikimultihopqa', 'infoseek', 'openwikitqa']
    else:
        dataset_name = [dataset_name]

    if method == "all":
        method_list = 'r1-router'
    else:
        method_list = [method]

    for method in method_list:
        for dataset in dataset_name:
            dataset_evaluate(dataset, method, model_name)


if __name__ == "__main__":
    main()