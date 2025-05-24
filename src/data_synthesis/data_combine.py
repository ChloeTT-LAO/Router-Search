import json
import os
from tqdm import tqdm
from ..prompt import r1_router_prompt, get_answer_prompt, get_final_answer_prompt
import re
import shutil


def extract_answer(answer):
    thought_pattern = r"<think>(.*?)</think>"
    subquestion_pattern = r"<sub-question>(.*?)</sub-question>"
    retriever_pattern = r"<ret>(.*?)</ret>"

    thought = re.search(thought_pattern, answer, re.DOTALL).group(1)
    subquestion = re.search(subquestion_pattern, answer, re.DOTALL).group(1)
    retriever = re.search(retriever_pattern, answer, re.DOTALL).group(1)
    retriever_list = [item.strip() for item in retriever.split(',')]

    return {'query': subquestion, 'retriever': retriever_list}


def convert(dataset, method):
    problems = []
    with open(os.path.join('/path/to/results/grpo-first', f"{dataset}_{method}.jsonl"), 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            problems.append(data)
    result = []

    for example in tqdm(problems, desc=f"Converting {dataset} {method}"):
        output = {'images': [], 'steps': []}
        for step in range(3):
            if example.get(f"input_{step}"):
                system_prompt = r1_router_prompt.replace("### Examples:", "\n")

                node_in = example[f"input_{step}"][0]['content']
                node_out = example[f"output_{step}"]
                single_step = {"step_type": "ret", 'problem': "", "system": system_prompt}
                for node in node_in:
                    if node["type"] == "text":
                        single_step['problem'] += node['text']
                    elif node["type"] == "image":
                        single_step['problem'] += "<image>"
                        image_name = node['image'].split('/')[-1]
                        image_save_name = f'/path/to/datasets/oven_images/{image_name}'
                        if not image_save_name in output['images']:
                            shutil.copy2(node['image'],
                                         f'/path/to/results/grpo/oven_images/{image_name}')
                            output['images'].append(image_save_name)

                answer = extract_answer(node_out)
                single_step['answer'] = answer
                output["steps"].append(single_step)

            if example.get(f"question_{step}"):
                system_prompt = get_answer_prompt
                node_in = example[f"question_{step}"][0]['content']
                node_out = example[f"answer_{step}"]
                single_step = {"step_type": "ans", 'problem': "", "system": system_prompt}
                for node in node_in:
                    if node["type"] == "text":
                        single_step['problem'] += node['text']
                    elif node["type"] == "image":
                        single_step['problem'] += "<image>"
                        image_name = node['image'].split('/')[-1]
                        image_save_name = f'/path/to/datasets/oven_images/{image_name}'
                        if not image_save_name in output['images']:
                            shutil.copy2(node['image'],
                                         f'/path/to/results/grpo/oven_images/{image_name}')
                            output['images'].append(image_save_name)

                single_step['answer'] = node_out
                output["steps"].append(single_step)

        system_prompt = get_final_answer_prompt
        node_in = example[f"question_3"][0]['content']

        if dataset == 'infoseek':
            answer = example['answer']
        elif dataset in ['openwikitqa', '2wikimultihopqa']:
            answer = [example['answer']]

        single_step = {"step_type": "ans", 'problem': "", "system": system_prompt}
        for node in node_in:
            if node["type"] == "text":
                single_step['problem'] += node['text']
            elif node["type"] == "image":
                single_step['problem'] += "<image>"
                image_name = node['image'].split('/')[-1]
                image_save_name = f'/path/to/datasets/oven_images/{image_name}'
                if not image_save_name in output['images']:
                    shutil.copy2(node['image'],
                                 f'/path/to/results/grpo/oven_images/{image_name}')
                    output['images'].append(image_save_name)

        single_step['answer'] = answer
        output["steps"].append(single_step)
        result.append(output)

    return result


if __name__ == "__main__":
    with open('/path/to/results/grpo/grpo_dataset_train_new_extra.jsonl', 'w') as o1, open(
            '/path/to/results/grpo/grpo_dataset_valid_new_extra.jsonl', 'w') as o2:
        for dataset in ['2wikimultihopqa', 'infoseek', 'openwikitqa']:
            datas = convert(dataset, 'r1-router')
            for data in datas[:int(len(datas) * 0.8)]:
                json.dump(data, o1, ensure_ascii=False)
                o1.write('\n')

            for data in datas[int(len(datas) * 0.8):]:
                json.dump(data, o2, ensure_ascii=False)
                o2.write('\n')
