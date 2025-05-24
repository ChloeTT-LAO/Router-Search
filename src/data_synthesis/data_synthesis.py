import argparse
import json
import os
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from ..prompt import r1_router_fewshot_msgs
import re
from qwen_vl_utils import process_vision_info

base_root = "path/to/datasets"
retrieve_root = "path/to/results/retrieve/{}"
infoseek_root = f"{base_root}/infoseek_images"

dataset_path = {
    "infoseek": 'infoseek_train.jsonl',
    'openwikitqa': 'openwikitqa_train.jsonl',
    '2wikimultihopqa': '2wikimultihopqa_train.jsonl',
}

model = None
processor = None
def initialize(dataset_name):
    global model, processor
    if dataset_name == 'infoseek':
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        return "qwen2.5-vl"
    else:
        return "r1-32b"


def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(v) for v in obj]
    else:
        return obj


def update_example(msg, method, dataset, model_name='qwen', do_sample=False):
    thought_pattern = r"<think>(.*?)</think>"
    subquestion_pattern = r"<sub-question>(.*?)</sub-question>"
    retriever_pattern = r"<ret>(.*?)</ret>"
    msgs = r1_router_fewshot_msgs[dataset].copy()
    msgs.extend(msg)
    cnt = 0
    while True:
        # print(msgs)
        cnt += 1
        answer = get_llm_response(model_name, msgs).lower()

        try:
            thought = re.search(thought_pattern, answer, re.DOTALL).group(1)
            subquestion = re.search(subquestion_pattern, answer, re.DOTALL).group(1)
            retriever = re.search(retriever_pattern, answer, re.DOTALL).group(1)
            retriever_list = [item.strip() for item in retriever.split(',')]

            for ret in retriever_list:
                if not ret in ['text retriever', 'text image retriever', 'table retriever', 'none']:
                    raise ValueError
            if 'rags' in method:
                if len(retriever_list) != 1:
                    raise ValueError

            if (subquestion == "none" and retriever != "none") or (subquestion != "none" and retriever == "none"):
                raise ValueError

            print(f"[INFO] Thought: {thought}")
            print(f"[INFO] Sub-question: {subquestion}")
            print(f"[INFO] Retriever: {retriever}\n")
            return answer, True
        except Exception as e:

            print(f"[ERROR] Output:{answer}")
            print(f"[ERROR] Exception:{e}, {cnt}\n")
            if cnt == 2:
                return "", False


def get_llm_response(model_name, msgs):
    if model_name == 'qwen2.5-vl':
        text_inputs = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(msgs)
        inputs = processor(
            text=[text_inputs],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=1.0, top_p=1.0,
                                       top_k=20)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
        return output_text[0]
    else:
        raise ValueError


def dataset_load(dataset, method, step=0, max_step=0, model_name='qwen'):
    problems = []
    if method == 'direct' or method == 'cot':
        root = '/path/to/datasets'
        file_name = dataset_path[dataset]
    else:
        root = "/path/to/retrieve"
        file_name = f"{dataset}_retrieve_{method}_step_{step}.jsonl"

    with open(os.path.join(root, file_name), 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            problems.append(data)
    return problems


def origin_question_gen(dataset, example):
    msgs = [{'role': 'user', 'content': []}]

    if dataset == 'infoseek':
        question = example['question']
        msgs[0]['content'].append({"type": "text", "text": f"Origin question: {example['question']}\n"})
        msgs[0]['content'].append({"type": "image", "image": os.path.join(infoseek_root, f"{example['image_id']}.jpg")})

    elif dataset in ["openwikitqa", "2wikimultihopqa"]:
        question = example['question']
        msgs[0]['content'].append({"type": "text", "text": f"Origin question: {example['question']}\n"})

    return msgs, question


def convert_ret_dict_to_str(ret_dict):
    ret_str = ""
    for ret in ret_dict.keys():
        for idx, psg in enumerate(ret_dict[ret]):
            ret_str += f"{ret}-{idx}: {psg}\n"
    return ret_str


def extract_answer(answer):
    thought_pattern = r"<think>(.*?)</think>"
    subquestion_pattern = r"<sub-question>(.*?)</sub-question>"
    retriever_pattern = r"<ret>(.*?)</ret>"

    thought = re.search(thought_pattern, answer, re.DOTALL).group(1)
    subquestion = re.search(subquestion_pattern, answer, re.DOTALL).group(1)
    retriever = re.search(retriever_pattern, answer, re.DOTALL).group(1)
    retriever_list = [item.strip() for item in retriever.split(',')]

    return {'thought': thought, 'query': subquestion, 'retriever': retriever_list}


def get_answer(msg, model_name='qwen', final=False):
    answer_pattern = r"(?:<answer>|</think>)(.*?)</answer>"

    if final:
        msgs = [{'role': 'system', 'content': [
            {"type": "text", "text": prompt.get_final_answer_prompt}
        ]}]
    else:
        msgs = [{'role': 'system', 'content': [
            {"type": "text", "text": prompt.get_answer_prompt}
        ]}]
    msgs.extend(msg)
    cnt = 0
    do_sample = False
    while True:
        cnt += 1
        answer = get_llm_response(model_name, msgs).lower()
        try:
            return re.search(answer_pattern, answer, re.DOTALL).group(1).split("<answer>")[-1], True
        except Exception as e:
            try:
                answer_pattern2 = r"<answer>(.*?)$"
                return re.search(answer_pattern2, answer, re.DOTALL).group(1).split("<answer>")[-1], True
            except:
                pass
            print(f"[ERROR] Exception: {e}")
            print(f"[ERROR] Answer: {answer}")
            print(f"[ERROR] Count: {cnt}")
            if cnt == 1:
                do_sample = True
            elif cnt == 2:
                return "", False


def gen(dataset, step, method, max_step, model_name):
    if step == 0:
        problems = dataset_load(dataset, "direct")
    else:
        problems = dataset_load(dataset, method, step)

    file_name = f'/path/to/results/grpo/{dataset}_{method}_step_{step + 1}_grpo.jsonl'
    cnt = 0
    with open(file_name, 'w', encoding='utf-8') as output_file:
        if step != 0:
            for example in problems:
                stop_earlier = False
                flag = True
                if step != 0:
                    rets = [f'ter', f'tir', f'tar']
                    if example.get(f'output_{step - 1}'):
                        node = extract_answer(example[f'output_{step - 1}'])
                        if node['query'].lower() == "none" or node['retriever'][0].lower() == 'none' or not node.get(
                                "query"):
                            answer = ""
                            msgs = None
                        else:
                            psg = example[f"ret_{step - 1}"]
                            image_path = None
                            ret_dict = {}
                            for ret in rets:
                                if psg.get(ret):
                                    ret_dict[ret] = psg[ret][:5]
                            # print(ret_dict)
                            if dataset == 'infoseek':
                                image_path = os.path.join(infoseek_root, f"{example['image_id']}.jpg")

                            msgs = [{'role': 'user', 'content': [
                                {"type": "text",
                                 "text": f"According to the related information searched, `ter` means this info is from text retriever, `tir` means this info is from text image retriever, `tar` means this info is from table retriever:{convert_ret_dict_to_str(ret_dict)}\n\n Give me the answer(with the format <answer></answer>) to the Question: {node['query']}"}
                            ]}]

                            if dataset == 'infoseek':
                                msgs[0]['content'].append({"type": "image", "image": image_path})

                            answer, flag = get_answer(msgs, model_name)
                            print(f"[INFO] Answer: {answer}")

                        if flag and msgs:
                            example[f'question_{step - 1}'] = msgs
                            example[f"answer_{step - 1}"] = answer

            if cnt < 1000 and flag:
                msgs, question = origin_question_gen(dataset, example)
                stop_earlier = False
                for st in range(0, step):
                    if not example.get(f'output_{st}'):
                        stop_earlier = True
                        break

                    node = extract_answer(example[f'output_{st}'])
                    if node[f'query'].lower() in ["", "none"] or node['retriever'][0].lower() == 'none' or not node.get(
                            "query"):
                        stop_earlier = True
                        break
                    msgs[0]['content'][0][
                        'text'] += f"Sub-question{st + 1}: {node[f'query']}\nAnswer{st + 1}: {example[f'answer_{st}']}\n"

                if step == max_step:
                    msgs[0]['content'][0]['text'] += f"Now, give me the final answer of the origin question: {question}"

                    answer, flag = get_answer(msgs, model_name, True)

                    if flag:
                        example[f"question_{step}"] = msgs
                        example[f'answer_{step}'] = answer

                        correct_answer = example['answer']

                        print(f"[INFO] Answer: {answer} Correct Answer: {correct_answer}")
                elif stop_earlier:
                    pass
                else:
                    output, flag = update_example(msgs, method, dataset)
                    if flag:
                        example[f'output_{step}'] = output
                        example[f'input_{step}'] = msgs

                if flag:
                    cnt += 1
                    json.dump(example, output_file, ensure_ascii=False)
                    output_file.write('\n')
                    output_file.flush()


def parse_arguments():
    parser = argparse.ArgumentParser(description="rag pipeline")
    parser.add_argument("--dataset_name", type=str, default="all")
    parser.add_argument("--method", type=str, default="mrrags")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="qwen2.5")
    parser.add_argument("--max_step", type=int, default=5)
    parser.add_argument("--final", action='store_true')
    return parser.parse_args()


def main():
    dataset_all = ['2wikimultihopqa', 'openwikitqa', 'infoseek']

    args = parse_arguments()
    method = args.method
    dataset_name = args.dataset_name
    model_name = args.model_name
    step = args.step
    if dataset_name == "all":
        dataset_name = dataset_all
    else:
        dataset_name = [dataset_name]

    print(f"[INFO] Generating answers with the following parameters:")
    print(f"[INFO] Dataset name: {dataset_name}")
    print(f"[INFO] Model name: {model_name}")

    for dataset in dataset_name:
        model_name = initialize(dataset)
        gen(dataset, step, method, 3, model_name)


if __name__ == "__main__":
    main()