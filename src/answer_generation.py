import math
import re
import numpy as np
import argparse
import json
import os
import torch
from PIL import Image
import random

import prompt
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

model = None
tokenizer = None
processor = None
base_root = "path/to/datasets"
retrieve_root = "path/to/results/retrieve/{}"
dyn_vqa_root = f"{base_root}/dyn_vqa"
infoseek_root = f"{base_root}/infoseek_images"
webqa_root = f"{base_root}/webqa"

dataset_path = {
    'dyn_vqa': 'DynVQA_en.202412.jsonl',
    "infoseek": 'infoseek_val_3000.jsonl',
    'openwikitqa': 'openwikitqa_valid.jsonl',
    '2wikimultihopqa': '2wikimultihopqa.jsonl',
    'webqa': 'webqa_dev.jsonl',
    'tabfact': 'tabfact_dev.jsonl',
}
k = 5


def initialize_llm(model_name):
    global model, processor, tokenizer
    if model_name == "qwen2.5":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    elif model_name == "r1-router":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "OpenBMB/R1-Router",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("OpenBMB/R1-Router")


def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(v) for v in obj]
    else:
        return obj


def dataset_load(dataset, method, step=0, max_step=0, model_name='qwen2.5', ret_type='text'):
    problems = []
    root = retrieve_root.format(model_name)

    if method == 'direct' or method == 'cot' or method == 'caption':
        root = base_root
        file_name = dataset_path[dataset]
    elif "mrrag" in method:
        file_name = f"{dataset}_retrieve_{method}_step_{step}.jsonl"

    with open(os.path.join(root, file_name), 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            problems.append(data)

    return problems


def get_llm_response(model_name, msgs, image=None, max_length=512):

    text_inputs = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(msgs)
    inputs = processor(
        text=[text_inputs],
        images=image_inputs,
        videos=None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_length, do_sample=True, temperature=1.0, top_p=1.0, top_k=20)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0]

def origin_question_gen(dataset, example):
    msgs = [{'role': 'user', 'content': []}]
            
    if dataset == 'infoseek':
        question = example['question']
        msgs[0]['content'].append({"type": "text", "text": f"Origin question: {example['question']}\n"}) 
        msgs[0]['content'].append({"type": "image", "image": os.path.join(infoseek_root, f"{example['image_id']}.jpg")}) 
    
    elif dataset in ["openwikitqa", "2wikimultihopqa", 'tabfact']:
        question = example['question']
        msgs[0]['content'].append({"type": "text", "text": f"Origin question: {example['question']}\n"})
    
    elif dataset == 'dyn_vqa':
        question = example['question']
        msgs[0]['content'].append({"type": "text", "text": f"Origin question: {example['question']}\n"}) 
        msgs[0]['content'].append({"type": "image", "image": os.path.join(dyn_vqa_root, f"{example['image_id']}.jpg")}) 
    
    elif dataset == 'webqa':
        question = example['question']
        msgs[0]['content'].append({"type": "text", "text": f"Origin question: {example['question']}\n"}) 
        msgs[0]['content'].append({"type": "image", "image": os.path.join(webqa_root, f"{example['image_id']}.jpg")}) 

    return msgs, question


def update_example(msg, method, dataset, model_name='qwen', do_sample=False):
    thought_pattern = r"<think>(.*?)</think>"
    subquestion_pattern = r"<sub-question>(.*?)</sub-question>"
    retriever_pattern = r"<ret>(.*?)</ret>"
    msgs = prompt.r1_router_fewshot_msgs[dataset].copy()

    if model_name == "r1-router":
        system = prompt.r1_router_prompt.replace("### Examples:", "")
        msgs = [{'role': 'system', 'content': [{'type': 'text', 'text': system}]}]

    msgs.extend(msg)
    cnt = 0
    while True:
        # print(msgs)
        cnt += 1
        answer = get_llm_response(model_name, msgs, max_length=2048).lower()
        
        try:
            thought = re.search(thought_pattern, answer, re.DOTALL).group(1).strip()
            subquestion = re.search(subquestion_pattern, answer, re.DOTALL).group(1).strip()
            retriever = re.search(retriever_pattern, answer, re.DOTALL).group(1).strip()
            retriever_list = [item.strip() for item in retriever.split(',')]

            for ret in retriever_list:
                if not ret in ['text retriever', 'text image retriever', 'table retriever', 'none']:
                    raise ValueError
            if retriever_list[0] == 'none' and subquestion != 'none':
                raise ValueError
            if retriever_list[0] != 'none' and subquestion == 'none':
                raise ValueError
            if 'rags' in method:
                if len(retriever_list) != 1:
                    raise ValueError
                
            print(f"[INFO] Thought: {thought}")
            print(f"[INFO] Sub-question: {subquestion}")
            print(f"[INFO] Retriever: {retriever}\n")
            return {'thought': thought, 'query': subquestion, 'retriever': retriever_list}
        except Exception as e:
            print(f"[ERROR] Output:{answer}")
            print(f"[ERROR] Exception:{e}, {cnt}\n")
            if cnt == 3:
                if "rags" in method:
                    formatt = """**You dont give me the answer with the correct format, please follow the Strict Output Format:
                            <Thought>
                            [Analyze the original question and determine the next required sub-question. Do NOT reveal answers or perform multi-hop reasoning.]
                            <Sub-question>
                            [Exactly ONE single-hop question. If no retrieval is needed, write 'None'.]
                            <Ret> 
                            [Choose 1 retriever from: Text Retriever, Text Image Retriever, Table Retriever. Write 'None' if <Sub-question> is 'None'.]
                            """
                elif "ragm" in method:
                    formatt = """You dont give me the answer with the correct format, please follow the Strict Output Format:
                            <Thought>
                            [Analyze the original question and determine the next required sub-question. Do NOT reveal answers or perform multi-hop reasoning.]
                            <Sub-question>
                            [Exactly ONE single-hop question. If no retrieval is needed, write 'None'.]
                            <Ret> 
                            [Choose 1-3 retrievers from: Text Retriever, Text Image Retriever, Table Retriever. Write 'None' if <Sub-question> is 'None'.]
                            """
                msgs.append({'role': 'assistant', 'content': [{"type": 'text', 'text': answer}]})
                msgs.append({'role': 'user', 'content': [{"type": 'text', 'text': formatt}]})
            elif cnt == 5:
                return {'thought': "none", 'query': "none", 'retriever': ["none"]}


def convert_ret_dict_to_str(ret_dict):
    ret_str = ""
    for ret in ret_dict.keys():
        for idx, psg in enumerate(ret_dict[ret]):
            ret_str += f"{ret}-{idx}: {psg}\n"
    return ret_str


def get_answer(msg, model_name='qwen', final=False):
    thought_pattern = r"<think>(.*?)</think>"
    answer_pattern = r"<answer>(.*?)</answer>"
    if final:
        msgs = [{'role': 'system', 'content':[
            {"type": "text", "text": prompt.get_final_answer_prompt}
        ]}]
    else:
        msgs = [{'role': 'system', 'content':[
            {"type": "text", "text": prompt.get_answer_prompt}
        ]}]
    msgs.extend(msg)
    cnt = 0
    while True:
        cnt += 1
        answer = get_llm_response(model_name, msgs, max_length=2048).lower().strip()
        try:
            thought = re.search(thought_pattern, answer, re.DOTALL).group(1).strip()
            ans = re.search(answer_pattern, answer, re.DOTALL).group(1).strip()
            return {"thought": thought, "answer": ans}
        except Exception as e:
            print(f"[ERROR] Exception: {e}")
            print(f"[ERROR] Answer: {answer}")
            print(f"[ERROR] Count: {cnt}")
            if cnt == 2:
                msgs = msg
            elif cnt == 3:
                return {"thought": "", "answer": answer}


def r1_router(dataset, step, method, max_step, model_name, count=1000):
    if step == 0:
        problems = dataset_load(dataset, "direct")
    else:
        problems = dataset_load(dataset, method, step, model_name=model_name)
    
    if step == max_step:
        file_name = f'./results/{model_name}/{dataset}_{method}{max_step}.jsonl'
    else:
        file_name = f'./results/{model_name}/r1_router_steps/{dataset}_{method}_step_{step+1}.jsonl'
    with open(file_name, 'w', encoding='utf-8') as output_file:
        for example in problems:
            if step != 0:
                rets = ['ter', 'tir', 'tar']
                
                node = example[f'plan_{step-1}']

                if node['query'].lower() == "none" or node['retriever'][0].lower() == 'none' or not node.get("query"):
                    answer = {"thought": "", "answer": ""}
                else:
                    image_path = None
                    ret_dict = {}
                    for ret in rets:
                        if node.get(ret):
                            ret_dict[ret] = node[ret][:k]
                            
                    if dataset == 'dyn_vqa':
                        image_path = os.path.join(dyn_vqa_root, f"{example['image_id']}.jpg")
                    elif dataset == 'infoseek':
                        image_path = os.path.join(infoseek_root, f"{example['image_id']}.jpg")
                    elif dataset == 'webqa':
                        image_path = os.path.join(webqa_root, f"{example['image_id']}.jpg")
                    
                    msgs = [{'role': 'user', 'content':[
                        {"type": "text", "text": f"According to the related information searched, `ter` means this info is from text retriever, `tir` means this info is from text image retriever, `tar` means this info is from table retriever:{convert_ret_dict_to_str(ret_dict)}\n\n Give me the answer(with the format <answer></answer>) to the Question: {node['query']}"}
                    ]}]

                    if dataset in ['infoseek', 'dyn_vqa', 'webqa']:
                        msgs[0]['content'].append({"type": "image", "image": image_path})

                    answer = get_answer(msgs, model_name)
                    print(f"[INFO] Answer: {answer}")
                    
                example[f"plan_{step-1}"][f"answer"] = answer
            
            msgs, question = origin_question_gen(dataset, example)
            
            stop_earlier = False
            
            for st in range(0, step):
                if example[f'plan_{st}'][f'query'].lower() in ["", "none"] or node['retriever'][0].lower() == 'none' or not node.get("query"):
                    stop_earlier = True
                    break

                ans = example[f'plan_{st}']['answer']['answer']
                msgs[0]['content'][0]['text'] += f"Sub-question{st+1}: {example[f'plan_{st}'][f'query']}\nAnswer{st+1}: {ans}\n"

            if step == max_step:
                msgs[0]['content'][0]['text'] += f"Based on the information above, give me the final answer of the origin question: {question}"
                
                answer = get_answer(msgs, model_name, True)
                
                example[f'llm_answer'] = answer

                correct_answer = example['answer']
                
                print(f"[INFO] Answer: {answer}\n[INFO] Correct Answer: {correct_answer}\n\n")
            elif stop_earlier:
                example[f'plan_{step}'] = {'query': 'none', 'retriever': ['none']}

            else:
                example[f'plan_{step}'] = update_example(msgs, method, dataset, model_name)
                if example[f'plan_{step}']['query'] in ['none', 'null'] or example[f'plan_{step}']['retriever'][0] == 'none':
                    example[f'plan_{step}']['query'] = 'none'
                    example[f'plan_{step}']['retriever'][0] = 'none'

            json.dump(convert_ndarray_to_list(example), output_file, ensure_ascii=False)
            output_file.write('\n')


def parse_arguments():
    parser = argparse.ArgumentParser(description="rag pipeline")
    parser.add_argument("--dataset_name", type=str, default="2wikimultihopqa")
    parser.add_argument("--method", type=str, default="r1-router")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--max_step", type=int, default=3)
    parser.add_argument("--model_name", type=str, default="r1-router")
    return parser.parse_args()


def main():
    args = parse_arguments()
    method = args.method
    dataset_name = args.dataset_name
    model_name = args.model_name

    print(f"[INFO] Generating answers with the following parameters:")
    print(f"[INFO] Method: {method}")
    print(f"[INFO] Dataset name: {dataset_name}")
    print(f"[INFO] Model name: {model_name}")
    initialize_llm(model_name)    
    
    if method == "r1-router":
        step = args.step
        max_step = args.max_step
        print(f"[INFO] Step: {step}")
        print(f"[INFO] Max step: {max_step}")

        r1_router(dataset_name, step, method, max_step, model_name)

if __name__ == "__main__":
    main()
    