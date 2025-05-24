import gc
import os
import re
import json
import faiss
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from ..table_retriever import set_seed, BiEncoder, get_table_dataloader
from UniIR.common.interactive_retriever import InteractiveRetriever

root2 = "./results/retrieve"
dataset = "hotpotqa"
step_root = "./results/{}/steps"
mrrag_root = "/path/to/results/grpo"
dataset_root = "/data3/pengchunyi/datasets"
file_path = {
    'openwikitqa': 'openwikitqa_valid.jsonl',
    'hotpotqa': 'hotpotqa-dev-kilt.jsonl',
    'dyn_vqa': 'DynVQA_en.202410.jsonl',
    'infoseek': 'infoseek_val_3000.jsonl',
}

wiki_index = None
index_root = "./wikipedia/index"

def wikidata_load():
    all_segments = []

    with (open('/path/to/wiki_extract.jsonl', 'r') as infile):
        for line in infile:
            data = json.loads(line.strip())
            all_segments.append(data['text'])

    print("[INFO] wiki passages extracted")
    return all_segments

def get_node(answer):
    thought_pattern = r"<think>(.*?)</think>"
    subquestion_pattern = r"<sub-question>(.*?)</sub-question>"
    retriever_pattern = r"<ret>(.*?)</ret>"

    thought = re.search(thought_pattern, answer, re.DOTALL).group(1)
    subquestion = re.search(subquestion_pattern, answer, re.DOTALL).group(1)
    retriever = re.search(retriever_pattern, answer, re.DOTALL).group(1)
    retriever_list = [item.strip() for item in retriever.split(',')]

    return {'thought': thought, 'query': subquestion, 'retriever': retriever_list}


def retrieve_single(query_embedding, k):
    query_embedding = np.array(query_embedding, dtype=np.float32)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    faiss.normalize_L2(query_embedding)
    distances, indices = wiki_index.search(query_embedding, k)

    return distances, indices


def query_retrieve_per_retriever(query, step):
    embed_q = {
        'tir': {
            'embed_query': [],
            'idx': [],
            'image_path': [],
        },
        'ter': {
            'embed_query': [],
            'idx': [],
        },
        'tar': {
            'embed_query': [],
            'idx': [],
        }
    }
    retriever_name = {
        f'text retriever': 'ter',
        f'text image retriever': 'tir',
        f'table retriever': 'tar',
    }
    for idx, q in enumerate(query):
        q['qid'] = idx
        if not q.get(f'output_{step - 1}'):
            continue
        node = get_node(q[f'output_{step - 1}'])
        if not node.get('retriever'):
            continue
        retrievers = node['retriever']
        que = node['query']
        for retriever in retrievers:
            if retriever.lower() == 'none' or que.lower() == 'none':
                break
            embed_q[retriever_name[retriever]]['embed_query'].append(que)
            embed_q[retriever_name[retriever]]['idx'].append(idx)
            if retriever_name[retriever] == 'tir':
                if q['dataset'] in ['infoseek', 'dyn_vqa']:
                    img_path = os.path.join(image_root[q['dataset']], f"{q['image_id']}.jpg")
                else:
                    img_path = None

                embed_q[retriever_name[retriever]]['image_path'].append(img_path)

    map_q = {}
    for ret in embed_q.keys():
        map_q[ret] = {q: i for i, q in enumerate(embed_q[ret]['idx'])}

    print("[INFO] Common Knowledge query embedding")
    embed_q['ter']['embed_query'] = encode(embed_q['ter']['embed_query'])

    print("[INFO] Text Image query adding")
    ti_retriever = InteractiveRetriever()
    tir = []
    for q, img_path in zip(embed_q['tir']['embed_query'], embed_q['tir']['image_path']):
        if img_path:
            tir.append(('image,text', q, img_path, 'text'))
        else:
            tir.append(('text', q, img_path, 'text'))

    if len(tir) > 0:
        ti_retriever.add_queries(tir)  # query_modality, query_txt, query_img_path, candidate_modality

    if len(tir) > 0:
        print("[INFO] Text Image Query Retrieving")
        retrieved_info = ti_retriever.retrieve(10, 10)
        embed_q['tir']['retrieved'] = []
        for info in retrieved_info:
            embed_q['tir']['retrieved'].append([text['txt'] for text in info])

    if len(embed_q['tar']['embed_query']) > 0:
        print("[INFO] Table Query Retrieving")
        embed_q['tar']['retrieved'] = table_retriever(embed_q['tar']['embed_query'])

    print("[INFO] Text Query Retrieving")
    embed_q['ter']['retrieved'] = retrieve_chunk(embed_q['ter']['embed_query'])

    for retriever in embed_q.keys():
        for idx, retrieved_idx in map_q[retriever].items():
            if not query[idx].get(f'ret_{step - 1}'):
                query[idx][f'ret_{step - 1}'] = {}
            query[idx][f'ret_{step - 1}'][retriever] = embed_q[retriever]['retrieved'][retrieved_idx]

    return query


def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(v) for v in obj]
    else:
        return obj

def table_retriever(table_query):
    encoder_name = "/path/to/bert-large-uncased"
    question_tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    table_tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    index = faiss.read_index("path/to/table/index/bert_table.index")
    accelerator = Accelerator(cpu=False, mixed_precision="fp16")
    set_seed(0)

    special_tokens = ["[Title]", "[Section title]", "[Caption]", "[Table name]", "[Header]", "[Rows]", "[Row]", "[sep]"]
    table_tokenizer.add_tokens(special_tokens)

    model = BiEncoder(encoder_name, encoder_name, question_tokenizer, table_tokenizer)
    model = model.to(accelerator.device)
    optimizer = AdamW(params=model.parameters(), lr=1e-5)
    model, optimizer = accelerator.prepare(model, optimizer)

    accelerator.load_state("/path/to/table/model")
    model = accelerator.unwrap_model(model)
    question_encoder = model.question_enc
    table_encoder = model.table_enc
    table_dataloader = get_table_dataloader(question_tokenizer, table_query)
    question_encoder, table_encoder, table_dataloader = accelerator.prepare(
        question_encoder, table_encoder, table_dataloader,
    )
    question_encoder.eval()
    table_encoder.eval()
    results = []
    tables = pd.read_json(os.path.join("/path/to/table/data/", "splitted_tables.json"))
    with tqdm(table_dataloader, desc=f"Table Data Retrieving") as batch_iterator:
        for batch in batch_iterator:
            question_tokenized = batch["question_tokenized"]
            num_batch = question_tokenized["input_ids"].size(0)
            question_tokenized = question_tokenized.to(accelerator.device)

            with torch.no_grad():
                output = question_encoder(**question_tokenized).pooler_output

            _, indicies = index.search(output.type(torch.float32).cpu().numpy().reshape(num_batch, -1), 10)

            results.extend([tables['flattened'][ind].reset_index(drop=True).values.tolist() for ind in indicies])

    return results


def retrieve_chunk(query_embeddings, batch_size=10, k=10):
    wiki_passages = wikidata_load()
    all_distance = []
    all_indeces = []
    retrieved_data = []

    for chunk in tqdm(range(1, 22), desc="Retrieving index"):
        index_cpu = faiss.read_index(f"/path/to/wikipedia/index/indexed_wiki_{chunk}.index")

        # Convert the CPU index to multiple GPU indices
        ngpus = faiss.get_num_gpus()
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True  # Use shard to divide the data across the GPUs
        global wiki_index
        wiki_index = faiss.index_cpu_to_all_gpus(index_cpu, co=co, ngpu=ngpus)  # This shards the index across all GPUs

        distance = []
        indices = []
        for i in range(0, len(query_embeddings), batch_size):
            batch = query_embeddings[i: i + batch_size]
            dis, ind = retrieve_single(batch, k)
            distance.append(dis)
            indices.append(ind)

        wiki_index = None
        all_distance.append(np.vstack(distance))
        all_indeces.append(np.vstack(indices))

    all_distance = np.hstack(all_distance)
    all_indeces = np.hstack(all_indeces)

    for i, distance in enumerate(all_distance):
        top_k_distance_idx = np.argsort(distance)[::-1][:k]
        retrieved_data.append([wiki_passages[indices] for indices in all_indeces[i][top_k_distance_idx]])

    del all_distance
    del all_indeces
    del wiki_passages

    gc.collect()
    torch.cuda.empty_cache()
    return retrieved_data


def parse_arguments():
    parser = argparse.ArgumentParser(description="retrieve pipeline")
    parser.add_argument("--mode", type=str, default="retrieve")
    parser.add_argument("--method", type=str, default="rag")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--max_step", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--corpus", type=str, default="query")
    parser.add_argument("--dataset_name", type=str, default="all")
    parser.add_argument("--model_name", type=str, default="qwen")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    mode = args.mode
    step = args.step
    method = args.method
    corpus = args.corpus
    k = args.k
    max_step = args.max_step
    dataset_name = args.dataset_name
    model_name = args.model_name

    root2 = root2.format(model_name)
    step_root = step_root.format(model_name)
    mrrag_root = mrrag_root.format(model_name)
    print(f"[INFO] Preprocessing corpus with the following parameters:")
    print(f"[INFO] Mode: {mode}")
    if dataset_name == 'all':
        dataset_name = ['2wikimultihopqa', 'openwikitqa', 'infoseek']
    else:
        dataset_name = [dataset_name]

    if mode == "retrieve":
        from ..bge_embedding import encode

        print(f"[INFO] Top-K: {k}")
        print(f"[INFO] Method: {method}\n")

        data = []
        data_dataset = {
            'openwikitqa': [],
            '2wikimultihopqa': [],
            'infoseek': [],
        }

        for dataset in dataset_name:
            file_name = f"{dataset}_{method}_step_{step}_grpo.jsonl"
            file_name = os.path.join(mrrag_root, file_name)
            retrieve_func = query_retrieve_per_retriever

            with open(file_name, 'r') as f:
                for line in f:
                    example = json.loads(line.strip())
                    example['dataset'] = dataset
                    data.append(example)

        data = retrieve_func(data, step)
        for l in data:
            dataset = l['dataset']
            l.pop('dataset', None)
            data_dataset[dataset].append(l)

        print("[INFO] writing jsonl")
        for dataset in dataset_name:
            output_name = f"{dataset}_retrieve_{method}_step_{step}.jsonl"
            with open(os.path.join(root2, output_name), 'w', encoding='utf-8') as o:
                for line in data_dataset[dataset]:
                    json.dump(convert_ndarray_to_list(line), o, ensure_ascii=False)
                    o.write('\n')
