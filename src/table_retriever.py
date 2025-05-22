import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
import transformers
from transformers import AutoModel
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import sqlite3

def set_seed(seed : int):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) # multi-gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    transformers.set_seed(seed)


class BiEncoder(nn.Module):
    def __init__(self, question_enc, table_enc, question_tokenizer, table_tokenizer):
        super().__init__()
        self.question_enc = AutoModel.from_pretrained(question_enc)
        self.table_enc = AutoModel.from_pretrained(table_enc)

        self.question_enc.resize_token_embeddings(len(question_tokenizer))
        self.table_enc.resize_token_embeddings(len(table_tokenizer))
    
    def forward(self,question_tokenized, table_tokenized):
        question_enc_output = self.question_enc(**question_tokenized).pooler_output
        table_enc_output = self.table_enc(**table_tokenized).pooler_output
        return question_enc_output, table_enc_output


def get_table_dataloader(tokenizer, table_query, batch_size=512):

    question_max_len = 64
    def collate_fn(samples):
        questions = samples
        
        question_tokenized = tokenizer(
            questions, return_tensors = "pt", padding = "max_length", truncation = True, max_length = question_max_len
        )

        return {"question" : questions,
                "question_tokenized" : question_tokenized,}
    
    dataloader = DataLoader(
        table_query, collate_fn = collate_fn, batch_size = batch_size, num_workers = 4
    )

    return dataloader