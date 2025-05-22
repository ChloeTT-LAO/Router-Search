from FlagEmbedding import BGEM3FlagModel
import torch
import gc

model = BGEM3FlagModel('BAAI/bge-m3',
                       use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

def encode(input_passages, batch_size=4096):
    encode_list = model.encode(input_passages, 
                            batch_size=batch_size, 
                            max_length=256, 
                        )['dense_vecs'] 
    # Explicitly call the garbage collector
    gc.collect()
    torch.cuda.empty_cache()
    return encode_list
