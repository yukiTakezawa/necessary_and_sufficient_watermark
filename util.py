import torch
from transformers import T5Tokenizer, GPT2Tokenizer, AutoModelForCausalLM, pipeline, AutoTokenizer, T5ForConditionalGeneration
import hashlib
import random
import numpy as np
from tqdm import tqdm
import math
from datasets import load_dataset

def generate_mask(num_vocab, old_word_id, gamma=0.5):
    rng = torch.Generator()
    rng.manual_seed(old_word_id.item() % (2**32 - 1))
    uniform_tensor = torch.randn(num_vocab, generator=rng)
    return (uniform_tensor <= torch.sort(uniform_tensor)[0][int(num_vocab*gamma)-1]).float()


def compute_z_score(output_ids, num_vocab, gamma, num_bos_tokens):
    """
    num_bos_tokens (int):
        When the pre-trained model is NLLB-200, the first two words is BOS and FORCED_BOS_TOKEN_ID. So, we need to set num_bos_tokens=2.
        When the pre-trained model is LLaMA, ... 
    """
    
    counter = 0
    length = len(output_ids[0]) - 1 - num_bos_tokens # <pad>を除く
    
    for t in range(num_bos_tokens, len(output_ids[0])-1):
        #green_words, _ = split_words(num_vocab, output_ids[0][t], gamma)
        green_mask = generate_mask(num_vocab, output_ids[0][t], gamma)
        
        if green_mask[output_ids[0][t+1]] == 1.0:
            counter += 1

    return (counter - gamma*length) / math.sqrt(gamma * (1-gamma) * length)


def postprocess_output(output_ids, eos_token_id):
    """
    output_ids : list
    eos_token_id : int
    """
    
    if eos_token_id in output_ids:
        eos_index = output_ids.index(eos_token_id)
        return output_ids[:eos_index+1]
    else:
        return output_ids

    
def postprocess_input(input_ids, bos_token_id):
    """
    input_ids : list
    bos_token_id : int
    """
    
    bos_index = input_ids.index(bos_token_id)
    return input_ids[bos_index:]


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indexes = [i for i in range(len(dataset)) if len(dataset[i]["input_ids"]) < 4096]
        
    def __len__(self):
        return len(self.indexes)
        
        
    def __getitem__(self, idx):
        #return self.dataset[self.indexes[idx]]
        length = self.dataset[self.indexes[idx]]["input_ids"].shape[0]
        input_length = int(0.9 * length)
        #return {"input_ids": self.dataset[self.indexes[idx]]["input_ids"][:input_length]} #, "last_ids": self.dataset[self.indexes[idx]]["input_ids"][input_length:]}
        return {"input_ids": self.dataset[self.indexes[idx]]["input_ids"][:input_length], "target_ids": self.dataset[self.indexes[idx]]["input_ids"][input_length:]}


class WMT16_DeEn(torch.utils.data.Dataset):
    def __init__(self, tokenizer, path="/data1/takezawa/dataset/"):
        self.dataset = load_dataset("wmt16", "de-en", split="test", cache_dir=path)
        self.dataset = self.dataset.map(lambda example: {"input_ids": tokenizer(example["translation"]["de"])["input_ids"],
                                                    "target_ids": tokenizer(example["translation"]["en"])["input_ids"]})
        
        self.dataset.set_format(type="torch", columns=["input_ids", "target_ids"])

    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):        
        return {"input_ids": self.dataset[idx]["input_ids"], "target_ids": self.dataset[idx]["target_ids"]}


class WMT16_EnDe(torch.utils.data.Dataset):
    def __init__(self, tokenizer, path="/data1/takezawa/dataset/"):
        self.dataset = load_dataset("wmt16", "de-en", split="test", cache_dir=path)
        self.dataset = self.dataset.map(lambda example: {"input_ids": tokenizer(example["translation"]["en"])["input_ids"],
                                                    "target_ids": tokenizer(example["translation"]["de"])["input_ids"]})
        
        self.dataset.set_format(type="torch", columns=["input_ids", "target_ids"])

    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):        
        return {"input_ids": self.dataset[idx]["input_ids"], "target_ids": self.dataset[idx]["target_ids"]}


class WMT14_FrEn(torch.utils.data.Dataset):
    def __init__(self, tokenizer, path="/data1/takezawa/dataset/"):
        self.dataset = load_dataset("wmt14", "fr-en", split="test", cache_dir=path)
        self.dataset = self.dataset.map(lambda example: {"input_ids": tokenizer(example["translation"]["fr"])["input_ids"],
                                                    "target_ids": tokenizer(example["translation"]["en"])["input_ids"]})
        
        self.dataset.set_format(type="torch", columns=["input_ids", "target_ids"])

    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):        
        return {"input_ids": self.dataset[idx]["input_ids"], "target_ids": self.dataset[idx]["target_ids"]}


    
class WMT14_EnFr(torch.utils.data.Dataset):
    def __init__(self, tokenizer, path="/data1/takezawa/dataset/"):
        self.dataset = load_dataset("wmt14", "fr-en", split="test", cache_dir=path)
        self.dataset = self.dataset.map(lambda example: {"input_ids": tokenizer(example["translation"]["en"])["input_ids"],
                                                    "target_ids": tokenizer(example["translation"]["fr"])["input_ids"]})
        
        self.dataset.set_format(type="torch", columns=["input_ids", "target_ids"])

    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):        
        return {"input_ids": self.dataset[idx]["input_ids"], "target_ids": self.dataset[idx]["target_ids"]}
