import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from datasets import load_from_disk, list_datasets, load_dataset, disable_caching
from tqdm import tqdm
import pandas as pd
import ast
import math

import argparse
from util import *

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

def main(args):
    # Load pretrained model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=args.model, padding_side='left', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=args.model, device_map="auto", torch_dtype=torch.float16, local_files_only=True)
    model.eval()
    tokenizer.pad_token_id = model.config.pad_token_id

    # Load results
    results = pd.read_csv(args.results)
    nlls = []
    
    with torch.no_grad():
        for i in tqdm(range(len(results))):
            input_ids = ast.literal_eval(results["input_ids"][i])
            output_ids = ast.literal_eval(results["output_ids"][i])
            
            prompt_length = len(input_ids)            
            input_ids =  input_ids + output_ids
            
            input_ids = torch.tensor(input_ids).view(1, -1).to(model.device)
            target_ids = input_ids.clone()
            target_ids[:, :prompt_length] = -100
            
            output = model(input_ids, labels=target_ids)
            nlls.append(output.loss.item())

    print(f"PPL: {math.exp(sum(nlls) / len(nlls))}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default="/data1/takezawa/watermark_results/llama2.pkl.gz", type=str,
                        help="path of where the results is stored.") 
    parser.add_argument('--model', default="/data1/takezawa/huggingface", type=str,
                        help="path of where the pre-trained model is stored.")
    parser.add_argument('--gamma', type=float,
                        help="the size of green words")
    args = parser.parse_args()

    torch.manual_seed(0)
    disable_caching()
    main(args)
