import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from datasets import load_from_disk, list_datasets, load_dataset, disable_caching
from tqdm import tqdm
import pandas as pd
import argparse

from util import *
import ast
import math

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

def main(args):
    # Load pretrained model
    print("loading model")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=args.model, torch_dtype=torch.float16, local_files_only=True)
    model.eval()
        
    results = pd.read_csv(args.in_results)
    z_score_list = []
    
    with torch.no_grad():
        for i in tqdm(range(len(results))):
            if args.human:
                output_ids = ast.literal_eval(results["target_ids"][i])
            else:
                output_ids = ast.literal_eval(results["output_ids"][i])

            if len(output_ids) > 1:
                z = compute_z_score(torch.tensor([output_ids]), model.config.vocab_size, args.gamma, num_bos_tokens=0)
            else:
                z = math.inf
            z_score_list.append(z)

            
    # save results
    results["z_score"] = z_score_list
    results.to_csv(args.out_results, index=False, compression='gzip')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_results', default="results/c4/no_watermark.pkl.gz", type=str)
    parser.add_argument('--out_results', type=str)
    parser.add_argument('--model', default="/data1/takezawa/huggingface", type=str)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--human', action='store_true')    
    args = parser.parse_args()

    torch.manual_seed(0)
    disable_caching()
    main(args)
