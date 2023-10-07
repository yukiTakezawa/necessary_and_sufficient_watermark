import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, AutoModelForSeq2SeqLM
from datasets import load_from_disk, list_datasets, load_dataset, disable_caching
from tqdm import tqdm
import pandas as pd
import argparse

from util import *
import ast

MODEL_NAME="facebook/nllb-200-3.3B"

def main(args):
    # Load pretrained model
    print("loading model")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=args.model, torch_dtype=torch.float16, revision="1a07f7d195896b2114afcb79b7b57ab512e7b43e")
    model.eval()
    
    results = pd.read_csv(args.in_results)
    z_score_list = []
    
    with torch.no_grad():
        for i in tqdm(range(len(results))):
            if args.human:
                output_ids = ast.literal_eval(results["target_ids"][i])
            else:
                output_ids = ast.literal_eval(results["output_ids"][i])

            if args.human:
                # the first two words in output_ids are code of output_language.
                z = compute_z_score(torch.tensor([output_ids]), model.config.vocab_size, args.gamma, num_bos_tokens=1)
            else:
                # the first two words in output_ids are BOS, and code of output_language.
                z = compute_z_score(torch.tensor([output_ids]), model.config.vocab_size, args.gamma, num_bos_tokens=2)
            z_score_list.append(z)

            
    # save results
    results["z_score"] = z_score_list
    results.to_csv(args.out_results, index=False, compression='gzip')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_results', default="/data1/takezawa/wmt_de_en/no_watermark.pkl.gz", type=str,
                        help="path of where the results are stored.")
    parser.add_argument('--out_results', type=str,
                        help="path of where the results are stored.")
    parser.add_argument('--model', default="/data/takezawa/huggingface", type=str,
                        help="path of where the model is stored.")
    parser.add_argument('--gamma', type=float,
                        help="the size of green words.")
    parser.add_argument('--human', action='store_true',
                        help="if true, compute z-scores of texts written by humans.")
    args = parser.parse_args()

    torch.manual_seed(0)
    disable_caching()
    main(args)
