import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from datasets import load_from_disk, list_datasets, load_dataset, disable_caching
from tqdm import tqdm
import pandas as pd
import argparse

from util import *
from ns_watermark import *
from soft_watermark import *

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

def main(args):
    # Load pretrained model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=args.model, padding_side='left', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=args.model, torch_dtype=torch.float16, local_files_only=True).cuda()
    model.eval()
    tokenizer.pad_token_id = model.config.pad_token_id 
    
    # Load datasets
    dataset = load_dataset(args.dataset, split="validation")
    dataset = dataset.map(lambda example: tokenizer(example["text"]))
    dataset.set_format(type="torch", columns=["input_ids"])
    my_dataset = MyDataset(dataset)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Split datasets
    train_set, val_set = torch.utils.data.random_split(my_dataset, [0.1, 0.9])
    
    if args.test:
        dataloader = DataLoader(val_set, batch_size=1, collate_fn=data_collator, shuffle=False)
    else:
        dataloader = DataLoader(train_set, batch_size=1, collate_fn=data_collator, shuffle=False)
        
    output_list, input_list, target_list = [], [], []
    watermark = NecessaryAndSufficientWatermark(gamma=args.gamma, z=4)
        
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):

            #if i > 10:
            #    break
            
            input_ids = data["input_ids"].to(model.device)
            attention_mask = data["attention_mask"].to(model.device)

            if args.method == "no_watermark":
                output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=100, num_beams=1, length_penalty=0.0)
                input_length = input_ids.shape[1]
                output_list.extend([output[i].cpu().tolist()[input_length:] for i in range(output.shape[0])])
                input_list.extend(input_ids[i].cpu().tolist() for i in range(input_ids.shape[0]))
                target_list.append(data["target_ids"][0].tolist())
                
            elif args.method == "ns_watermark":
                output = watermark.generate(model, input_ids, max_length=100, alpha=args.alpha)
                output_list.extend([output[i].cpu().tolist() for i in range(len(output))])
                input_list.extend(input_ids[i].cpu().tolist() for i in range(input_ids.shape[0]))                
                target_list.append(data["target_ids"][0].tolist())
                
            elif args.method == "soft_watermark":
                output = soft_watermark(model, input_ids, gamma=args.gamma, delta=args.delta, max_length=100, num_beams=1)
                input_length = input_ids.shape[1]
                output_list.extend([output[i].cpu().tolist() for i in range(len(output))])
                input_list.extend(input_ids[i].cpu().tolist() for i in range(input_ids.shape[0]))
                target_list.append(data["target_ids"][0].tolist())
            
    # save results
    new_output_list = [postprocess_output(output, model.config.eos_token_id) for output in output_list]
    new_input_list = [postprocess_input(output, model.config.bos_token_id) for output in input_list]
    df = pd.DataFrame(data={'index': list(range(len(output_list))), 'output_ids': new_output_list, 'input_ids': new_input_list, 'target_ids': target_list})
    df.to_csv(args.results, index=False, compression='gzip')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="/data1/c4/realnewslike", type=str,
                        help="path of where the dataset is stored.") 
    parser.add_argument('--results', default="/data1/watermark_results/llama2.pkl.gz", type=str,
                        help="path of where to save the results of the experiment.")
    parser.add_argument('--model', default="/data1/huggingface", type=str,
                        help="path of where the pre-trained model is stored.")
    parser.add_argument('--test', action='store_true',
                        help="if True, the test dataset is used.")
    parser.add_argument('--method', default="no_watermark", type=str,
                        help="{ns_watermark, ns_watermark,soft_watermark}.")
    parser.add_argument('--gamma', type=float,
                        help="the size of green words.")
    parser.add_argument('--delta', default=10.0, type=float,
                        help="the offset used in the soft_watermark.")
    parser.add_argument('--alpha', type=float,
                        help="the approximation rate used in ns_watermark.")
    args = parser.parse_args()

    torch.manual_seed(0)
    disable_caching()
    main(args)
