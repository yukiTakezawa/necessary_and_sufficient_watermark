import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, AutoModelForSeq2SeqLM
from datasets import load_from_disk, list_datasets, load_dataset, disable_caching
from tqdm import tqdm
import pandas as pd
import argparse

from util import *
from ns_watermark_for_encoder_decoder import *
from soft_watermark import *

MODEL_NAME="facebook/nllb-200-3.3B"

def main(args):
    # Load datasets
    print("loading dataset")
    if args.language =="de-en":
        src_lang="deu_Latn"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=args.model, src_lang=src_lang, revision="1a07f7d195896b2114afcb79b7b57ab512e7b43e")
        my_dataset = my_dataset = WMT16_DeEn(tokenizer, args.dataset)
        forced_bos_token_id = tokenizer.lang_code_to_id["eng_Latn"]
    elif args.language =="en-de":
        src_lang="eng_Latn"        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=args.model, src_lang=src_lang, revision="1a07f7d195896b2114afcb79b7b57ab512e7b43e")
        my_dataset = my_dataset = WMT16_EnDe(tokenizer, args.dataset)
        forced_bos_token_id = tokenizer.lang_code_to_id["deu_Latn"]
    elif args.language =="fr-en":
        src_lang="fra_Latn"        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=args.model, src_lang=src_lang, revision="1a07f7d195896b2114afcb79b7b57ab512e7b43e")
        my_dataset = my_dataset = WMT14_FrEn(tokenizer, args.dataset)
        forced_bos_token_id = tokenizer.lang_code_to_id["eng_Latn"]
    elif args.language =="en-fr":
        src_lang="eng_Latn"        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=args.model, src_lang=src_lang, revision="1a07f7d195896b2114afcb79b7b57ab512e7b43e")
        my_dataset = my_dataset = WMT14_EnFr(tokenizer, args.dataset)
        forced_bos_token_id = tokenizer.lang_code_to_id["fra_Latn"]
    else:
        print("ERROR")

    # Load pretrained model
    print("loading model")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=args.model, torch_dtype=torch.float16, revision="1a07f7d195896b2114afcb79b7b57ab512e7b43e").cuda()
    model.eval()
    
        
    # Split datasets
    train_set, val_set = torch.utils.data.random_split(my_dataset, [0.1, 0.9])
    
    if args.test:
        dataloader = val_set
    else:
        dataloader = train_set        

    watermark = NecessaryAndSufficientWatermark_EncDec(gamma=args.gamma, z=4)
    output_list, input_list, target_list = [], [], []
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            input_ids = data["input_ids"].view(1, -1).to(model.device)
            target_ids = data["target_ids"]
            
            if args.method == "no_watermark":
                output = model.generate(input_ids, max_length=100, forced_bos_token_id=forced_bos_token_id, num_beams=1, length_penalty=0.0)
                
            elif args.method == "ns_watermark":
                # この解決策は微妙だが、max_lengthの意味がmodel.generate()と違う
                output = watermark.generate(model, input_ids, forced_bos_token_id=forced_bos_token_id, max_length=98, alpha=args.alpha, num_beams=1)

            elif args.method == "soft_watermark":
                # この解決策は微妙だが、max_lengthの意味がmodel.generate()と違う
                output = soft_watermark(model, input_ids, gamma=args.gamma, delta=args.delta, forced_bos_token_id=forced_bos_token_id, max_length=98, encoder_decoder_model=True, num_beams=1)

            elif args.method == "adaptive_soft_watermark":
                # この解決策は微妙だが、max_lengthの意味がmodel.generate()と違う
                output = adaptive_soft_watermark(model, input_ids, gamma=args.gamma, delta_list=[4,6,8,10,12], forced_bos_token_id=forced_bos_token_id, max_length=98, encoder_decoder_model=True, num_beams=1, num_bos_tokens=2)
                
            elif args.method == "adaptive_soft_watermark2":
                # この解決策は微妙だが、max_lengthの意味がmodel.generate()と違う
                output = adaptive_soft_watermark(model, input_ids, gamma=args.gamma, delta_list=[4,6,8,10,12,14], forced_bos_token_id=forced_bos_token_id, max_length=98, encoder_decoder_model=True, num_beams=1, num_bos_tokens=2)
                
            else:
                print("ERROR")
                

            output_list.append(output[0].tolist())
            input_list.append(input_ids[0].cpu().tolist())
            target_list.append(target_ids.tolist())

                         
    # save results
    new_output_list = [output for output in output_list]
    new_input_list = [output for output in input_list]
    df = pd.DataFrame(data={'index': list(range(len(output_list))), 'output_ids': new_output_list, 'input_ids': new_input_list, 'target_ids': target_list})
    df.to_csv(args.results, index=False, compression='gzip')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="/data1/takezawa/dataset/", type=str,
                        help="path of where the dataset is stored.") 
    parser.add_argument('--results', default="/data1/takezawa/wmt_de_en/no_watermark.pkl.gz", type=str,
                        help="path of where to save the results of the experiment.")
    parser.add_argument('--model', default="/data1/takezawa/huggingface", type=str,
                        help="path of where the pre-trained model is stored.")                        
    parser.add_argument('--language', default="de-en", type=str,
                        help="{en-de,de-en,fr-en,en-fr}")
    parser.add_argument('--test', action='store_true',
                        help="if True, the test dataset is used.")
    parser.add_argument('--method', default="no_watermark", type=str,
                        help="{ns_watermark, ns_watermark,soft_watermark}.")                        
    parser.add_argument('--gamma', default=0.05, type=float,
                        help="the size of green words.")                        
    parser.add_argument('--delta', default=10.0, type=float,
                        help="the offset used in the soft_watermark.")
    parser.add_argument('--alpha', default=math.inf, type=float,
                        help="the approximation rate used in ns_watermark.")                        
    args = parser.parse_args()

    torch.manual_seed(0)
    disable_caching()
    main(args)
